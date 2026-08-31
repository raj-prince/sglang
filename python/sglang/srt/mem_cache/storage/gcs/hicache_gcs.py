# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import concurrent.futures
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    MetadataCache,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)


def _get_env(key: str, default: Any = None) -> Any:
    """Safe lookup for SGLang environment variables with fallback to os.environ."""
    try:
        from sglang.srt.environ import envs
        if hasattr(envs, key):
            val = getattr(envs, key).get()
            if val is not None:
                return val
    except Exception:
        pass
    return os.environ.get(key, default)


class HiCacheGCS(HiCacheStorage):
    """Google Cloud Storage (GCS) Rapid Bucket KV-Cache Offload Storage Backend.

    This storage backend integrates with GCS Rapid Buckets and standard GCS buckets
    via fsspec and gcsfs to offload and prefetch hierarchical KV-cache blocks.
    It provides high-throughput concurrent I/O, in-memory metadata caching for low-latency
    prefix matching, zero-copy buffer staging, and multi-pool support.
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Optional[HostKVCache] = None,
    ):
        self.storage_config = storage_config
        self.registered_pools: Dict[str, HostKVCache] = {}

        if mem_pool_host is not None:
            self.register_mem_pool_host(mem_pool_host)

        extra_cfg = storage_config.extra_config or {}

        # 1. Parse bucket and path prefix
        storage_path = extra_cfg.get("storage_path") or extra_cfg.get("bucket")
        if storage_path is None:
            storage_path = _get_env("SGLANG_HICACHE_GCS_BUCKET", "sglang-rapid-kv")

        if storage_path.startswith("gs://"):
            storage_path = storage_path[5:]
        elif storage_path.startswith("gcs://"):
            storage_path = storage_path[6:]

        parts = storage_path.strip("/").split("/", 1)
        self.bucket = parts[0]
        base_prefix = parts[1] if len(parts) > 1 else ""

        prefix_param = extra_cfg.get("prefix")
        if prefix_param is None:
            prefix_param = _get_env("SGLANG_HICACHE_GCS_PREFIX", "")

        if prefix_param:
            self.prefix = f"{base_prefix}/{prefix_param}".strip("/") if base_prefix else prefix_param.strip("/")
        else:
            self.prefix = base_prefix

        # 2. Worker concurrency & endpoints
        num_workers_raw = extra_cfg.get("num_workers")
        if num_workers_raw is None:
            num_workers_raw = _get_env("SGLANG_HICACHE_GCS_NUM_WORKERS", 64)
        self.num_workers = int(num_workers_raw) if num_workers_raw is not None else 64

        self.endpoint_url = extra_cfg.get("endpoint_url") or _get_env("SGLANG_HICACHE_GCS_ENDPOINT_URL", None)

        # 3. Model & Distributed Rank namespace
        tp_rank = storage_config.tp_rank
        tp_size = storage_config.tp_size
        pp_rank = storage_config.pp_rank
        pp_size = storage_config.pp_size
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        is_mla_model = storage_config.is_mla_model
        model_name = "-".join(storage_config.model_name.split("/")) if storage_config.model_name else "default_model"

        # Construct unique rank identifier to prevent cross-worker collisions
        rank_suffix = f"{model_name}"
        if not is_mla_model:
            rank_suffix += f"_tp{tp_rank}_{tp_size}"
        if pp_size > 1:
            rank_suffix += f"_pp{pp_rank}_{pp_size}"
        if attn_cp_size > 1:
            rank_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"

        self.rank_suffix = rank_suffix
        if self.prefix:
            self.namespace_dir = f"{self.bucket}/{self.prefix}/{rank_suffix}"
        else:
            self.namespace_dir = f"{self.bucket}/{rank_suffix}"

        # 4. Initialize filesystem via fsspec
        self._init_fs(extra_cfg)

        # 5. ThreadPool for parallel async transfers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="gcs_rapid_io",
        )

        # 6. In-memory MetadataCache for sub-millisecond prefix hits
        enable_metadata_cache = extra_cfg.get("enable_metadata_cache")
        if enable_metadata_cache is None:
            enable_metadata_cache = _get_env("SGLANG_HICACHE_GCS_ENABLE_METADATA_CACHE", True)
        self.enable_metadata_cache = bool(enable_metadata_cache) if enable_metadata_cache is not None else True

        if self.enable_metadata_cache:
            ttl_raw = extra_cfg.get("metadata_ttl")
            if ttl_raw is None:
                ttl_raw = _get_env("SGLANG_HICACHE_GCS_METADATA_TTL", 60.0)
            self.metadata_ttl = float(ttl_raw) if ttl_raw is not None else 60.0
            self.metadata_cache = MetadataCache(self.metadata_ttl)
        else:
            self.metadata_cache = None

        logger.info(
            f"Initialized HiCacheGCS (bucket={self.bucket}, namespace={self.namespace_dir}, "
            f"workers={self.num_workers}, metadata_cache={self.enable_metadata_cache})"
        )

    def _init_fs(self, extra_cfg: Dict[str, Any]):
        """Initialize fsspec filesystem supporting GCS, mock memory FS, or custom instance."""
        if "fs" in extra_cfg and extra_cfg["fs"] is not None:
            self.fs = extra_cfg["fs"]
            return

        import fsspec

        protocol = extra_cfg.get("protocol", "gcs")
        target_protocol = extra_cfg.get("target_protocol", "gcs" if protocol in ("simplecache", "filecache", "blockcache") else protocol)
        cache_storage = extra_cfg.get("cache_storage") or _get_env("SGLANG_HICACHE_GCS_CACHE_STORAGE", None)

        fs_kwargs: Dict[str, Any] = {}
        if "project" in extra_cfg:
            fs_kwargs["project"] = extra_cfg["project"]
        if "token" in extra_cfg:
            fs_kwargs["token"] = extra_cfg["token"]
        if self.endpoint_url:
            fs_kwargs["endpoint_url"] = self.endpoint_url
        if "storage_options" in extra_cfg and isinstance(extra_cfg["storage_options"], dict):
            fs_kwargs.update(extra_cfg["storage_options"])

        if target_protocol in ("gcs", "gs") or protocol in ("gcs", "gs"):
            try:
                import gcsfs  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "Package 'gcsfs' is required for the GCS storage backend. "
                    "Please install it via 'pip install gcsfs'."
                ) from e

        if cache_storage is not None or protocol in ("simplecache", "filecache", "blockcache"):
            cache_type = protocol if protocol in ("simplecache", "filecache", "blockcache") else "simplecache"
            cache_kwargs = {
                "target_protocol": target_protocol,
                "target_options": fs_kwargs,
                "cache_storage": cache_storage or "/tmp/sglang_kv_cache",
            }
            if "cache_expiry_time" in extra_cfg:
                cache_kwargs["expiry_time"] = extra_cfg["cache_expiry_time"]
            if "cache_max_size" in extra_cfg:
                cache_kwargs["max_size"] = extra_cfg["cache_max_size"]
            self.fs = fsspec.filesystem(cache_type, **cache_kwargs)
            logger.info(f"Initialized fsspec local SSD cache layer ({cache_type}) at {cache_kwargs['cache_storage']} -> {target_protocol}")
        else:
            self.fs = fsspec.filesystem(protocol, **fs_kwargs)

    def _get_object_path(self, key: str, pool_name: str = PoolName.KV) -> str:
        """Derive object storage path for a key and cache pool."""
        if pool_name in ("__default__", PoolName.KV, str(PoolName.KV)):
            return f"{self.namespace_dir}/{key}.bin"
        return f"{self.namespace_dir}/{pool_name}/{key}.bin"

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        super().register_mem_pool_host(mem_pool_host)
        self.register_mem_host_pool_v2(mem_pool_host, PoolName.KV)

    def register_mem_host_pool_v2(self, host_pool: HostKVCache, host_pool_name: str):
        if not hasattr(self, "registered_pools"):
            self.registered_pools = {}
        self.registered_pools[str(host_pool_name)] = host_pool

    def exists(self, key: str, pool_name: str = PoolName.KV) -> bool:
        """Check if an object exists in GCS / local metadata cache."""
        path = self._get_object_path(key, pool_name)
        if self.metadata_cache is not None and self.metadata_cache.contains(path):
            return True

        try:
            exists = bool(self.fs.exists(path))
            if exists and self.metadata_cache is not None:
                self.metadata_cache.add(path)
            return exists
        except Exception as e:
            logger.debug(f"Failed to check existence of {path}: {e}")
            return False

    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """Check consecutive existing keys from the start."""
        for i, key in enumerate(keys):
            if not self.exists(key):
                return i
        return len(keys)

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        """Batch check existence with per-pool hit policies."""
        # 1. Longest contiguous KV prefix check
        kv_pages = 0
        for key in keys:
            if not self.exists(key, PoolName.KV):
                break
            kv_pages += 1

        hit_count: Dict[str, int] = {str(PoolName.KV): kv_pages} if kv_pages else {}
        final_pages = kv_pages

        # 2. Secondary pools (Mamba, SWA, Draft, DSA)
        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = str(transfer.name)
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not self.exists(keys[i], name)),
                    kv_pages,
                )
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        self.exists(keys[i], name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _read_single_page(
        self, pool_name: str, key: str, host_pool: HostKVCache, page_offset: int
    ) -> bool:
        """Read a single serialized page from GCS directly into host memory pool."""
        path = self._get_object_path(key, pool_name)
        try:
            data_page = host_pool.get_dummy_flat_data_page()
            buf = memoryview(data_page.view(torch.uint8).contiguous().numpy())
            with self.fs.open(path, "rb") as f:
                bytes_read = f.readinto(buf)
                if bytes_read != len(buf):
                    logger.warning(
                        f"Short read for {path}: expected {len(buf)} bytes, got {bytes_read}"
                    )
                    return False
            host_pool.set_from_flat_data_page(page_offset, data_page)
            if self.metadata_cache is not None:
                self.metadata_cache.add(path)
            return True
        except Exception as e:
            logger.debug(f"Failed to read {path} from GCS: {e}")
            if self.metadata_cache is not None:
                self.metadata_cache.remove(path)
            return False

    def _write_single_page(
        self, pool_name: str, key: str, host_pool: HostKVCache, page_offset: int
    ) -> bool:
        """Write a single page from host memory pool into GCS Rapid Bucket."""
        # MLA Rank-0 write deduplication
        if (
            self.storage_config.is_mla_model
            and self.storage_config.tp_rank != 0
            and pool_name in (PoolName.KV, str(PoolName.KV))
        ):
            return True

        path = self._get_object_path(key, pool_name)
        try:
            data_page = host_pool.get_data_page(page_offset, flat=True)
            data_bytes = data_page.contiguous().view(dtype=torch.uint8).numpy().tobytes()
            with self.fs.open(path, "wb") as f:
                f.write(data_bytes)
            if self.metadata_cache is not None:
                self.metadata_cache.add(path)
            return True
        except Exception as e:
            logger.error(f"Failed to write {path} to GCS: {e}")
            if self.metadata_cache is not None:
                self.metadata_cache.remove(path)
            return False

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        """Concurrent batch read from GCS Rapid Bucket into host memory."""
        results: Dict[str, List[bool]] = {}
        for transfer in transfers:
            pool_name = str(transfer.name)
            host_pool = self.registered_pools.get(pool_name)
            if host_pool is None:
                host_pool = getattr(self, "mem_pool_host", None)
            if host_pool is None:
                logger.error(f"Host pool not registered for {pool_name}")
                results[pool_name] = [False] * len(transfer.keys or [])
                continue

            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() not in (len(keys), len(keys) * page_size):
                logger.error(
                    f"batch_get_v2 indices length mismatch for {pool_name}: "
                    f"expected {len(keys)} or {len(keys) * page_size}, got {host_indices.numel() if host_indices is not None else 0}"
                )
                results[pool_name] = [False] * len(keys)
                continue

            stride = page_size if host_indices.numel() == len(keys) * page_size else 1

            futures = [
                self.executor.submit(
                    self._read_single_page,
                    pool_name,
                    key,
                    host_pool,
                    host_indices[i * stride].item(),
                )
                for i, key in enumerate(keys)
            ]
            results[pool_name] = [f.result() for f in futures]
        return results

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        """Concurrent batch write from host memory into GCS Rapid Bucket."""
        results: Dict[str, List[bool]] = {}
        for transfer in transfers:
            pool_name = str(transfer.name)
            host_pool = self.registered_pools.get(pool_name)
            if host_pool is None:
                host_pool = getattr(self, "mem_pool_host", None)
            if host_pool is None:
                logger.error(f"Host pool not registered for {pool_name}")
                results[pool_name] = [False] * len(transfer.keys or [])
                continue

            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() not in (len(keys), len(keys) * page_size):
                logger.error(
                    f"batch_set_v2 indices length mismatch for {pool_name}: "
                    f"expected {len(keys)} or {len(keys) * page_size}, got {host_indices.numel() if host_indices is not None else 0}"
                )
                results[pool_name] = [False] * len(keys)
                continue

            stride = page_size if host_indices.numel() == len(keys) * page_size else 1

            futures = [
                self.executor.submit(
                    self._write_single_page,
                    pool_name,
                    key,
                    host_pool,
                    host_indices[i * stride].item(),
                )
                for i, key in enumerate(keys)
            ]
            results[pool_name] = [f.result() for f in futures]
        return results

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        """Retrieve raw page tensor associated with key."""
        path = self._get_object_path(key, PoolName.KV)
        try:
            if target_location is not None:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                with self.fs.open(path, "rb") as f:
                    bytes_read = f.readinto(buf)
                    if bytes_read != len(buf):
                        return None
                if self.metadata_cache is not None:
                    self.metadata_cache.add(path)
                return target_location
            else:
                with self.fs.open(path, "rb") as f:
                    content = f.read()
                tensor = torch.frombuffer(content, dtype=torch.uint8)
                if self.metadata_cache is not None:
                    self.metadata_cache.add(path)
                return tensor
        except Exception as e:
            logger.debug(f"Failed to get {path}: {e}")
            if self.metadata_cache is not None:
                self.metadata_cache.remove(path)
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[Optional[torch.Tensor]]:
        if target_locations is None:
            target_locations = [None] * len(keys)
        futures = [
            self.executor.submit(self.get, key, loc, target_sizes)
            for key, loc in zip(keys, target_locations)
        ]
        return [f.result() for f in futures]

    def set(
        self,
        key: str,
        value: Optional[torch.Tensor] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """Store raw page tensor to GCS Rapid Bucket."""
        if value is None:
            return False

        if (
            self.storage_config.is_mla_model
            and self.storage_config.tp_rank != 0
        ):
            return True

        path = self._get_object_path(key, PoolName.KV)
        try:
            data_bytes = value.contiguous().view(dtype=torch.uint8).numpy().tobytes()
            with self.fs.open(path, "wb") as f:
                f.write(data_bytes)
            if self.metadata_cache is not None:
                self.metadata_cache.add(path)
            return True
        except Exception as e:
            logger.error(f"Failed to set {path}: {e}")
            if self.metadata_cache is not None:
                self.metadata_cache.remove(path)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[torch.Tensor]] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if values is None or len(keys) != len(values):
            return False
        futures = [
            self.executor.submit(self.set, key, val)
            for key, val in zip(keys, values)
        ]
        results = [f.result() for f in futures]
        return all(results)

    def clear(self) -> bool:
        """Clear all stored entries under the current namespace in GCS."""
        try:
            if self.fs.exists(self.namespace_dir):
                self.fs.rm(self.namespace_dir, recursive=True)
            if self.metadata_cache is not None:
                self.metadata_cache.clear()
            logger.info(f"Cleared GCS namespace: {self.namespace_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear GCS namespace {self.namespace_dir}: {e}")
            return False

    def close(self):
        """Shutdown executor thread pool."""
        self.executor.shutdown(wait=False)
