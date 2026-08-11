# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import concurrent.futures
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Set

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    MetadataCache,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.pool_host import HostKVCache

logger = logging.getLogger(__name__)


class HiCacheGCS(HiCacheStorage):
    """
    HiCacheGCS provides high-performance Hierarchical KV Caching using Google Cloud Storage (GCS).
    
    Features:
    - Parallel multi-threaded download/upload using ThreadPoolExecutor for high throughput.
    - Fast in-memory MetadataCache to eliminate HTTP RTT head calls on local match queries.
    - MLA multi-rank aware backup skipping (prevents redundant storage operations across TP ranks).
    - Full support for page-oriented v1 and v2 APIs across hybrid pools (KV, MAMBA, SWA, etc.).
    """

    def __init__(
        self,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Optional[Any] = None,
    ):
        super().__init__()
        try:
            from google.cloud import storage
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage is required to use the HiCacheGCS backend. "
                "Please install it via `pip install google-cloud-storage`."
            ) from e

        self.storage_config = storage_config
        extra_config = storage_config.extra_config or {}

        # Resolve bucket name from extra_config or environment variable
        self.bucket_name = (
            extra_config.get("bucket_name")
            or os.getenv("SGLANG_HICACHE_GCS_BUCKET")
        )
        if not self.bucket_name:
            raise ValueError(
                "Missing bucket_name configuration for GCS backend. Please specify "
                "'bucket_name' in --hicache-storage-backend-extra-config or set "
                "the SGLANG_HICACHE_GCS_BUCKET environment variable."
            )

        self.prefix = extra_config.get("prefix", "sglang_hicache").strip("/")
        self.max_workers = int(extra_config.get("max_workers", 32))
        
        # Metadata cache positive lookup toggle & TTL
        self.enable_metadata_cache = bool(
            extra_config.get("enable_metadata_cache", True)
        )
        self.metadata_ttl = float(extra_config.get("metadata_ttl", 300.0))

        # Initialize GCS Client and Bucket
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)

        # Build rank / model suffix to isolate KV keys across models and non-MLA TP ranks
        tp_rank = storage_config.tp_rank
        tp_size = storage_config.tp_size
        pp_rank = storage_config.pp_rank
        pp_size = storage_config.pp_size
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        model_name = storage_config.model_name
        is_mla_model = storage_config.is_mla_model

        model_name_clean = "-".join(model_name.split("/")) if model_name else ""
        self.config_suffix = f"_{model_name_clean}" if model_name_clean else ""
        if not is_mla_model:
            self.config_suffix += f"_{tp_rank}_{tp_size}"
        if pp_size > 1:
            self.config_suffix += f"_{pp_size}_{pp_rank}"
        if attn_cp_size > 1:
            self.config_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="gcs_hicache_worker"
        )

        if self.enable_metadata_cache:
            self.metadata_cache = MetadataCache(self.metadata_ttl)
            self._scan_existing_blobs_to_metadata_cache()
        else:
            self.metadata_cache = None

        if mem_pool_host is not None:
            self.register_mem_pool_host(mem_pool_host)

        logger.info(
            f"Initialized HiCacheGCS backend: bucket={self.bucket_name}, "
            f"prefix={self.prefix}, max_workers={self.max_workers}, "
            f"metadata_cache={self.enable_metadata_cache}"
        )

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def _get_component_key(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        if component_name is None or component_name in ("__default__", PoolName.KV):
            return self._get_suffixed_key(key)
        return self._get_suffixed_key(f"{key}.{component_name}")

    def _get_blob_name(
        self, key: str, component_name: Optional[str] = None
    ) -> str:
        component_key = self._get_component_key(key, component_name)
        return f"{self.prefix}/{component_key}.bin"

    def _scan_existing_blobs_to_metadata_cache(self) -> None:
        """Scan existing blobs in GCS prefix matching this rank/model configuration."""
        try:
            prefix_search = f"{self.prefix}/"
            blobs = self.client.list_blobs(self.bucket, prefix=prefix_search)
            count = 0
            for blob in blobs:
                if not blob.name.endswith(".bin"):
                    continue
                # Extract key stem under prefix
                filename = blob.name[len(prefix_search) :]
                stem = filename[:-4]
                if stem.endswith(self.config_suffix):
                    self.metadata_cache.add(stem)
                    count += 1
            logger.info(
                f"HiCacheGCS: Pre-populated metadata cache with {count} existing KV blobs."
            )
        except Exception as e:
            logger.warning(f"Failed to scan GCS blobs into metadata cache: {e}")

    def get(
        self,
        key: str,
        target_location: Optional[torch.Tensor] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        blob_name = self._get_blob_name(key)
        try:
            blob = self.bucket.blob(blob_name)
            raw_bytes = blob.download_as_bytes()
            if target_location is not None:
                expected_size = target_location.numel() * target_location.element_size()
                if len(raw_bytes) != expected_size:
                    raise IOError(
                        f"GCS read size mismatch for {key}: got {len(raw_bytes)}, expected {expected_size}"
                    )
                buf = memoryview(
                    target_location.view(torch.uint8).contiguous().numpy()
                )
                buf[:len(raw_bytes)] = raw_bytes
                res_tensor = target_location
            else:
                res_tensor = torch.from_buffer(raw_bytes, dtype=torch.uint8)

            if self.metadata_cache is not None:
                stem = self._get_component_key(key)
                self.metadata_cache.add(stem)
            return res_tensor
        except Exception as e:
            if self.metadata_cache is not None:
                stem = self._get_component_key(key)
                self.metadata_cache.remove(stem)
            logger.debug(f"Failed to fetch {key} from HiCacheGCS storage: {e}")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[List[torch.Tensor]] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        locs = target_locations or [None] * len(keys)
        futures = [
            self.executor.submit(self.get, k, loc)
            for k, loc in zip(keys, locs)
        ]
        return [f.result() for f in futures]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        stem = self._get_component_key(key)
        if self.exists(key):
            return True

        val_tensor = value if value is not None else target_location
        if val_tensor is None:
            return False

        blob_name = self._get_blob_name(key)
        try:
            blob = self.bucket.blob(blob_name)
            raw_bytes = val_tensor.contiguous().view(torch.uint8).numpy().tobytes()
            blob.upload_from_string(
                raw_bytes, content_type="application/octet-stream"
            )
            if self.metadata_cache is not None:
                self.metadata_cache.add(stem)
            return True
        except Exception as e:
            logger.error(f"Failed to save key {key} to HiCacheGCS storage: {e}")
            if self.metadata_cache is not None:
                self.metadata_cache.remove(stem)
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[List[Any]] = None,
        target_locations: Optional[List[Any]] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        vals = values or target_locations or [None] * len(keys)
        futures = [
            self.executor.submit(self.set, k, val)
            for k, val in zip(keys, vals)
        ]
        return all(f.result() for f in futures)

    def exists(self, key: str) -> bool:
        stem = self._get_component_key(key)
        if self.metadata_cache is not None and self.metadata_cache.contains(stem):
            return True

        blob_name = self._get_blob_name(key)
        try:
            exists_flag = self.bucket.blob(blob_name).exists()
            if exists_flag and self.metadata_cache is not None:
                self.metadata_cache.add(stem)
            return exists_flag
        except Exception:
            return False

    def _collect_existing_component_keys(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
    ) -> Set[str]:
        target_stems = {self._get_component_key(key) for key in keys}
        for transfer in pool_transfers or []:
            for key in keys:
                target_stems.add(self._get_component_key(key, transfer.name))

        existing_stems = set()
        to_check = []
        for stem in target_stems:
            if self.metadata_cache is not None and self.metadata_cache.contains(stem):
                existing_stems.add(stem)
            else:
                to_check.append(stem)

        if to_check:
            def check_blob(stem: str) -> Optional[str]:
                blob_name = f"{self.prefix}/{stem}.bin"
                try:
                    if self.bucket.blob(blob_name).exists():
                        return stem
                except Exception:
                    pass
                return None

            futures = [self.executor.submit(check_blob, s) for s in to_check]
            for f in futures:
                found_stem = f.result()
                if found_stem:
                    existing_stems.add(found_stem)
                    if self.metadata_cache is not None:
                        self.metadata_cache.add(found_stem)

        return existing_stems

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        existing_stems = self._collect_existing_component_keys(keys, pool_transfers)

        def has_component(page_idx: int, name: str) -> bool:
            return self._get_component_key(keys[page_idx], name) in existing_stems

        kv_pages = next(
            (
                i
                for i in range(len(keys))
                if self._get_component_key(keys[i]) not in existing_stems
            ),
            len(keys),
        )

        hit_count: dict[str, int] = {PoolName.KV: kv_pages} if kv_pages else {}
        final_pages = kv_pages

        for transfer in pool_transfers or []:
            if final_pages == 0:
                break
            name = transfer.name
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = next(
                    (i for i in range(kv_pages) if not has_component(i, name)),
                    kv_pages,
                )
            else:  # trailing_pages
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                boundary = 0
                for prefix_len in range(kv_pages, 0, -1):
                    if all(
                        has_component(i, name)
                        for i in range(max(0, prefix_len - trailing), prefix_len)
                    ):
                        boundary = prefix_len
                        break
            if boundary:
                hit_count[name] = boundary
            final_pages = min(final_pages, boundary)

        return PoolTransferResult(final_pages, hit_count)

    def _log_key(self, pool_name: str, key: str) -> str:
        return key if pool_name == PoolName.KV else f"{key}.{pool_name}"

    def _read_page(self, pool_name: str, key: str, host_pool, page_offset: int) -> bool:
        storage_key = self._log_key(pool_name, key)
        data_page = self.get(storage_key, host_pool.get_dummy_flat_data_page())
        if data_page is None:
            return False
        host_pool.set_from_flat_data_page(page_offset, data_page)
        return True

    def _write_page(
        self, pool_name: str, key: str, host_pool, page_offset: int
    ) -> bool:
        storage_key = self._log_key(pool_name, key)
        data_page = host_pool.get_data_page(page_offset, flat=True)
        return self.set(storage_key, data_page)

    def _batch_io_v2(self, transfers: List[PoolTransfer], op_fn):
        results: dict[str, List[bool]] = {}
        for transfer in transfers:
            host_pool = self.registered_pools[transfer.name]
            keys = transfer.keys or []
            page_size = getattr(host_pool, "page_size", 1) or 1
            expected = len(keys) * page_size
            host_indices = transfer.host_indices

            if host_indices is None or host_indices.numel() != expected:
                logger.error(
                    "%s indices length mismatch for %s: expected %s, got %s",
                    op_fn.__name__,
                    transfer.name,
                    expected,
                    host_indices.numel() if host_indices is not None else 0,
                )
                results[transfer.name] = [False] * len(keys)
                continue

            futures = [
                self.executor.submit(
                    op_fn,
                    transfer.name,
                    key,
                    host_pool,
                    host_indices[i * page_size].item(),
                )
                for i, key in enumerate(keys)
            ]
            results[transfer.name] = [f.result() for f in futures]
        return results

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        return self._batch_io_v2(transfers, self._read_page)

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> dict[str, List[bool]]:
        # MLA protection: Only tp_rank == 0 initiates write-back to avoid redundant storage IO across ranks
        if (
            self.storage_config.is_mla_model
            and self.storage_config.tp_rank != 0
        ):
            return {
                transfer.name: [True] * len(transfer.keys or [])
                for transfer in transfers
            }
        return self._batch_io_v2(transfers, self._write_page)

    def clear(self) -> bool:
        try:
            prefix_search = f"{self.prefix}/"
            blobs = list(self.client.list_blobs(self.bucket, prefix=prefix_search))
            if blobs:
                def delete_blob(blob):
                    try:
                        blob.delete()
                    except Exception:
                        pass

                list(self.executor.map(delete_blob, blobs))
            if self.metadata_cache is not None:
                self.metadata_cache.clear()
            logger.info(f"Cleared all entries in HiCacheGCS storage prefix {self.prefix}.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheGCS storage: {e}")
            return False
