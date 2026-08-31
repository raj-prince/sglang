# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

import unittest
import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.gcs.hicache_gcs import HiCacheGCS


class MockHostKVCache:
    """Mock implementation of HostKVCache for unit testing."""

    def __init__(
        self,
        page_size: int = 16,
        page_bytes: int = 4096,
        num_pages: int = 100,
        dtype=torch.float32,
    ):
        self.page_size = page_size
        self.page_bytes = page_bytes
        self.dtype = dtype
        self.layout = "page_first"
        # Host memory backing buffer
        self.pages = [
            torch.full((page_bytes,), fill_value=i + 1, dtype=torch.uint8)
            for i in range(num_pages)
        ]

    def get_data_page(self, page_offset: int, flat: bool = True) -> torch.Tensor:
        idx = page_offset // self.page_size
        return self.pages[idx]

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros((self.page_bytes,), dtype=torch.uint8)

    def set_from_flat_data_page(self, page_offset: int, data_page: torch.Tensor):
        idx = page_offset // self.page_size
        if idx >= len(self.pages):
            self.pages.extend([torch.zeros((self.page_bytes,), dtype=torch.uint8) for _ in range(idx - len(self.pages) + 1)])
        self.pages[idx] = data_page.clone()


def create_test_config(
    *,
    is_mla_model: bool = False,
    tp_rank: int = 0,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    attn_cp_rank: int = 0,
    attn_cp_size: int = 1,
    extra_config: dict = None,
) -> HiCacheStorageConfig:
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test_org/test_model",
        extra_config=extra_config or {},
    )


class TestHiCacheGCS(unittest.TestCase):
    def setUp(self):
        # Use in-memory fsspec protocol for ultra-fast, isolated unit testing
        self.extra_config = {
            "protocol": "memory",
            "bucket": "test-rapid-bucket",
            "prefix": "test_kv",
            "num_workers": 8,
            "metadata_ttl": 60.0,
        }
        self.config = create_test_config(extra_config=self.extra_config)
        self.mock_pool = MockHostKVCache(page_size=16, page_bytes=1024, num_pages=50)

    def test_factory_creation(self):
        """Test backend instantiation via StorageBackendFactory."""
        backend = StorageBackendFactory.create_backend(
            "gcs", self.config, self.mock_pool
        )
        self.assertIsInstance(backend, HiCacheGCS)
        self.assertEqual(backend.bucket, "test-rapid-bucket")
        backend.close()

        backend_rapid = StorageBackendFactory.create_backend(
            "rapid_bucket", self.config, self.mock_pool
        )
        self.assertIsInstance(backend_rapid, HiCacheGCS)
        backend_rapid.close()

    def test_single_and_batch_get_set(self):
        """Test raw tensor get, set, exists, and batch operations."""
        backend = HiCacheGCS(self.config, self.mock_pool)

        tensor1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8)
        tensor2 = torch.tensor([6, 7, 8, 9, 10], dtype=torch.uint8)

        # 1. Single set & exists & get
        self.assertFalse(backend.exists("key1"))
        self.assertTrue(backend.set("key1", tensor1))
        self.assertTrue(backend.exists("key1"))

        target = torch.zeros(5, dtype=torch.uint8)
        ret = backend.get("key1", target)
        self.assertIsNotNone(ret)
        self.assertTrue(torch.equal(target, tensor1))

        # 2. Batch set & batch exists & batch get
        keys = ["batch_key_1", "batch_key_2"]
        values = [tensor1, tensor2]
        self.assertTrue(backend.batch_set(keys, values))
        self.assertEqual(backend.batch_exists(keys), 2)
        self.assertEqual(backend.batch_exists(["batch_key_1", "non_existent"]), 1)

        targets = [torch.zeros(5, dtype=torch.uint8), torch.zeros(5, dtype=torch.uint8)]
        res = backend.batch_get(keys, targets)
        self.assertEqual(len(res), 2)
        self.assertTrue(torch.equal(targets[0], tensor1))
        self.assertTrue(torch.equal(targets[1], tensor2))

        backend.close()

    def test_v2_batch_roundtrip(self):
        """Test the v2 batch interface: batch_set_v2, batch_exists_v2, and batch_get_v2."""
        backend = HiCacheGCS(self.config, self.mock_pool)
        backend.register_mem_host_pool_v2(self.mock_pool, PoolName.KV)

        # Setup 4 pages (offset 0, 16, 32, 48)
        keys = ["page_hash_0", "page_hash_1", "page_hash_2", "page_hash_3"]
        host_indices = torch.tensor([0, 16, 32, 48], dtype=torch.int64)

        transfer = PoolTransfer(
            name=PoolName.KV,
            host_indices=host_indices,
            keys=keys,
        )

        # 1. batch_set_v2
        set_result = backend.batch_set_v2([transfer])
        self.assertIn(str(PoolName.KV), set_result)
        self.assertEqual(set_result[str(PoolName.KV)], [True, True, True, True])

        # 2. batch_exists_v2
        exists_result = backend.batch_exists_v2(keys)
        self.assertEqual(exists_result.kv_hit_pages, 4)

        # Check partial prefix hit
        exists_partial = backend.batch_exists_v2(keys + ["unknown_key"])
        self.assertEqual(exists_partial.kv_hit_pages, 4)

        # 3. Modify memory pool and reload via batch_get_v2
        orig_data_0 = self.mock_pool.get_data_page(0).clone()
        # Wipe local pool page
        self.mock_pool.set_from_flat_data_page(0, torch.zeros_like(orig_data_0))
        self.assertFalse(torch.equal(self.mock_pool.get_data_page(0), orig_data_0))

        # Restore from GCS backend
        get_result = backend.batch_get_v2([transfer])
        self.assertEqual(get_result[str(PoolName.KV)], [True, True, True, True])
        self.assertTrue(torch.equal(self.mock_pool.get_data_page(0), orig_data_0))

        backend.close()

    def test_multi_pool_transfers(self):
        """Test auxiliary pools (e.g. DRAFT or MAMBA) with ALL_PAGES and TRAILING_PAGES policies."""
        backend = HiCacheGCS(self.config, self.mock_pool)
        draft_pool = MockHostKVCache(page_size=16, page_bytes=1024, num_pages=20)
        backend.register_mem_host_pool_v2(draft_pool, PoolName.DRAFT)

        kv_keys = ["k0", "k1", "k2", "k3"]
        draft_keys = ["k0", "k1", "k2", "k3"]

        # Write KV pool
        backend.batch_set_v2([
            PoolTransfer(name=PoolName.KV, host_indices=torch.tensor([0, 16, 32, 48]), keys=kv_keys)
        ])

        # Write Draft pool only for first 2 keys
        backend.batch_set_v2([
            PoolTransfer(name=PoolName.DRAFT, host_indices=torch.tensor([0, 16]), keys=["k0", "k1"])
        ])

        # ALL_PAGES policy: missing k2/k3 shrinks usable prefix to 2
        transfer_all = PoolTransfer(
            name=PoolName.DRAFT,
            hit_policy=PoolHitPolicy.ALL_PAGES,
            keys=draft_keys,
        )
        res_all = backend.batch_exists_v2(kv_keys, [transfer_all])
        self.assertEqual(res_all.kv_hit_pages, 2)

        # TRAILING_PAGES policy: only checking tail ["k0", "k1"]
        transfer_trailing = PoolTransfer(
            name=PoolName.DRAFT,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
            keys=["k0", "k1"],
        )
        res_trailing = backend.batch_exists_v2(["k0", "k1"], [transfer_trailing])
        self.assertEqual(res_trailing.kv_hit_pages, 2)

        backend.close()

    def test_mla_rank0_write_deduplication(self):
        """Test that for MLA models, non-zero TP ranks skip redundant writes."""
        cfg_rank0 = create_test_config(is_mla_model=True, tp_rank=0, tp_size=4, extra_config=self.extra_config)
        cfg_rank1 = create_test_config(is_mla_model=True, tp_rank=1, tp_size=4, extra_config=self.extra_config)

        backend_rank0 = HiCacheGCS(cfg_rank0, self.mock_pool)
        backend_rank1 = HiCacheGCS(cfg_rank1, self.mock_pool)

        test_tensor = torch.tensor([1, 2, 3], dtype=torch.uint8)

        # Rank 1 write should return True (no-op skipped) without actually uploading to GCS
        self.assertTrue(backend_rank1.set("mla_key_1", test_tensor))
        self.assertFalse(backend_rank1.fs.exists(backend_rank1._get_object_path("mla_key_1")))

        # Rank 0 write should physically persist to GCS
        self.assertTrue(backend_rank0.set("mla_key_1", test_tensor))
        self.assertTrue(backend_rank0.fs.exists(backend_rank0._get_object_path("mla_key_1")))

        backend_rank0.close()
        backend_rank1.close()

    def test_clear(self):
        """Test clearing storage namespace."""
        backend = HiCacheGCS(self.config, self.mock_pool)
        backend.set("clear_test_key", torch.tensor([1, 2, 3], dtype=torch.uint8))
        self.assertTrue(backend.exists("clear_test_key"))

        self.assertTrue(backend.clear())
        self.assertFalse(backend.exists("clear_test_key"))
        backend.close()


if __name__ == "__main__":
    unittest.main()
