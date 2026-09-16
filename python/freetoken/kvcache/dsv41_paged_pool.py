"""Source-only paged KV storage for DeepSeek-V4.1 ratio-1/2 CSA2."""

from __future__ import annotations

import torch

from freetoken.utils import init_logger

from .dsv4_paged_pool import CompressStateRing, DSV4PagedKVCache


logger = init_logger(__name__)


class DSV41PagedKVCache(DSV4PagedKVCache):
    """Reuse DSV4's full/window address currency with V4.1 source ownership."""

    @staticmethod
    def _args(config):
        args = getattr(config.model_config, "dsv41_args", None)
        if args is None:
            raise ValueError("V4.1 cache requires ModelConfig.dsv41_args")
        return args

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .dsv4_cost_model import _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .dsv41_cost_model import dsv41_auto_cost_model

        args = cls._args(config)
        P = int(args.window_size)
        floor = _dsv4_window_floor_pages(config, P)
        per_page, fixed, reserve = dsv41_auto_cost_model(
            args,
            _dsv4_swa_ratio(config),
            floor,
            P=P,
            n_scratch=config.max_running_req + 1,
        )
        return per_page, fixed, P, reserve

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        from freetoken.utils import mem_GB

        from .dsv4_cost_model import _dsv4_swa_ratio, _dsv4_window_floor_pages
        from .dsv41_cost_model import (
            _dsv41_pool_sizes,
            dsv41_pool_bytes,
            dsv41_solve_num_pages,
        )

        args = cls._args(config)
        P = int(args.window_size)
        num_pages = config.num_page_override
        if num_pages is None:
            sizes = dsv41_solve_num_pages(
                available_memory,
                args,
                _dsv4_swa_ratio(config),
                floor_win_pages=_dsv4_window_floor_pages(config, P),
                P=P,
                n_scratch=config.max_running_req + 1,
            )
            num_pages = sizes.full_token // P - 1
        else:
            floor = _dsv4_window_floor_pages(config, P)
            if num_pages < floor:
                raise ValueError(
                    f"--num-pages {num_pages} ({num_pages * P} tokens) is below the "
                    f"DSV4.1 window working-set floor {floor} pages ({floor * P} tokens)"
                )
            sizes = _dsv41_pool_sizes(config, num_pages + 1)
        if num_pages <= 1:
            raise ValueError("Not enough memory for the V4.1 KV cache")
        real = dsv41_pool_bytes(sizes, args, config.max_running_req + 1)
        logger.info(
            f"Resolved {num_pages * P}-token DSV4.1 KV geometry "
            f"({sizes.n_win_pages} window pages); authority payload = {mem_GB(real)}"
        )
        return int(num_pages)

    @classmethod
    def allocation_bytes(cls, config, num_pages: int) -> int:
        """Byte-exact V4.1 allocation for an explicit usable-page target."""
        from .dsv41_cost_model import _dsv41_pool_sizes, dsv41_pool_bytes

        sizes = _dsv41_pool_sizes(config, int(num_pages) + 1)
        return dsv41_pool_bytes(
            sizes, cls._args(config), config.max_running_req + 1
        )

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        from .dsv4_cost_model import _dsv4_window_floor_pages

        args = cls._args(config)
        return _dsv4_window_floor_pages(config, args.window_size) * args.window_size

    def validate_rebuild(
        self,
        config,
        *,
        num_pages: int | None,
        target_moe: int,
        per_expert_bytes: int,
        baseline_free: int,
        weights_bytes: int,
        current_num_pages: int,
        extra_fixed_bytes: int = 0,
        extra_note: str = "",
        num_swa_pages: int | None = None,
        **targets,
    ) -> None:
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from freetoken.utils import mem_GB

        from .base import CacheRebuildRejected
        from .dsv4_cost_model import _dsv4_window_floor_pages
        from .dsv41_cost_model import _dsv41_pool_sizes, dsv41_pool_bytes

        args = self._args(config)
        if num_pages is not None:
            floor = _dsv4_window_floor_pages(config, args.window_size)
            if num_pages < floor:
                raise CacheRebuildRejected(
                    f"num_pages {num_pages} is below the DSV4.1 window working-set "
                    f"floor {floor} (max_running_req={config.max_running_req})"
                )
        if num_pages is not None or num_swa_pages is not None:
            target_pages = num_pages if num_pages is not None else current_num_pages
            sizes = _dsv41_pool_sizes(
                config, target_pages + 1, num_swa_pages=num_swa_pages
            )
        else:
            sizes = self.sizes
        budget = net_cache_budget_bytes(
            config.memory_ratio, baseline_free, weights_bytes, 0
        )
        need = target_moe * per_expert_bytes + dsv41_pool_bytes(
            sizes, args, config.max_running_req + 1
        )
        if need > budget:
            raise CacheRebuildRejected(
                f"requested DSV4.1 cache needs {mem_GB(need)} > budget "
                f"{mem_GB(budget)}; old cache kept, still serving"
            )

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        from .dsv41_cost_model import _dsv41_pool_sizes

        self.rebuild(
            _dsv41_pool_sizes(
                config, num_pages + 1, num_swa_pages=num_swa_pages
            )
        )

    def _alloc_buffers(self) -> None:
        sizes, device, dtype = self.sizes, self._device, self._dtype
        self.full_to_window = torch.full(
            (sizes.full_token + 1,), -1, dtype=torch.int64, device=device
        )
        if not hasattr(self, "full_loc_map"):
            self.full_loc_map: torch.Tensor | None = None
        self.window_pool = [
            torch.zeros(sizes.n_win_slots, self.head_dim, device=device, dtype=dtype)
            for _ in range(self._n_layers)
        ]
        self.cmp_pool: list[torch.Tensor | None] = [None] * self._n_layers
        self.idx_pool: list[torch.Tensor | None] = [None] * self._n_layers
        self.state_ring: list[CompressStateRing | None] = [None] * self._n_layers
        self.indexer_state_ring: list[CompressStateRing | None] = [None] * self._n_layers
        self.cmp_scratch_base: list[int | None] = [None] * self._n_layers
        self.idx_scratch_base: list[int | None] = [None] * self._n_layers
        for layer in self.args.kv_source_layers:
            ratio = self.compress_ratios[layer]
            if ratio not in (1, 2):
                raise ValueError(f"V4.1 source layer {layer} has ratio {ratio}")
            self.cmp_scratch_base[layer] = sizes.cmp_blocks[layer]
            self.idx_scratch_base[layer] = sizes.idx_blocks[layer]
            self.cmp_pool[layer] = torch.zeros(
                sizes.cmp_blocks[layer] + self.n_scratch,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            self.idx_pool[layer] = torch.zeros(
                sizes.idx_blocks[layer] + self.n_scratch,
                self.index_head_dim,
                device=device,
                dtype=dtype,
            )
            if ratio == 2:
                self.state_ring[layer] = CompressStateRing(
                    n_slots=sizes.state_slots[layer],
                    ring_size=2,
                    overlap=False,
                    head_dim=self.head_dim,
                    device=device,
                )

    def source_layer_of(self, layer_id: int) -> int:
        ratio = self.compress_ratios[layer_id]
        candidates = [
            source
            for source in self.args.kv_source_layers
            if source <= layer_id and self.compress_ratios[source] == ratio
        ]
        if not candidates:
            raise ValueError(f"layer {layer_id} has no V4.1 KV source")
        return max(candidates)

    def compressed_pool_for(self, layer_id: int) -> torch.Tensor:
        source = self.source_layer_of(layer_id)
        result = self.cmp_pool[source]
        assert result is not None
        return result

    def index_pool_for(self, layer_id: int) -> torch.Tensor:
        source = self.source_layer_of(layer_id)
        result = self.idx_pool[source]
        assert result is not None
        return result

    def ring_size(self, layer_id: int) -> int:
        if self.compress_ratios[layer_id] != 2:
            raise ValueError(f"layer {layer_id} has no ratio-2 pending-pair ring")
        return 2

    def unit_bytes(self) -> tuple[int, int]:
        from .dsv41_cost_model import dsv41_kv_unit_bytes, dsv41_window_unit_bytes

        return dsv41_kv_unit_bytes(self.args, self.P), dsv41_window_unit_bytes(
            self.args, self.P
        )


class DSV41ExpertWorkerKVCache(DSV41PagedKVCache):
    """Logical DSV4.1 page/SWA state for an authority-EP expert-only rank.

    Every EP scheduler must make byte-identical admission, prefix-cache, and page-allocation
    decisions.  Expert workers therefore retain the small full->window address map and the
    window free-list, but they never execute attention and do not need window/cmp/idx/state
    payload tensors.  Keeping this as the same pool duck type lets the generic CacheManager run
    unchanged while avoiding a full 256K attention-cache replica on every expert-only GPU.
    """

    needs_rebind_on_rebuild = False

    def _alloc_buffers(self) -> None:
        sizes, device = self.sizes, self._device
        self.full_to_window = torch.full(
            (sizes.full_token + 1,), -1, dtype=torch.int64, device=device
        )
        if not hasattr(self, "full_loc_map"):
            self.full_loc_map: torch.Tensor | None = None

        # Preserve the shape-bearing attributes used by rebuild/telemetry, but allocate no
        # attention payload.  Any accidental attention access below fails loudly.
        self.window_pool: list[torch.Tensor] = []
        self.cmp_pool: list[torch.Tensor | None] = [None] * self._n_layers
        self.idx_pool: list[torch.Tensor | None] = [None] * self._n_layers
        self.state_ring: list[CompressStateRing | None] = [None] * self._n_layers
        self.indexer_state_ring: list[CompressStateRing | None] = [None] * self._n_layers
        self.cmp_scratch_base: list[int | None] = [None] * self._n_layers
        self.idx_scratch_base: list[int | None] = [None] * self._n_layers

    def unit_bytes(self) -> tuple[int, int]:
        # This rank owns only logical scheduler metadata, not KV payload capacity.
        return 0, 0

    def validate_rebuild(
        self,
        config,
        *,
        num_pages: int | None,
        target_moe: int,
        per_expert_bytes: int,
        baseline_free: int,
        weights_bytes: int,
        current_num_pages: int,
        extra_fixed_bytes: int = 0,
        extra_note: str = "",
        num_swa_pages: int | None = None,
        **targets,
    ) -> None:
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from freetoken.utils import mem_GB

        from .base import CacheRebuildRejected
        from .dsv4_cost_model import _dsv4_window_floor_pages
        from .dsv41_cost_model import _dsv41_pool_sizes

        args = self._args(config)
        target_pages = num_pages if num_pages is not None else current_num_pages
        floor = _dsv4_window_floor_pages(config, args.window_size)
        if target_pages < floor:
            raise CacheRebuildRejected(
                f"num_pages {target_pages} is below the DSV4.1 window working-set "
                f"floor {floor} (max_running_req={config.max_running_req})"
            )

        sizes = (
            _dsv41_pool_sizes(config, target_pages + 1, num_swa_pages=num_swa_pages)
            if num_pages is not None or num_swa_pages is not None
            else self.sizes
        )
        # The logical full->window map is the only size-dependent payload.  Include its exact
        # bytes and the tiny window-page free-list upper bound in the fit check.
        metadata_bytes = (sizes.full_token + 1) * 8 + max(0, sizes.n_win_pages - 1) * 8
        budget = net_cache_budget_bytes(
            config.memory_ratio, baseline_free, weights_bytes, extra_fixed_bytes
        )
        need = target_moe * per_expert_bytes + metadata_bytes
        if need > budget:
            raise CacheRebuildRejected(
                f"requested expert-worker cache (moe={target_moe} slots, "
                f"logical-kv={target_pages} pages{extra_note}) needs {mem_GB(need)} > "
                f"budget {mem_GB(budget)}; old cache kept, still serving"
            )

    def k_cache(self, index: int) -> torch.Tensor:
        raise RuntimeError("expert-only DSV4.1 rank has no attention KV payload")

    def v_cache(self, index: int) -> torch.Tensor:
        raise RuntimeError("expert-only DSV4.1 rank has no attention KV payload")

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        raise RuntimeError("expert-only DSV4.1 rank cannot execute attention")


__all__ = ["DSV41ExpertWorkerKVCache", "DSV41PagedKVCache"]
