"""Opt-in conditional mapped-pinned handoff for single-token CPU experts.

The wait remains a CUDA front-end stream memory operation: no GPU spin kernel.
An empty task publishes done=1 without ringing the CPU coordinator. Each replay
still writes a fresh GPU output (zero for every empty/invalid route).
"""
import torch
import triton
import triton.language as tl

from freetoken.kernel.pinned import device_ptr


# Each layer receives a different flag slot. Specializing on 0/alignment or
# constant 1 would leave modules unwarmed after the layer-0 startup fence and
# could context-synchronize behind an unmatched NCCL operation on first use.
@triton.jit(do_not_specialize=["slot"])
def _submit(x, ids, weights, host_x_addr, host_ids_addr, host_w_addr,
            done_addr, ready_addr, active_out, slot,
            H: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    routes = tl.arange(0, triton.next_power_of_2(TOPK))
    rid = tl.load(ids + routes, routes < TOPK, other=-1)
    w = tl.load(weights + routes, routes < TOPK, other=0.)
    active = tl.sum(((rid >= 0) & (w != 0.)).to(tl.int32)) > 0
    tl.store(active_out, active.to(tl.int32))
    done = done_addr.to(tl.pointer_type(tl.int64)) + slot
    ready = ready_addr.to(tl.pointer_type(tl.int64)) + slot
    if active:
        tl.store(done, 0)
        offsets = tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < H, other=0)
        tl.store(host_x_addr.to(tl.pointer_type(tl.bfloat16)) + offsets, values, offsets < H)
        tl.store(host_ids_addr.to(tl.pointer_type(tl.int32)) + routes, rid, routes < TOPK)
        tl.store(host_w_addr.to(tl.pointer_type(tl.float32)) + routes, w, routes < TOPK)
        # Every thread orders its own posted host writes before the block barrier;
        # only afterwards may the publishing thread ring the CPU doorbell.
        tl.inline_asm_elementwise("membar.sys; mov.b32 $0, $1;", "=r,r",
            [tl.arange(0, 128)], dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        tl.store(ready, 1)
    else:
        tl.store(done, 1)


@triton.jit
def _pull(host_y_addr, active_ptr, out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = tl.load(active_ptr) != 0
    # The installed Triton lowers cache_modifier=".cv" to an ordinary cached
    # 16-bit load. Spell out the PTX: the CPU overwrites these host addresses on
    # every dispatch, so a cache-friendly test alone cannot prove freshness.
    bits = tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; mov.b16 $0, 0; "
        "@p ld.global.cv.b16 $0, [$1]; }", "=h,l,r",
        [host_y_addr + offsets.to(tl.int64) * 2,
         ((offsets < COUNT) & active).to(tl.int32)],
        dtype=tl.uint16, is_pure=False, pack=1,
    )
    values = bits.to(tl.bfloat16, bitcast=True)
    tl.store(out + offsets, values, offsets < COUNT)


def submit(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
           io: dict, done: torch.Tensor, ready: torch.Tensor, active: torch.Tensor,
           slot: int) -> None:
    assert hidden.shape[0] == 1 and hidden.dtype == torch.bfloat16
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    _submit[(1,)](hidden, ids, weights,
        device_ptr(io["x"]), device_ptr(io["ids"]), device_ptr(io["w"]),
        device_ptr(done), device_ptr(ready), active, slot,
        H=hidden.shape[1], TOPK=ids.shape[1], BLOCK=triton.next_power_of_2(hidden.shape[1]),
        num_warps=4)


def pull(io: dict, active: torch.Tensor, out: torch.Tensor) -> None:
    _pull[(triton.cdiv(out.numel(), 1024),)](
        device_ptr(io["y"]), active, out, COUNT=out.numel(), BLOCK=1024)
