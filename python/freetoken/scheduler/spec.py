"""MTP speculative-decoding driver (the scheduler side of HANDOFF §4).

Turns single-stream DSV4 decode into draft -> 2-token verify cycles:

  1. The draft sidecar (models/deepseek_v4/mtp_sidecar.py, on FREETOKEN_MTP_DEVICE)
     predicts token t+2 from (final-layer hc state at t, sampled token t+1).
  2. Each cycle the driver runs ONE fused verify extend [last_sampled@p, draft@p+1]
     through Transformer.spec_extend: per-token decode attention + batched MoE, so
     the expert fetch — the thing decode is latency-bound on — serves 2 tokens.
  3. Row 0 of the verify samples the TRUE token at p+1; acceptance is (row0 == draft),
     exact at any temperature (row0 is an exact sample from the correct distribution;
     when it equals the draft, row 1's sample is a valid next token for free).
  4. Reject costs nothing structural: all DSV4 attention state (window ring, cmp/idx
     rows, compressor carry blocks) is position-addressed, so re-extending the same
     position overwrites the stale draft entries. Only a page-boundary allocation
     needs explicit cleanup (CacheManager.free_spec_tail).

The sidecar is fed ONLY committed pairs — (state@p, token@p+1) where token@p+1 was
emitted — so its 128-slot ring never needs a rejection rewind, and a fresh draft is
available even immediately after a reject (the pair (state@p, s1) predicts p+2).
Every cycle is therefore a 2-token verify: E[tokens/cycle] = 1 + acceptance.

Engagement rules (MVP): tp == 1, FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 (the driver
consumes the model's per-forward hc stash at drain time, which the next overlapped
forward would clobber), and exactly one request in the system. A newly arriving
request exits speculation at the next cycle boundary, where the request's counters
are indistinguishable from a normal mid-decode request's, and hands it back to the
decode manager. No re-adoption mid-generation.

Adoption is limited to SINGLE-CHUNK, COLD prompts of at most FREETOKEN_MTP_MAX_PROMPT
tokens (default 4096): the checkpoint's reference MTP block ingests multi-token
batches only at position 0 (its pos>0 branch is single-token), so a chunked prompt or
a prefix-cache hit has no supported bulk-ingest path, and the plain-torch sparse_attn
gather bounds the one-shot size. Those requests decode normally. (Lifting this needs
a multi-token pos>0 path patched into the sidecar's reference block — future work,
alongside draft-latency optimization: the accept path pays two ~8 ms single-token
sidecar steps per cycle for the same reason.)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import Batch, Req
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger

from .prefill import ChunkedReq, _maybe_pinned

if TYPE_CHECKING:
    from .scheduler import Scheduler

logger = init_logger(__name__)

_LOG_EVERY_CYCLES = 100


def create_spec_manager(scheduler: Scheduler) -> "SpecDecodeManager | None":
    """Build the driver if every prerequisite holds, else warn once and return None
    (serving continues without speculation)."""
    import os

    from freetoken.env import ENV

    config = scheduler.config
    if getattr(config.model_config, "dsv4_args", None) is None:
        logger.warning_rank0("FREETOKEN_MTP_SPEC: not a DSV4 model; speculation disabled")
        return None
    if config.tp_info.size > 1:
        logger.warning_rank0(
            "FREETOKEN_MTP_SPEC: tp > 1 is not supported yet (the sidecar drafts on rank 0 "
            "only); speculation disabled"
        )
        return None
    if not ENV.DISABLE_OVERLAP_SCHEDULING:
        logger.warning_rank0(
            "FREETOKEN_MTP_SPEC requires FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 (the driver "
            "reads each forward's hc stash at drain time); speculation disabled"
        )
        return None
    try:
        from freetoken.models.deepseek_v4.mtp_sidecar import MTPSidecar

        t0 = time.monotonic()
        sidecar = MTPSidecar(config.model_path, max_seq_len=scheduler.engine.max_seq_len)
        logger.info_rank0(
            f"MTP spec: draft sidecar ready on {sidecar.device} "
            f"in {time.monotonic() - t0:.1f}s"
        )
        if sidecar.device == scheduler.engine.device:
            logger.warning_rank0(
                "MTP spec: sidecar shares the serving GPU; its resident bf16 weights cost "
                "~14 GB there. Point FREETOKEN_MTP_DEVICE at a spare GPU if available."
            )
    except Exception:
        logger.exception("FREETOKEN_MTP_SPEC: sidecar construction failed; speculation disabled")
        return None
    max_prompt = int(os.environ.get("FREETOKEN_MTP_MAX_PROMPT", "4096"))
    return SpecDecodeManager(scheduler, sidecar, max_prompt=max_prompt)


class SpecDecodeManager:
    def __init__(self, scheduler: Scheduler, sidecar, max_prompt: int = 4096) -> None:
        self.sched = scheduler
        self.sidecar = sidecar
        self.max_prompt = max_prompt  # one-shot sidecar ingest bound (see module docstring)
        # --- active speculation (at most one request) ---
        self.req: Req | None = None
        self.pending_draft: int | None = None  # draft for the position after req's last token
        # --- health / stats ---
        self.disabled = False          # latched on any sidecar failure
        self.cycles = 0
        self.accepts = 0
        self.emitted = 0
        self._log_cycles = 0
        self._log_emitted = 0
        self._log_accepts = 0
        self._log_t0 = time.monotonic()
        # Per-window cycle-cost breakdown (seconds): GPU verify (the drain's event sync),
        # sidecar drafting, and everything else (scheduling + drain host work), logged with
        # the acceptance line. sync time is stamped by the scheduler's drain.
        self.t_sync = 0.0
        self._t_draft = 0.0

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self.req is not None

    @property
    def runnable(self) -> bool:
        return self.req is not None

    def _latch_disabled(self) -> None:
        """A sidecar error must never take serving down: latch speculation off and hand any
        active request back to normal decode."""
        logger.exception("MTP spec: sidecar failure; speculation disabled for this server run")
        self.disabled = True
        if self.req is not None:
            self._exit_to_decode()

    # -------------------------------------------------------------- scheduling
    def before_schedule(self) -> None:
        """Cycle-boundary exit: a pending prefill (new request) ends single-stream
        speculation BEFORE the prefill admission pass, so the adder sees the request's
        remaining tokens reserved through the decode manager."""
        if self.req is not None and self.sched.prefill_manager.runnable:
            self._exit_to_decode()

    def schedule_next_batch(self) -> Batch | None:
        if self.req is None:
            return None
        req = self.req
        draft, self.pending_draft = self.pending_draft, None
        # A draft needs headroom for BOTH its own slot and the accept-path bonus token;
        # with less, run the final token as a plain 1-token verify step.
        if draft is not None and req.remain_len >= 2:
            req.append_host(req.input_ids.new_tensor([draft]))
            req.device_len += 1
        # Stage the chunk tokens into the device token pool (the PrefillAdder pattern);
        # ordered before the forward by stream program order (non-overlap mode).
        ids = req.input_ids[req.cached_len : req.device_len]
        self.sched.token_pool[req.table_idx, req.cached_len : req.device_len].copy_(
            _maybe_pinned(ids), non_blocking=True
        )
        batch = Batch(reqs=[req], phase="prefill")
        batch.spec_verify = True
        batch.rows_per_req = [req.extend_len]
        return batch

    def _exit_to_decode(self) -> None:
        """Hand the request back to the decode manager at a cycle boundary. The counters
        already look like a normal mid-decode request's, and token_pool holds the pending
        input token at cached_len (the spec write mapping stored each verify row's sample
        at its own position), so generic decode picks up seamlessly."""
        req = self.req
        assert req is not None
        self.req = None
        self.pending_draft = None
        req.spec_active = False
        self.sched.decode_manager.running_reqs.add(req)

    def abort_req(self, uid: int) -> Req | None:
        if self.req is not None and self.req.uid == uid:
            req = self.req
            self.req = None
            self.pending_draft = None
            req.spec_active = False
            return req
        return None

    # ------------------------------------------------- prompt ingestion / adoption
    def observe_batch(self, batch: Batch, next_tokens_cpu: torch.Tensor, finished) -> None:
        """Called after the generic drain of every NON-spec batch. Adopts a request into
        speculation when its prompt just completed as ONE cold chunk (see the module
        docstring for why chunked / prefix-hit prompts are out of MVP scope) and it is the
        lone live request. Always consumes the model's hc stash so a multi-GB prompt-chunk
        tensor never outlives its drain."""
        hc = self.sched.engine.model.take_spec_hc()
        if self.disabled or self.req is not None or not batch.is_prefill:
            return
        if len(batch.reqs) != 1 or hc is None:
            return
        req = batch.reqs[0]
        T = hc.shape[0]
        if (
            isinstance(req, ChunkedReq)          # multi-chunk prompt: no bulk-ingest path
            or req.cached_len - T != 0           # not the whole prompt from position 0
            or req.cache_handle.cached_len != 0  # prefix hit: those positions have no hc
            or T > self.max_prompt               # sidecar one-shot ingest bound
            or req.aborted
            or req.mm_embeds is not None         # sidecar drafts from token ids only
            or req in finished
            or not req.can_decode
            or self.sched.decode_manager.running_reqs != {req}
            or self.sched.prefill_manager.runnable
        ):
            return
        try:
            # Pair every prompt state with the token AFTER it; the last state's pair token
            # is the first sampled token s0. Returns the draft for the position after s0.
            self.sidecar.reset()
            tokens = torch.cat(
                [req.input_ids[1:T].long(), next_tokens_cpu[0].view(1).long()]
            )
            d1 = self.sidecar.extend(hc, tokens)
        except Exception:
            self._latch_disabled()
            return
        self.sched.decode_manager.remove_req(req)
        req.spec_active = True
        self.req = req
        self.pending_draft = d1

    # ------------------------------------------------------------------- drain
    def process_drained(self, batch: Batch, next_tokens_cpu: torch.Tensor) -> List[DetokenizeMsg]:
        """Consume a spec-verify forward: acceptance, emits, rewind-on-reject, sidecar
        advance, next draft. Called after copy_done.synchronize()."""
        req = batch.reqs[0]
        hc = self.sched.engine.model.take_spec_hc()  # [T, hc, dim]
        assert req is self.req and hc is not None
        T = batch.rows_per_req[0]
        base = req.cached_len - T  # position of the first fed token
        if req.aborted:
            # normal_loop processes aborts with no forward in flight, so an abort observed
            # here was set before this batch launched -- defensive; free and stop.
            self._drop_active()
            self.sched._free_req_resources(req)
            return []
        toks = [int(next_tokens_cpu[t].item()) for t in range(T)]

        reply: List[DetokenizeMsg] = []
        finished = False
        committed_pairs: list[int] = []  # tokens for pairs (hc[i] -> committed_pairs[i])
        if T == 2:
            draft = int(req.input_ids[base + 1])
            s1, s2 = toks
            self.cycles += 1
            self._log_cycles += 1
            if s1 == draft:
                self.accepts += 1
                self._log_accepts += 1
                finished = self._emit(reply, req, s1, ids_len=base + 2)
                if finished:
                    # Drop the un-earned bonus token; ids already end at the accepted draft.
                    req.device_len = base + 2
                else:
                    req.append_host(req.input_ids.new_tensor([s2]))
                    finished = self._emit(reply, req, s2, ids_len=base + 3)
                    committed_pairs = [s1, s2]
            else:
                # Reject: the true sample replaces the draft; position base+1 is re-extended
                # next cycle (stale KV overwritten in place). Free the tail page first if
                # the rejected extend crossed a page boundary (uses the pre-rewind cached_len).
                self.sched.cache_manager.free_spec_tail(req, keep_len=base + 1)
                req.spec_rewind(base + 2, s1)
                finished = self._emit(reply, req, s1, ids_len=base + 2)
                committed_pairs = [s1]
        else:
            # Draft-less 1-token step (output budget edge): identical to a decode step.
            s = toks[0]
            req.append_host(req.input_ids.new_tensor([s]))
            finished = self._emit(reply, req, s, ids_len=base + 2)
            committed_pairs = [s]

        if finished:
            self._drop_active()
            self.sched._free_req_resources(req)
        elif committed_pairs:
            try:
                t0 = time.monotonic()
                self.pending_draft = self.sidecar.extend(
                    hc[: len(committed_pairs)], committed_pairs
                )
                self._t_draft += time.monotonic() - t0
            except Exception:
                self._latch_disabled()
        self._maybe_log()
        return reply

    def _drop_active(self) -> None:
        req = self.req
        self.req = None
        self.pending_draft = None
        if req is not None:
            req.spec_active = False

    def _emit(self, reply: List[DetokenizeMsg], req: Req, token: int, *, ids_len: int) -> bool:
        """Append one DetokenizeMsg for ``token`` (already in req.input_ids), mirroring the
        generic drain's finish rules. Returns whether the request finished on it."""
        self.emitted += 1
        self._log_emitted += 1
        hit_length = ids_len >= req.max_device_len
        hit_eos = not req.sampling_params.ignore_eos and token in self.sched.eos_token_ids
        matched_stop = (
            self.sched._match_stop_str(req)
            if not hit_eos and req.sampling_params.stop_strs
            else None
        )
        finished = hit_length or hit_eos or matched_stop is not None
        reply.append(
            DetokenizeMsg(
                uid=req.uid,
                next_token=token,
                finished=finished,
                finish_reason=(
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                ),
                matched_stop=matched_stop,
                stop_strs=req.sampling_params.stop_strs or None,
            )
        )
        return finished

    def _maybe_log(self) -> None:
        if self._log_cycles < _LOG_EVERY_CYCLES:
            return
        dt = max(time.monotonic() - self._log_t0, 1e-6)
        n = self._log_cycles
        logger.info_rank0(
            f"MTP spec: accept {self._log_accepts}/{n} "
            f"({100.0 * self._log_accepts / n:.1f}%), "
            f"{self._log_emitted / dt:.1f} tok/s over the last {self._log_emitted} tokens "
            f"| per cycle: verify-sync {1e3 * self.t_sync / n:.0f}ms, "
            f"draft {1e3 * self._t_draft / n:.0f}ms, "
            f"other {1e3 * max(dt - self.t_sync - self._t_draft, 0) / n:.0f}ms"
        )
        self._log_cycles = self._log_accepts = self._log_emitted = 0
        self.t_sync = self._t_draft = 0.0
        self._log_t0 = time.monotonic()
