"""Mapped handoff must reuse its startup-loaded module across layer flag slots."""
import pytest

from triton._C.libtriton import native_specialize_impl
from triton.backends.compiler import BaseBackend

from freetoken.moe.cpu_handoff import _submit


@pytest.mark.parametrize("slot", [0, 1, 2, 3, 16, 17, 32, 39])
def test_flag_slot_has_one_runtime_specialization(slot):
    parameter = next(param for param in _submit.params if param.name == "slot")
    assert parameter.do_not_specialize
    # Test the installed JIT's native scalar classifier without a CUDA context.
    key = native_specialize_impl(
        BaseBackend, slot, False, not parameter.do_not_specialize,
        not parameter.do_not_specialize_on_alignment,
    )
    assert key == ("i32", None)
