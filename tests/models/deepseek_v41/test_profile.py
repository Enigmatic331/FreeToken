from contextlib import contextmanager

import freetoken.models.deepseek_v41.profile as dsv41_profile


def test_profile_range_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(dsv41_profile, "_ENABLED", False)
    monkeypatch.setattr(
        dsv41_profile,
        "_nvtx_range",
        lambda _name: (_ for _ in ()).throw(AssertionError("NVTX must stay disabled")),
    )
    with dsv41_profile.profile_range("DSV41/Test"):
        pass


def test_profile_decorator_qualifies_layer_and_preserves_result(monkeypatch):
    seen = []

    @contextmanager
    def fake_range(name):
        seen.append(("enter", name))
        yield
        seen.append(("exit", name))

    monkeypatch.setattr(dsv41_profile, "_ENABLED", True)
    monkeypatch.setattr(dsv41_profile, "_nvtx_range", fake_range)

    class Stage:
        layer_id = 7

        @dsv41_profile.profile("DSV41/Layer_{}/Decode", layer_id_field="layer_id")
        def run(self, value):
            return value + 1

    assert Stage().run(41) == 42
    assert seen == [
        ("enter", "DSV41/Layer_7/Decode"),
        ("exit", "DSV41/Layer_7/Decode"),
    ]
