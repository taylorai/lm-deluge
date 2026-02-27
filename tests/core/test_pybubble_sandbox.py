"""Tests for PybubbleSandbox platform and dependency guards."""

import sys
import types
from unittest.mock import patch

import lm_deluge.tool.prefab.sandbox as sandbox_module
from lm_deluge.tool.prefab.sandbox.pybubble_sandbox import PybubbleSandbox


class _FakeRuntimeSandbox:
    def __init__(self, *args, **kwargs):
        del args, kwargs


def _which_missing_bwrap(executable: str) -> str | None:
    if executable == "bwrap":
        return None
    return f"/usr/bin/{executable}"


def _which_missing_slirp(executable: str) -> str | None:
    if executable == "slirp4netns":
        return None
    return f"/usr/bin/{executable}"


def _which_present(executable: str) -> str:
    return f"/usr/bin/{executable}"


def test_lazy_export_is_linux_only():
    if sys.platform == "linux":
        exported = getattr(sandbox_module, "PybubbleSandbox")
        assert exported is PybubbleSandbox
    else:
        try:
            getattr(sandbox_module, "PybubbleSandbox")
        except AttributeError as e:
            assert "only available on Linux" in str(e)
        else:
            raise AssertionError("Expected PybubbleSandbox to be unavailable")


def test_constructor_reports_missing_pybubble_dependency():
    with patch("lm_deluge.tool.prefab.sandbox.pybubble_sandbox.sys.platform", "linux"):
        with patch.dict(sys.modules, {"pybubble": None}):
            try:
                PybubbleSandbox(network_access=False)
            except RuntimeError as e:
                assert "optional 'pybubble' dependency" in str(e)
            else:
                raise AssertionError("Expected RuntimeError when pybubble is missing")


def test_constructor_reports_missing_bwrap():
    fake_module = types.SimpleNamespace(Sandbox=_FakeRuntimeSandbox)

    with patch("lm_deluge.tool.prefab.sandbox.pybubble_sandbox.sys.platform", "linux"):
        with patch.dict(sys.modules, {"pybubble": fake_module}):
            with patch(
                "lm_deluge.tool.prefab.sandbox.pybubble_sandbox.shutil.which",
                side_effect=_which_missing_bwrap,
            ):
                try:
                    PybubbleSandbox(network_access=False)
                except RuntimeError as e:
                    assert "bubblewrap executable" in str(e)
                else:
                    raise AssertionError("Expected RuntimeError when bwrap is missing")


def test_constructor_reports_missing_slirp4netns_for_outbound_access():
    fake_module = types.SimpleNamespace(Sandbox=_FakeRuntimeSandbox)

    with patch("lm_deluge.tool.prefab.sandbox.pybubble_sandbox.sys.platform", "linux"):
        with patch.dict(sys.modules, {"pybubble": fake_module}):
            with patch(
                "lm_deluge.tool.prefab.sandbox.pybubble_sandbox.shutil.which",
                side_effect=_which_missing_slirp,
            ):
                try:
                    PybubbleSandbox(network_access=True)
                except RuntimeError as e:
                    assert "outbound_access=True" in str(e)
                else:
                    raise AssertionError(
                        "Expected RuntimeError when slirp4netns is missing"
                    )

                sandbox = PybubbleSandbox(network_access=True, outbound_access=False)
                assert sandbox.network_access is True
                assert sandbox.outbound_access is False


def test_network_namespace_fallback_detection():
    fake_module = types.SimpleNamespace(Sandbox=_FakeRuntimeSandbox)

    with patch("lm_deluge.tool.prefab.sandbox.pybubble_sandbox.sys.platform", "linux"):
        with patch.dict(sys.modules, {"pybubble": fake_module}):
            with patch(
                "lm_deluge.tool.prefab.sandbox.pybubble_sandbox.shutil.which",
                side_effect=_which_present,
            ):
                sandbox = PybubbleSandbox(network_access=True)
                assert sandbox._should_fallback_to_host_network(
                    RuntimeError(
                        "Network namespace watchdog exited before becoming ready."
                    )
                )

                sandbox_no_fallback = PybubbleSandbox(
                    network_access=True,
                    fallback_to_host_network=False,
                )
                assert not sandbox_no_fallback._should_fallback_to_host_network(
                    RuntimeError(
                        "Network namespace watchdog exited before becoming ready."
                    )
                )

                sandbox_no_network = PybubbleSandbox(network_access=False)
                assert not sandbox_no_network._should_fallback_to_host_network(
                    RuntimeError(
                        "Network namespace watchdog exited before becoming ready."
                    )
                )


def main():
    test_lazy_export_is_linux_only()
    test_constructor_reports_missing_pybubble_dependency()
    test_constructor_reports_missing_bwrap()
    test_constructor_reports_missing_slirp4netns_for_outbound_access()
    test_network_namespace_fallback_detection()
    print("\n✅ PybubbleSandbox tests passed")


if __name__ == "__main__":
    main()
