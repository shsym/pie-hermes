"""The vendored inferlet must resolve its SDK path dep from the in-tree
submodule (pie-vllm), not from the user's sibling-checkout of pieclaw."""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERLET = REPO_ROOT / "inferlets" / "hermes-openai-compat"


def test_cargo_path_dep_points_inside_worktree():
    cargo = (INFERLET / "Cargo.toml").read_text()
    # Must reference the in-tree submodule, not '../../../pie-vllm' or any
    # parent-escaping path.
    assert 'path = "../../pie-vllm/pie/sdk/rust/inferlet"' in cargo


def _has_wasm_target() -> bool:
    if shutil.which("rustup") is None:
        return False
    r = subprocess.run(
        ["rustup", "target", "list", "--installed"],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and "wasm32-wasip2" in r.stdout.splitlines()


@pytest.mark.skipif(
    not _has_wasm_target(),
    reason="wasm32-wasip2 target not installed (run: rustup target add wasm32-wasip2)",
)
def test_cargo_check_succeeds():
    """Fails fast when the path dep is wrong or pie-vllm isn't checked out.
    Skipped cleanly on hosts without the wasm target so local dev doesn't break."""
    r = subprocess.run(
        ["cargo", "check", "--target", "wasm32-wasip2"],
        cwd=INFERLET, capture_output=True, text=True,
    )
    assert r.returncode == 0, f"cargo check failed:\n{r.stderr[-4000:]}"
