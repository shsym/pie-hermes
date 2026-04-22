"""scripts/build-inferlet.sh must produce a .wasm artifact we can point
pie's POD at without further processing."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build-inferlet.sh"


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert os.stat(SCRIPT).st_mode & stat.S_IXUSR


def test_script_produces_wasm_artifact():
    """Integration test — only runs when the wasm toolchain is available."""
    r = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0 and "rustup target add wasm32-wasip2" in r.stderr:
        pytest.skip("wasm32-wasip2 target not installed on host")
    assert r.returncode == 0, f"build failed:\n{r.stderr[-4000:]}"
    wasm = (
        REPO_ROOT / "inferlets" / "hermes-openai-compat" / "target"
        / "wasm32-wasip2" / "release" / "hermes_openai_compat.wasm"
    )
    assert wasm.is_file() and wasm.stat().st_size > 50_000, \
        f"wasm artifact missing or suspiciously small: {wasm}"
