"""Static assertions that the pie-SDK primitives the optimization ideas rely
on are present at the pinned pie-vllm commit. These are build-time guards —
they prevent a silent rug-pull when a future sync moves the submodule."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SDK = REPO_ROOT / "pie-vllm" / "pie" / "sdk" / "rust" / "inferlet" / "src"


def _src(rel: str) -> str:
    return (SDK / rel).read_text()


def test_fork_exists():
    assert re.search(r"pub fn fork\(&self\)\s*->\s*Self", _src("context.rs"))


def test_mask_token_range_exists():
    assert re.search(
        r"pub fn mask_token_range\(\s*&mut self,\s*start:\s*usize,\s*end:\s*usize,\s*mask:\s*bool\s*\)",
        _src("context.rs"),
    )


def test_drop_masked_kv_pages_exists():
    assert re.search(r"pub fn drop_masked_kv_pages\(\s*&mut self\s*\)", _src("context.rs"))


def test_export_resources_exists():
    # Export is re-exported through lib.rs; the underlying symbol lives in api.rs.
    assert "export_resources" in _src("api.rs")
    assert "export_resources" in _src("lib.rs")


def test_import_resources_exists():
    assert "import_resources" in _src("api.rs")
    assert "import_resources" in _src("lib.rs")
