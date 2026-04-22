"""Submodules must be initialized and pinned where the build depends on them."""
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parent.parent


def _submodule_sha(path: str) -> str:
    out = subprocess.check_output(
        ["git", "ls-tree", "HEAD", path], cwd=REPO_ROOT
    ).decode().strip()
    # format: "<mode> <type> <sha>\t<path>"
    return out.split()[2]


def test_hermes_agent_submodule_registered():
    assert _submodule_sha("hermes-agent"), "hermes-agent submodule not in tree"


def test_pie_vllm_submodule_registered():
    assert _submodule_sha("pie-vllm"), "pie-vllm submodule not in tree (required for inferlet SDK)"


def test_pie_vllm_sdk_inferlet_path_exists():
    """The inferlet crate we vendor expects this relative path."""
    assert (REPO_ROOT / "pie-vllm" / "pie" / "sdk" / "rust" / "inferlet" / "Cargo.toml").is_file()
