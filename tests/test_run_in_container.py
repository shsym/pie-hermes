"""scripts/run_in_container.sh must honor HERMES_CONFIG_PATH env var for
per-run config selection (Phase 1 needs GPT-branded / Gemini-branded
variants without editing the script)."""
import os
import subprocess
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "run_in_container.sh"


def _extract_config_resolution_line() -> str:
    """Return the line in the shell script that resolves the config source.

    The assertion is structural: we want the path to be an env-var
    lookup with a default, NOT a hardcoded string."""
    text = SCRIPT.read_text()
    # Look for the line feeding configs/cli-config.pie.yaml to sed.
    lines = [l for l in text.splitlines() if "cli-config.pie" in l and "sed" in l]
    assert lines, "could not find the config-source line in run_in_container.sh"
    return lines[0]


def test_config_source_is_env_driven():
    line = _extract_config_resolution_line()
    # Must reference HERMES_CONFIG_PATH somewhere on the line.
    assert "HERMES_CONFIG_PATH" in line, (
        f"config source is hardcoded, not env-driven:\n{line}"
    )


def test_config_source_has_default():
    """Absent HERMES_CONFIG_PATH, the script must still work as before —
    default to configs/cli-config.pie.yaml."""
    line = _extract_config_resolution_line()
    assert "configs/cli-config.pie.yaml" in line, (
        f"default config path missing from line:\n{line}"
    )


def test_shellcheck_clean():
    """Catch shell syntax regressions with a cheap parse."""
    r = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"bash -n failed:\n{r.stderr}"
