"""Auto-load bench capture hook in subprocess-launched Python interpreters.

Python's ``site`` module imports ``sitecustomize`` at startup if it's
importable on ``sys.path``. The container sets
``PYTHONPATH=/opt/pie-hermes:/opt/pie-hermes/bootstrap`` so this file is
found and runs before Hermes creates any ``openai.OpenAI`` client.

Gated on ``HERMES_CAPTURE_FILE``/``HERMES_CAPTURE_DIR`` so regular
Hermes sessions outside of a capture run incur zero overhead."""

import os

if os.environ.get("HERMES_CAPTURE_FILE") or os.environ.get("HERMES_CAPTURE_DIR"):
    try:
        from bench.install import install

        install()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "pie-hermes sitecustomize failed: %s", exc
        )
