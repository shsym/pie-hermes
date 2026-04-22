"""Pie-hermes PromptOptimizer implementations, registered via
``bench.install_optimizers.install()`` at Python startup (through
``bootstrap/sitecustomize.py``).

All optimizers return ``agent.prompt_optimizer.PromptOptimization`` (or a
bare list, which the hermes-agent hook's ``_coerce`` shim wraps)."""
