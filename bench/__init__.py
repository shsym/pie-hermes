"""pie-hermes benchmarking package.

Out-of-tree harness that drives the vendored Hermes Agent submodule
against a pie (openai-compat-v2) backend and records full LLM traffic
for prompt-composition analysis. No changes are made to the Hermes
tree — capture is installed by wrapping ``openai.OpenAI.__init__`` at
import time (see ``bench.install``)."""
