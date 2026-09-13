# Used Sonnet 5 with Medium Effort, for writing a small pytest test file that tests the get_text_after_colon helper function in app.py without needing a GPU or network access.
# prompt: Write 1 to 2 pytest tests for the get_text_after_colon function in app.py. The app.py file loads a Qwen model onto a GPU, connects to a remote CLIP Interrogator Hugging Face Space, and launches a Gradio server, all at the top level of the file when it is imported, so importing app.py directly in a test will fail on a normal CI runner. Do not modify app.py. Instead, stub out the transformers and spaces modules, patch gradio_client.Client and gradio.Blocks.launch and gradio.Blocks.queue to no-ops before importing app, then write two small tests, one for a string that has a colon and one for a string that does not have a colon.

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make sure `import app` finds app.py in the repo root regardless of the
# directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# transformers: stub out model/tokenizer loading so `.from_pretrained(...)`
# and `.half().cuda()` return mocks instead of downloading real weights.
fake_transformers = MagicMock()
sys.modules["transformers"] = fake_transformers

# spaces: stub out the whole module, but keep `@spaces.GPU` as an identity
# decorator so the real gen_safety_advice function is preserved as-is.
fake_spaces = MagicMock()
fake_spaces.GPU = lambda fn: fn
sys.modules["spaces"] = fake_spaces

# gradio_client: real package (lightweight, no GPU), but patch Client so
# `clipi_client = Client(...)` doesn't make a real network call at import.
import gradio_client
gradio_client.Client = MagicMock()

# gradio: real package, but no-op the launch/queue calls so the module-level
# `demo.queue(...).launch(...)` doesn't try to start a real server.
import gradio
gradio.Blocks.queue = lambda self, *args, **kwargs: self
gradio.Blocks.launch = lambda self, *args, **kwargs: None

import app  # noqa: E402  (must import after the stubs above are in place)


def test_get_text_after_colon_strips_prefix():
    assert app.get_text_after_colon("Title: some advice") == "some advice"


def test_get_text_after_colon_no_colon_returns_original():
    assert app.get_text_after_colon("no colon here") == "no colon here"
