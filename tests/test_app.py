"""Small unit test for app.py's text-cleanup helper.

app.py loads the local Qwen model onto a GPU, connects to a remote
CLIP Interrogator Space, and launches the Gradio server unconditionally at
import time. None of that is possible on a plain CI runner, so before
importing app we swap those heavy/networked pieces for harmless stand-ins.
app.py itself is not modified in any way to make this work.
"""
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
