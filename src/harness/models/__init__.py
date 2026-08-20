"""Model providers: one interface, one real backend, one fake for tests.

``FakeProvider`` is not a convenience — it is what lets the agent loop, the
protocol layer and the tool layer be tested without the 18 GB model, and what
makes prompt-prefix stability assertable in CI.
"""

from harness.models.base import ModelProvider
from harness.models.fake import FakeProvider
from harness.models.llamacpp import DEFAULT_BASE_URL, LlamaCppProvider

__all__ = ["DEFAULT_BASE_URL", "FakeProvider", "LlamaCppProvider", "ModelProvider"]
