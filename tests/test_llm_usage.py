from __future__ import annotations

import os
import inspect
import unittest
from unittest.mock import patch

from workshop_api.llm import (
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_OPENROUTER_MODEL,
    LLMClient,
    LLMUsage,
)
from workshop_api.llm_server import ChatRequest


class LLMClientDefaultsTests(unittest.TestCase):
    def test_openrouter_defaults_to_glm_5_3_flash(self) -> None:
        with patch.dict(
            os.environ,
            {"WORKSHOP_LLM_PROVIDER": "openrouter"},
            clear=True,
        ):
            client = LLMClient.direct_from_env()

        self.assertEqual(DEFAULT_OPENROUTER_MODEL, "z-ai/glm-5.3-flash")
        self.assertEqual(client.model, DEFAULT_OPENROUTER_MODEL)

    def test_proof_and_proxy_defaults_allow_20k_output_tokens(self) -> None:
        self.assertEqual(DEFAULT_LLM_MAX_TOKENS, 20_000)
        self.assertEqual(
            inspect.signature(LLMClient.prove).parameters["max_tokens"].default,
            20_000,
        )
        self.assertEqual(
            inspect.signature(LLMClient.chat_with_usage)
            .parameters["max_tokens"]
            .default,
            20_000,
        )
        self.assertEqual(ChatRequest.model_fields["max_tokens"].default, 20_000)


class LLMUsagePricingTests(unittest.TestCase):
    def test_glm_5_3_flash_uses_model_specific_cache_price(self) -> None:
        usage = LLMUsage.from_counts(
            model="z-ai/glm-5.3-flash",
            input_tokens=1_000_000,
            cache_tokens=250_000,
            output_tokens=1_000_000,
        )

        self.assertAlmostEqual(usage.input_cost_usd, 0.05625)
        self.assertAlmostEqual(usage.cached_input_cost_usd, 0.00375)
        self.assertAlmostEqual(usage.output_cost_usd, 0.25)
        self.assertAlmostEqual(usage.total_cost_usd, 0.31)

    def test_mistral_cache_price_is_preserved(self) -> None:
        usage = LLMUsage.from_counts(
            model="mistral-medium-latest",
            input_tokens=1_000_000,
            cache_tokens=250_000,
            output_tokens=1_000_000,
        )

        self.assertAlmostEqual(usage.input_cost_usd, 1.125)
        self.assertAlmostEqual(usage.cached_input_cost_usd, 0.0375)
        self.assertAlmostEqual(usage.output_cost_usd, 7.5)
        self.assertAlmostEqual(usage.total_cost_usd, 8.6625)


if __name__ == "__main__":
    unittest.main()
