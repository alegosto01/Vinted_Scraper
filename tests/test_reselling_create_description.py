import sys
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiments.old.reselling_process.create_description.generate import (  # noqa: E402
    build_prompt_characteristics_block,
    build_description_payload,
    build_gemini_generation_config,
    build_gemini_model_candidates,
    build_structured_facts,
    build_llama_description_prompt,
    clean_description_sentences,
    extract_gemini_finish_reason,
    extract_gemini_response_text,
    format_ollama_runtime_error,
    gemini_output_looks_incomplete,
    generate_description_text,
    llm_output_needs_fallback,
    retrieve_similar_examples,
)


class ResellingCreateDescriptionTests(unittest.TestCase):
    def test_clean_description_sentences_keeps_facts_and_drops_noise(self):
        text = (
            "Condizioni ottime, usate due volte. Guarda gli altri annunci. "
            "Con scatola originale e lacci extra."
        )

        sentences = clean_description_sentences(text)

        self.assertEqual(
            sentences,
            ["Condizioni ottime, usate due volte.", "Con scatola originale e lacci extra."],
        )

    def test_retrieve_similar_examples_prefers_same_brand_and_variant(self):
        query = pd.Series(
            {
                "Title": "Nike Air Max 97 silver bullet",
                "Brand": "Nike",
                "Brand_norm": "nike",
                "EmbedText": "nike air max 97 silver bullet | nike | 42",
                "BlockKey": "nike__air_max_97",
                "ProductId": 123,
                "VariantId": 456,
                "Price": 120,
                "item_id": "1",
                "SearchName": "nike",
            }
        )
        corpus = pd.DataFrame(
            [
                {
                    "Title": "Nike Air Max 97 silver bullet EU42",
                    "Brand": "Nike",
                    "Brand_norm": "nike",
                    "EmbedText": "nike air max 97 silver bullet | nike | 42",
                    "BlockKey": "nike__air_max_97",
                    "ProductId": 123,
                    "VariantId": 456,
                    "Price": 118,
                    "item_id": "2",
                    "SearchName": "nike",
                    "Description": "Con scatola originale.",
                    "SourceKind": "full_scrape",
                },
                {
                    "Title": "Adidas Ultraboost 22",
                    "Brand": "Adidas",
                    "Brand_norm": "adidas",
                    "EmbedText": "adidas ultraboost 22 | adidas | 42",
                    "BlockKey": "adidas__ultraboost_22",
                    "ProductId": 999,
                    "VariantId": 999,
                    "Price": 119,
                    "item_id": "3",
                    "SearchName": "nike",
                    "Description": "Altro modello.",
                    "SourceKind": "full_scrape",
                },
            ]
        )

        retrieved = retrieve_similar_examples(query, corpus, limit=2)

        self.assertEqual(retrieved.iloc[0]["item_id"], "2")

    def test_generate_description_text_reads_like_listing_copy(self):
        query = pd.Series(
            {
                "Title": "Nike Air Max 97 silver bullet",
                "Brand": "Nike",
                "Size": "42",
                "Condition": "Ottime condizioni",
                "Description": "Condizioni ottime. Con scatola originale. Piccolo segno sul tallone.",
                "SearchName": "nike",
            }
        )
        similar = pd.DataFrame(
            [
                {
                    "Title": "Nike Air Max 97 silver bullet",
                    "Brand": "Nike",
                    "Brand_norm": "nike",
                    "Description": "Con box originale e extra laces. Modello iconico e facile da abbinare.",
                    "Condition": "Ottime condizioni",
                    "MarketStatus": "sold",
                    "similarity_score": 9.0,
                },
                {
                    "Title": "Nike Air Max 97 OG",
                    "Brand": "Nike",
                    "Brand_norm": "nike",
                    "Description": "Scarpa running retro.",
                    "Condition": "Ottime condizioni",
                    "MarketStatus": "sold",
                    "similarity_score": 7.5,
                },
            ]
        )

        facts = build_structured_facts(query, similar)
        description = generate_description_text(facts, similar)

        self.assertIn("Vendo Nike Air Max 97 silver bullet, taglia/variante 42, ottime condizioni.", description)
        self.assertIn("Condizioni ottime.", description)
        self.assertIn("Completo di scatola originale.", description)
        self.assertIn("Modello iconico e facile da abbinare.", description)
        self.assertIn("Segnalo solo piccolo segno sul tallone.", description)
        self.assertNotIn("Condizioni:", description)
        self.assertNotIn("Incluso:", description)
        self.assertNotIn("Da segnalare:", description)

    def test_generate_description_text_skips_non_italian_borrowed_copy(self):
        query = pd.Series(
            {
                "Title": "Gucci Marmont Mini",
                "Brand": "Gucci",
                "Size": "Unica",
                "Condition": "Ottime condizioni",
                "Description": "Borsa usata una volta.",
                "SearchName": "borse",
            }
        )
        similar = pd.DataFrame(
            [
                {
                    "Title": "Gucci Marmont Mini",
                    "Brand": "Gucci",
                    "Brand_norm": "gucci",
                    "Description": "Colour: pink Condition: excellent, worn once Comes with dust bag.",
                    "Condition": "Excellent",
                    "MarketStatus": "sold",
                    "similarity_score": 9.0,
                }
            ]
        )

        facts = build_structured_facts(query, similar)
        description = generate_description_text(facts, similar)

        self.assertIn("Borsa usata una volta.", description)
        self.assertNotIn("Colour:", description)
        self.assertNotIn("worn once", description)

    def test_build_llama_description_prompt_contains_guardrails_and_facts(self):
        facts = {
            "title": "Gucci Marmont Mini",
            "brand": "Gucci",
            "size": "Unica",
            "condition": "Excellent condition",
            "price": 1290.0,
            "factual_sentences": ["Soft black leather.", "Gold-tone hardware."],
            "accessory_notes": ["Comes with dust bag."],
            "flaw_notes": ["Light mark inside."],
        }
        similar = pd.DataFrame(
            [
                {
                    "Title": "Gucci Marmont Mini",
                    "Description": "Compact shoulder bag with iconic quilting.",
                }
            ]
        )

        prompt = build_llama_description_prompt(facts, similar, language="english")

        self.assertIn("fully in English", prompt)
        self.assertIn("very appealing, human, and credible", prompt)
        self.assertIn("Do not invent authenticity claims", prompt)
        self.assertIn("If a detail is not provided, omit it completely", prompt)
        self.assertIn("Keep the condition wording close to the provided facts", prompt)
        self.assertIn("You may add light persuasive language", prompt)
        self.assertIn("Do not add generic luxury filler", prompt)
        self.assertIn("Do not refer to the listing", prompt)
        self.assertIn("- Item: Gucci Marmont Mini", prompt)
        self.assertIn("- Brand: Gucci", prompt)
        self.assertIn("- Condition: Excellent condition", prompt)
        self.assertIn("- Similar listing cues:", prompt)

    def test_build_prompt_characteristics_block_omits_meaningless_size(self):
        facts = {
            "title": "Zaino balenciaga",
            "brand": "Balenciaga",
            "size": "",
            "condition": "Nuovo senza cartellino",
            "price": 998.2,
            "factual_sentences": ["Zaino balenciaga unisex."],
            "accessory_notes": ["Ha la sua dustbag."],
            "flaw_notes": [],
        }

        block = build_prompt_characteristics_block(facts, pd.DataFrame())

        self.assertNotIn("Size or variant", block)

    def test_build_prompt_characteristics_block_translates_common_english_facts(self):
        facts = {
            "title": "La nuit de l'homme 100ml Yves Saint Laurent",
            "brand": "Yves Saint Laurent",
            "size": "",
            "condition": "Nuovo con cartellino",
            "price": 30.63,
            "factual_sentences": ["La nuit de l'homme 100ml Yves Saint Laurent Neuf sous blister"],
            "accessory_notes": ["Ha la sua dustbag", "Con scatola originale"],
            "flaw_notes": [],
        }

        block = build_prompt_characteristics_block(facts, pd.DataFrame(), language="english")

        self.assertIn("Condition: brand new with tags", block)
        self.assertIn("brand new, sealed in cellophane", block)
        self.assertIn("includes its dust bag", block)
        self.assertIn("includes the original box", block)

    def test_format_ollama_runtime_error_explains_memory_constraint(self):
        message = "Error: model requires more system memory (6.3 GiB) than is available (2.4 GiB)"

        formatted = format_ollama_runtime_error(message, model="llama3.1:8b")

        self.assertIn("cannot run 'llama3.1:8b'", formatted)
        self.assertIn("requires 6.3 GiB", formatted)
        self.assertIn("only 2.4 GiB", formatted)
        self.assertIn("use a smaller local model", formatted)

    def test_extract_gemini_response_text_reads_candidate_parts(self):
        payload = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"text": "First line."},
                            {"text": "Second line."},
                        ]
                    }
                }
            ]
        }

        text = extract_gemini_response_text(payload)

        self.assertEqual(text, "First line.\nSecond line.")

    def test_extract_gemini_finish_reason_reads_candidate_finish_reason(self):
        payload = {
            "candidates": [
                {
                    "finishReason": "MAX_TOKENS",
                    "content": {"parts": [{"text": "Partial draft"}]},
                }
            ]
        }

        finish_reason = extract_gemini_finish_reason(payload)

        self.assertEqual(finish_reason, "MAX_TOKENS")

    def test_gemini_output_looks_incomplete_for_cut_sentence(self):
        self.assertTrue(gemini_output_looks_incomplete("Presenting a superb Patek Philippe 5070P-001"))
        self.assertFalse(gemini_output_looks_incomplete("Beautiful watch in excellent condition. Message me for details."))

    def test_build_gemini_model_candidates_orders_fallbacks_once(self):
        candidates = build_gemini_model_candidates("gemini-2.5-pro")

        self.assertEqual(candidates, ["gemini-2.5-pro", "gemini-3.1-pro-preview", "gemini-2.5-flash"])

    def test_build_gemini_generation_config_disables_thinking_for_flash(self):
        config = build_gemini_generation_config("gemini-2.5-flash", max_output_tokens=768)

        self.assertEqual(config["maxOutputTokens"], 768)
        self.assertEqual(config["thinkingConfig"], {"thinkingBudget": 0})

    @mock.patch("experiments.old.reselling_process.create_description.generate.generate_description_with_gemini_api")
    def test_build_description_payload_uses_gemini_backend(self, mock_generate_gemini):
        query = pd.Series(
            {
                "Title": "Gucci Marmont Mini",
                "Brand": "Gucci",
                "Brand_norm": "gucci",
                "EmbedText": "gucci marmont mini | gucci | unica",
                "BlockKey": "gucci__marmont_mini",
                "ProductId": 123,
                "VariantId": 456,
                "Price": 1290,
                "item_id": "1",
                "SearchName": "borse",
                "Description": "Borsa usata una volta.",
                "Condition": "Ottime condizioni",
            }
        )
        corpus = pd.DataFrame(
            [
                {
                    "Title": "Gucci Marmont Mini",
                    "Brand": "Gucci",
                    "Brand_norm": "gucci",
                    "EmbedText": "gucci marmont mini | gucci | unica",
                    "BlockKey": "gucci__marmont_mini",
                    "ProductId": 123,
                    "VariantId": 456,
                    "Price": 1280,
                    "item_id": "2",
                    "SearchName": "borse",
                    "Description": "Soft leather with dust bag.",
                    "Condition": "Excellent condition",
                    "SourceKind": "full_scrape",
                }
            ]
        )
        mock_generate_gemini.return_value = ("Generated by Gemini", "prompt text")

        payload = build_description_payload(
            query,
            corpus=corpus,
            use_gemini_api=True,
            use_ollama=True,
            gemini_model="gemini-3.1-pro-preview",
            gemini_api_key="secret",
        )

        self.assertEqual(payload["generated_description"], "Generated by Gemini")
        self.assertEqual(payload["generation_mode"], "gemini_api")
        self.assertEqual(payload["model_name"], "gemini-3.1-pro-preview")
        mock_generate_gemini.assert_called_once()

    @mock.patch("experiments.old.reselling_process.create_description.generate.generate_description_with_gemini_api")
    def test_build_description_payload_falls_back_to_available_gemini_model_on_quota(self, mock_generate_gemini):
        query = pd.Series(
            {
                "Title": "Gucci Marmont Mini",
                "Brand": "Gucci",
                "Brand_norm": "gucci",
                "EmbedText": "gucci marmont mini | gucci | unica",
                "BlockKey": "gucci__marmont_mini",
                "ProductId": 123,
                "VariantId": 456,
                "Price": 1290,
                "item_id": "1",
                "SearchName": "borse",
                "Description": "Borsa usata una volta.",
                "Condition": "Ottime condizioni",
            }
        )
        corpus = pd.DataFrame(
            [
                {
                    "Title": "Gucci Marmont Mini",
                    "Brand": "Gucci",
                    "Brand_norm": "gucci",
                    "EmbedText": "gucci marmont mini | gucci | unica",
                    "BlockKey": "gucci__marmont_mini",
                    "ProductId": 123,
                    "VariantId": 456,
                    "Price": 1280,
                    "item_id": "2",
                    "SearchName": "borse",
                    "Description": "Soft leather with dust bag.",
                    "Condition": "Excellent condition",
                    "SourceKind": "full_scrape",
                }
            ]
        )
        mock_generate_gemini.side_effect = [
            RuntimeError("Gemini API request for 'gemini-3.1-pro-preview' failed: quota exceeded"),
            RuntimeError("Gemini API request for 'gemini-2.5-pro' failed: quota exceeded"),
            ("Generated by Flash", "prompt text"),
        ]

        payload = build_description_payload(
            query,
            corpus=corpus,
            use_gemini_api=True,
            gemini_model="gemini-3.1-pro-preview",
            gemini_api_key="secret",
        )

        self.assertEqual(payload["generated_description"], "Generated by Flash")
        self.assertEqual(payload["model_name"], "gemini-2.5-flash")
        self.assertEqual(mock_generate_gemini.call_count, 3)

    def test_llm_output_needs_fallback_for_listing_chatter_and_condition_upgrade(self):
        facts = {
            "condition": "Nuovo senza cartellino",
        }

        self.assertTrue(
            llm_output_needs_fallback(
                "This item is pristine and the listing on this platform shows its signature style.",
                facts,
            )
        )
        self.assertFalse(
            llm_output_needs_fallback(
                "Balenciaga backpack in nuovo senza cartellino condition, complete with dustbag.",
                facts,
            )
        )


if __name__ == "__main__":
    unittest.main()