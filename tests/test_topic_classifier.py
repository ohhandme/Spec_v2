"""Focused tests for the frozen six-topic UltraChat prompt instrument."""

import unittest

from topic_classifier import TOPICS, classify_topic


class TopicClassifierTest(unittest.TestCase):
    def test_each_topic_has_explicit_cue(self) -> None:
        examples = {
            "code": "Write a Python program that sorts a list of names.",
            "creative_writing": "Write a short story about a lost astronomer.",
            "knowledge_qa": "What is the role of photosynthesis in an ecosystem?",
            "summarization_transformation": "Summarize the following report in three sentences.",
            "practical_howto": "Give me a recipe for vegetable soup.",
            "business_marketing": "Develop a marketing plan for a new bakery.",
        }
        self.assertEqual(set(examples), set(TOPICS))
        for topic, prompt in examples.items():
            with self.subTest(topic=topic):
                decision = classify_topic(prompt)
                self.assertEqual(decision.label, topic)
                self.assertIn(topic, decision.hits)

    def test_cross_topic_abstains(self) -> None:
        decision = classify_topic("Write a Python program to create a marketing plan.")
        self.assertEqual(decision.label, "ambiguous")
        self.assertEqual(set(decision.hits), {"code", "business_marketing"})

    def test_uncued_prompt_abstains(self) -> None:
        self.assertEqual(classify_topic("Hello there!").label, "unclassified")

    def test_normalization_and_evidence(self) -> None:
        decision = classify_topic("Ｗｒｉｔｅ   a   Ｐｙｔｈｏｎ   program")
        self.assertEqual(decision.label, "code")
        self.assertTrue(decision.hits["code"])

    def test_non_text_rejected(self) -> None:
        with self.assertRaises(TypeError):
            classify_topic(None)  # type: ignore[arg-type]

    def test_context_and_recommendations_do_not_create_false_topic(self) -> None:
        self.assertEqual(classify_topic(
            "Method: pour tea and cool it. Why are superfoods popular? "
            "Could you suggest another ingredient?").label, "unclassified")
        self.assertEqual(classify_topic(
            "What are some good walking trails to explore?").label, "unclassified")
        self.assertEqual(classify_topic(
            "Digital marketing is essential to your marketing strategy."
        ).label, "unclassified")

    def test_explicit_wrapped_question_is_read_before_context(self) -> None:
        decision = classify_topic(
            "Generate response to the question/instruction based on a piece of "
            "given material Question/Instruction: What is photosynthesis? "
            "Given material: Photosynthesis is a process.")
        self.assertEqual(decision.label, "knowledge_qa")


if __name__ == "__main__":
    unittest.main()
