from rag_explorer.pipeline import RagPipeline
import unittest


class PipelineTests(unittest.TestCase):
    def test_pipeline_answers_with_citations(self):
        answer = RagPipeline(data_dir="data/raw", top_k=2).ask("What is RAG?")

        self.assertTrue(answer.citations)
        self.assertIn("retriev", answer.answer.lower())


if __name__ == "__main__":
    unittest.main()
