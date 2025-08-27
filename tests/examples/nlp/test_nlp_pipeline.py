import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from examples.nlp.nlp_pipeline import NLPPipeline


def test_pipeline_runs():
    pipeline = NLPPipeline()
    texts = ["<p>OpenAI creates AI.</p>"]
    results = pipeline.run_all(texts)
    assert set(results.keys()) == {"classification", "ner", "sentiment", "generation"}
    assert len(results["classification"]) == 1
    assert len(results["generation"]) == 1
