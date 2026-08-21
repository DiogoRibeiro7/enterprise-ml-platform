import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[3]))

# This example needs the NLP extras and downloads HuggingFace models on first
# run. Skip rather than fail the whole collection when they are absent.
pytest.importorskip("langdetect", reason="NLP example extras not installed")
pytest.importorskip("transformers", reason="NLP example extras not installed")

from examples.nlp.nlp_pipeline import NLPPipeline  # noqa: E402


def test_pipeline_runs():
    pipeline = NLPPipeline()
    texts = ["<p>OpenAI creates AI.</p>"]
    results = pipeline.run_all(texts)
    assert set(results.keys()) == {"classification", "ner", "sentiment", "generation"}
    assert len(results["classification"]) == 1
    assert len(results["generation"]) == 1
