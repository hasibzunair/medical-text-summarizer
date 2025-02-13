import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, src_path)

from predictor import split_sentences


def test_split_sentences():
    text = "I am a machine learning engineer. And I am building a medical text summarizer. This is fun!"
    sentences = split_sentences(text)
    expected = [
        "I am a machine learning engineer.",
        "And I am building a medical text summarizer.",
        "This is fun!",
    ]
    assert sentences == expected, f"Should be {expected}, but got {sentences}."
