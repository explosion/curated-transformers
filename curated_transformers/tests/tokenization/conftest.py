import pytest
import spacy


@pytest.fixture(scope="module")
def sample_docs():
    nlp = spacy.blank("en")
    doc1 = nlp.make_doc("I saw a girl with a telescope.")
    doc2 = nlp.make_doc("Today we will eat poké bowl.")
    return [doc1, doc2]
