import os
import re
import numpy as np
import tiktoken

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# load env variables
load_dotenv()

# init client instance
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def split_sentences(text):
    """Splits text into sentences."""
    assert isinstance(text, str), f"Should be a type string. Got {type(text)}."
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def count_tokens(text, model="gpt-3.5-turbo"):
    """Compute number of tokens."""
    assert isinstance(text, str), f"Should be a type string. Got {type(text)}."
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_embeddings(texts):
    """Get embeddings for a list of strings or a single string."""

    assert isinstance(texts, (str, list)), f"Got {type(texts)}."

    # handle string as list
    if isinstance(texts, str):
        texts = [texts]

    # create embeddings
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def create_references(summary, source_text, threshold=0.7):
    """Create references for summary to the source document."""

    assert isinstance(summary, str), f"Should be a type string. Got {type(summary)}."
    assert isinstance(
        source_text, str
    ), f"Should be a type string. Got {type(source_text)}."

    # split text into sentences
    summary_sentences = split_sentences(summary)
    source_sentences = split_sentences(source_text)

    # create embeddings in one call to reduce API usage
    all_sentences = summary_sentences + source_sentences
    all_embeddings = np.array(get_embeddings(all_sentences))

    # split
    summary_embeddings = all_embeddings[: len(summary_sentences)]
    source_embeddings = all_embeddings[len(summary_sentences) :]

    # compute cosine similarity
    similarity_matrix = cosine_similarity(summary_embeddings, source_embeddings)

    # match each summary sentence with source sentence of the highest score
    references = []
    for i, summary_sentence in enumerate(summary_sentences):
        best_match_idx = np.argmax(similarity_matrix[i])
        best_match_score = similarity_matrix[i][best_match_idx]

        # discard references below threshold
        if best_match_score >= threshold:
            best_match_sentence = source_sentences[best_match_idx]
            references.append(
                {
                    "summary_sentence": summary_sentence,
                    "matched_source_sentence": best_match_sentence,
                    "similarity_score": round(best_match_score, 2),
                }
            )
    return references


def summarize_notes(text, clinical_role="general"):
    """Summarize the text for a user defined role."""

    assert isinstance(text, str), f"Should be a type string. Got {type(text)}."

    prompt = f"""
    You are provided with a text input. 
    Your task is to summarize the text for a {clinical_role} clinician using only the information in the text. 
    Do not add or create any information beyond what is provided in the text. 
    If the text does not contain any medical or clinical information, output exactly "<|no_medical_text|>" and nothing else. 
    Otherwise, produce a concise paragraph summary without any bullet points or extra formatting. 
    Include details such as relevant dates, patient demographics, symptoms, past medical history and findings, and recommended next steps when applicable. 
    Make sure to highlight any critical findings (like abnormal lab results, imaging abnormalities, or urgent clinical concerns).
    Ensure that the summary is concise, easy to read, and formatted to quickly convey important information to a {clinical_role} clinician.

    Here is the text to be summarized:
    {text}
    """

    # dynamically set max tokens
    # todos:
    # - handle cases when prompt is too long (exceeds context window) by truncating or telling user
    # - make sure max tokens is not greater than max model tokens
    prompt_tokens = count_tokens(prompt, model="gpt-3.5-turbo")
    buffer, max_model_tokens = 100, 4096
    available_tokens = max_model_tokens - prompt_tokens - buffer
    max_tokens = max(200, available_tokens)

    try:
        # run summarizer
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at summarizing clinical notes.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=max_tokens,
        )

        # get summary text
        summary = response.choices[0].message.content.strip()

        # get tokens
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0

        # create references to sourtce for summary
        references = create_references(summary, text, threshold=0.7)

        return summary, tokens, references

    except Exception as e:
        raise Exception("Error generating summary: " + str(e))
