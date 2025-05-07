import re
import spacy
import importlib

try:
    importlib.util.find_spec("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def glimpse_tokenizer(text: str) -> list:
    # More general-purpose tokenizer that handles both natural paragraph text and structured reviews.

    # Normalize long dashes
    text = re.sub(r"[-]{2,}", "\n", text)

    # Keep line breaks meaningful (but fallback to sentence splitting)
    chunks = re.split(r"\n+", text)
    sentences = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Section headers and bullets become single “sentences”
        if re.match(r"^(Summary|Strengths?|Weaknesses?|Minor)\s*:?", chunk, re.IGNORECASE):
            sentences.append(chunk)
            continue

        if re.match(r"^(\d+(\.\d+)*\.|-)\s+.+", chunk):
            sentences.append(chunk)
            continue

        # Otherwise, apply SpaCy sentence splitting
        doc = nlp(chunk)
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])

    return sentences