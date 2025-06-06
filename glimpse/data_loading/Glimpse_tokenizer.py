import re
import spacy
import importlib
import nltk

############################################
### CHANGE THIS LINE TO CHOOSE TOKENIZER ###
ORIGINAL_TOKENIZER = False
############################################

try:
    importlib.util.find_spec("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def glimpse_tokenizer(text: str) -> list:
    
    # If the original tokenizer is set to True, use the original tokenizer
    if ORIGINAL_TOKENIZER:
        return original_tokenizer(text)
    
    # else, use the new tokenizer
    else: 
        
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
    
# reuse the original glimpse tokenizer
# def glimpse_tokenizer(text: str) -> list:    
#     return tokenize_sentences(text)

# Default glimpse tokenizer from the original code
def original_tokenizer(text: str) -> list:
    """
    Tokenizes the input text into sentences.
    
    @param text: The input text to be tokenized
    @return: A list of tokenized sentences
    """
    text = text.replace('-----', '\n')
    sentences = nltk.sent_tokenize(text)
    # remove empty sentences
    sentences = [sentence for sentence in sentences if sentence != ""]
    
    return sentences