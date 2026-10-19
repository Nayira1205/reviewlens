"""
preprocess.py - turns raw review text into clean text a model can learn from.

Why this file exists
--------------------
Raw reviews are messy: "SUPERRRR!!! READ MORE", "dont buy", "Not good :(".
A model sees "Good" and "good" as two different words, so we first make the
text consistent (this is called text normalisation / preprocessing).

The one rule that matters for SENTIMENT analysis
-------------------------------------------------
Standard stop-word lists delete words like "not" and "no". That would turn
"not good" into "good" and flip the meaning. So we build our own stop-word
list that KEEPS negations (and words like "very", "but").

Usage
-----
    from src.preprocess import clean_text
    clean_text("Not good, battery doesn't last!!!")   # -> 'not good battery not last'
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def _nltk_has(*paths):
    """True if NLTK can find any of these resources (as a folder or a .zip file)."""
    for path in paths:
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            pass
    return False


# Download the NLTK word lists only if they are missing on this computer.
# (NLTK often keeps WordNet as a .zip file, so we check for both forms.)
if not _nltk_has("corpora/stopwords", "corpora/stopwords.zip"):
    nltk.download("stopwords", quiet=True)
if not _nltk_has("corpora/wordnet", "corpora/wordnet.zip"):
    nltk.download("wordnet", quiet=True)

# Words we must NOT remove, because they carry sentiment.
KEEP = {
    "no", "nor", "not", "very", "too", "but",
    "don", "didn", "doesn", "isn", "wasn", "weren", "won", "wouldn",
    "couldn", "shouldn", "hasn", "haven", "hadn", "aren", "ain",
}

# NLTK's English stop words (the, is, and, ...) minus the words we keep.
STOP_WORDS = set(stopwords.words("english")) - KEEP

# Many reviewers write "dont" or "cant" without the apostrophe.
# We expand them so they become the same signal as "not".
NEG_FIX = {
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "cant": "can not", "wont": "will not", "isnt": "is not",
    "wasnt": "was not", "arent": "are not", "couldnt": "could not",
    "shouldnt": "should not", "wouldnt": "would not",
    "havent": "have not", "hasnt": "has not",
}

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Return a cleaned version of one review (a lowercase string of words)."""
    text = str(text).lower()                            # 1. lowercase ("Good" -> "good")
    text = re.sub(r"read more", " ", text)              # 2. scraping leftover
    text = re.sub(r"https?://\S+|<[^>]+>", " ", text)   # 3. remove URLs and HTML tags
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't\b", " not", text)               # 4. don't -> do not
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)          # 5. goooood -> good
    text = re.sub(r"[^a-z\s]", " ", text)               # 6. keep letters only
    tokens = text.split()                               # 7. split into words (tokenization)
    tokens = [w for t in tokens for w in NEG_FIX.get(t, t).split()]   # 8. dont -> do not
    tokens = [t for t in tokens if t not in STOP_WORDS]               # 9. drop stop words
    # 10. lemmatize: reduce words to dictionary form (batteries -> battery).
    #     Run as verb first, then noun, because the default assumes nouns.
    tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(t, pos="v")) for t in tokens]
    return " ".join(tokens)
