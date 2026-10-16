import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

# Words we must NOT remove, because they carry sentiment
KEEP = {"no", "nor", "not", "very", "too", "but",
        "don", "didn", "doesn", "isn", "wasn", "weren", "won", "wouldn",
        "couldn", "shouldn", "hasn", "haven", "hadn", "aren", "ain"}

STOP_WORDS = set(stopwords.words("english")) - KEEP
lemmatizer = WordNetLemmatizer()

NEG_FIX = {"dont": "do not", "doesnt": "does not", "didnt": "did not",
           "cant": "can not", "wont": "will not", "isnt": "is not",
           "wasnt": "was not", "arent": "are not", "couldnt": "could not",
           "shouldnt": "should not", "wouldnt": "would not",
           "havent": "have not", "hasnt": "has not"}

def clean_text(text):
    text = str(text).lower()                           # case folding
    text = re.sub(r"read more", " ", text)             # scraping leftover
    text = re.sub(r"https?://\S+|<[^>]+>", " ", text)  # URLs and HTML tags
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't\b", " not", text)              # don't -> do not
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)         # goooood -> good
    text = re.sub(r"[^a-z\s]", " ", text)              # keep letters only
    tokens = text.split()                              # tokenization
    tokens = [w for t in tokens for w in NEG_FIX.get(t, t).split()]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(lemmatizer.lemmatize(t, pos="v")) for t in tokens]
    return " ".join(tokens)