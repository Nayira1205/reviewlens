"""
aspects.py - find WHAT a review talks about (battery, camera, price, ...).

Why this file exists
--------------------
A review like "camera is great but battery is bad" is positive about one thing
and negative about another. A single score for the whole review hides this.
Aspect-based sentiment analysis (ABSA) scores each ASPECT separately.

How it works (a simple, explainable method)
-------------------------------------------
1. Split a review into short pieces (clauses) at punctuation and at words like
   "but" and "however":      "camera great" | "battery bad"
2. Check each clause against keyword patterns for each aspect.
3. Later (in the notebook) the sentiment model scores each clause on its own.

This is a keyword (dictionary) method. It is easy to explain and check, but it
only finds aspects whose keywords we listed. See the limitations in notebook 04.

Usage
-----
    from src.aspects import extract_aspect_clauses
    extract_aspect_clauses("Camera is great but battery is bad")
    # -> [('camera', 'camera is great'), ('battery', 'battery is bad')]
"""

import re

# aspect name -> regular expression of keywords (\b = word boundary)
ASPECTS = {
    "battery":      r"\bbatter(y|ies)\b|\bbackup\b|\bcharging\b|\bcharges?\b|\bcharged\b",
    "camera":       r"\bcamera\b|\bselfie\b|\bphotos?\b",
    "display":      r"\bdisplay\b|\bscreen\b|\bbrightness\b|\bresolution\b|\bpicture\b",
    "sound":        r"\bsound\b|\baudio\b|\bbass\b|\bvolume\b|\bspeakers?\b|\bmic\b|\bmusic\b",
    "performance":  r"\bperformance\b|\bspeed\b|\blags?\b|\blaggy\b|\bhang(s|ing)?\b|\bprocessor\b|\bgaming\b|\bsmooth(ly)?\b",
    "price":        r"\bprice\b|\bvalue for money\b|\bmoney\b|\bcost\b|\bbudget\b|\bworth\b|\bcheap\b|\bexpensive\b|\baffordable\b",
    "build quality": r"\bbuild\b|\bmaterial\b|\bplastic\b|\bsturdy\b|\bdurab\w*\b|\bstrong\b|\bquality\b",
    "design":       r"\bdesign\b|\blooks?\b|\blooking\b|\bcolou?rs?\b|\bsize\b|\bweight\b|\bslim\b|\bstylish\b|\bcompact\b",
    "connectivity": r"\bbluetooth\b|\bwi-?fi\b|\bconnect\w*\b|\bsignal\b|\bpairing\b",
    "delivery":     r"\bdeliver\w*\b|\bpackag\w*\b|\bpacking\b|\bpacked\b|\bseller\b|\bcourier\b",
    "heating":      r"\bheat\w*\b|\bhot\b",
    "cooling":      r"\bcooling\b|\bair ?flow\b",
    "noise":        r"\bnois(e|y)\b|\bsilent\b|\bquiet\b",
}

_COMPILED = {name: re.compile(pattern) for name, pattern in ASPECTS.items()}

# Pieces of text that start a new clause.
_STRONG_SPLIT = re.compile(r"[.,;!?]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bwhile\b|\botherwise\b|\bexcept\b")

# "sound quality" is about SOUND, not build quality, so we shorten these phrases first.
_QUALITY_PHRASE = re.compile(r"\b(sound|audio|mic|picture|video|camera|display|screen|print)\s+quality\b")


def find_aspects(clause):
    """Return the list of aspect names mentioned in one clause."""
    clause = _QUALITY_PHRASE.sub(r"\1", clause.lower())
    return [name for name, pattern in _COMPILED.items() if pattern.search(clause)]


def split_clauses(text):
    """Split a review into short clauses.

    First we split at punctuation and at words like "but". Then we also split
    at " and " - but only when BOTH sides are real phrases (2+ words) that
    mention an aspect, so that "sound and picture quality is good" is not cut
    in half.
    """
    clauses = []
    for piece in _STRONG_SPLIT.split(str(text).lower()):
        piece = piece.strip()
        if not piece:
            continue
        parts = piece.split(" and ")
        # split only if every part is a real phrase (2+ words) that mentions an aspect
        if len(parts) > 1 and all(len(p.split()) >= 2 and find_aspects(p) for p in parts):
            clauses.extend(p.strip() for p in parts)
        else:
            clauses.append(piece)
    return clauses


def extract_aspect_clauses(text):
    """Return a list of (aspect, clause) pairs for one review."""
    pairs = []
    for clause in split_clauses(text):
        for aspect in find_aspects(clause):
            pairs.append((aspect, clause))
    return pairs