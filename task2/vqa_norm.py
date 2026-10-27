"""VQA answer normalization and soft accuracy.

The normalization follows the official VQA evaluation script (vqaEval.py from
github.com/GT-Vision-Lab/VQA): punctuation handling, number words to digits,
article removal and contraction fixes. It is applied to training targets,
references and predictions alike.
"""
import re
from collections import Counter

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll",
    "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
           "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")


def _punctuation(text):
    out = text
    for p in PUNCT:
        if (p + " " in text or " " + p in text) or COMMA_STRIP.search(text) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return PERIOD_STRIP.sub("", out)


def _digits_articles(text):
    words = [NUMBERS.get(w, w) for w in text.lower().split()]
    words = [CONTRACTIONS.get(w, w) for w in words if w not in ARTICLES]
    return " ".join(words)


def normalize(ans):
    ans = ans.replace("\n", " ").replace("\t", " ").strip()
    return _digits_articles(_punctuation(ans))


def consensus(raw_answers):
    """(normalized consensus answer, its vote count, most common raw string in that group)."""
    norm = [normalize(a) for a in raw_answers]
    top, votes = Counter(norm).most_common(1)[0]
    raw = Counter(a for a, n in zip(raw_answers, norm) if n == top).most_common(1)[0][0]
    return top, votes, raw


def soft_accuracy(pred, raw_answers):
    """Local VQA accuracy: mean over the ten leave-one-out subsets of min(1, matches / 3)."""
    p = normalize(pred)
    refs = [normalize(a) for a in raw_answers]
    total = 0.0
    for i in range(len(refs)):
        others = sum(1 for j, r in enumerate(refs) if j != i and r == p)
        total += min(1.0, others / 3)
    return total / len(refs)
