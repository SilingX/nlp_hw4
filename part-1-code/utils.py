import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    detok = TreebankWordDetokenizer()

    qwerty_neighbors = {
        "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "ersfcx", "e": "wsdr",
        "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
        "k": "ijolm,", "l": "kop;.", "m": "njk,", "n": "bhjm", "o": "iklp",
        "p": "ol;", "q": "was", "r": "edft", "s": "awedxz", "t": "rfgy",
        "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"
    }

    def is_alpha_word(tok: str) -> bool:
        return tok.isalpha()

    def preserve_case_like(ref: str, w: str) -> str:
        if ref.isupper():
            return w.upper()
        if len(ref) > 0 and ref[0].isupper():
            return w.capitalize()
        return w

    def pick_synonym(tok: str):
        """Pick a synonym for token using WordNet; return None if unavailable."""
        lemma = tok.lower()
        syns = wordnet.synsets(lemma)
        if not syns:
            return None
        cands = []
        for s in syns:
            for l in s.lemmas():  # type: ignore 
                name = l.name()
                if "_" in name or name.lower() == lemma:
                    continue
                cands.append(name)
        if not cands:
            return None
        return preserve_case_like(tok, random.choice(cands))

    def apply_typo(tok: str) -> str:
        """Replace one character with a QWERTY-neighbor key."""
        if len(tok) < 3 or not tok.isalpha():
            return tok
        idxs = list(range(len(tok)))
        random.shuffle(idxs)
        for i in idxs:
            ch = tok[i]
            low = ch.lower()
            neighbors = qwerty_neighbors.get(low, "")
            if not neighbors:
                continue
            rep = random.choice(list(neighbors))
            rep = rep.upper() if ch.isupper() else rep
            return tok[:i] + rep + tok[i + 1:]
        return tok

    p_typo = 0.13
    p_syn = 0.2

    text = example.get("text", "")
    if not text:
        return example

    tokens = word_tokenize(text)
    new_tokens = []
    for tok in tokens:
        if is_alpha_word(tok):
            r = random.random()
            if r < p_typo:
                new_tok = apply_typo(tok)
            elif r < p_typo + p_syn:
                syn = pick_synonym(tok)
                new_tok = syn if syn else tok
            else:
                new_tok = tok
        else:
            new_tok = tok
        new_tokens.append(new_tok)

    example["text"] = detok.detokenize(new_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
