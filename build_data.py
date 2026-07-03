"""Build packed evaluation sequences + measurement-only filter masks.

Follows arXiv:2606.20936 (Li & Merrill) Section 6:
  - ALL:        every scored position
  - COPY5:      positions completing a repeated 5-gram in the visible prefix
  - NOCOPY:     positions NOT completing a repeated n-gram for any n<=4
                (equivalently: first in-window occurrence of the target token type)
  - CONTENT:    prose -> open-class content words (Noun/Verb/Adj/Adverb/Interjection,
                auxiliaries excluded); python -> identifiers/strings/comments/numbers
The paper's state filter is TOP-10∩NO-COPY; we approximate with CONTENT∩NOCOPY.

Masks are indexed by target position: mask[i] describes the token at position i,
scored as the prediction from prefix tokens[<i]. Position 0 is never scored.
"""

import itertools
import json
import re
import sys

import numpy as np

L = 2048           # window length in tokens
N_PROSE = 48       # windows from PG-19
N_WIKI = 16        # windows from wikitext-103 test
N_PY = 32          # windows from python files
COPY_MAX = 5

WORD_RE = re.compile(r"\w+|[^\w\s]")
AUX = {"be", "am", "is", "are", "was", "were", "been", "being",
       "have", "has", "had", "having", "do", "does", "did", "doing",
       "'s", "'re", "'ve", "'d", "'m", "'ll", "n't"}
CONTENT_PREFIX = ("NN", "VB", "JJ", "RB", "UH")


def copy_masks(ids):
    """ids: list[int] length L. Returns (copy5, nocopy) bool arrays of length L."""
    L_ = len(ids)
    copy5 = np.zeros(L_, dtype=bool)
    nocopy = np.zeros(L_, dtype=bool)
    seen1 = set()
    seen5 = set()
    for i in range(L_):
        tok = ids[i]
        nocopy[i] = tok not in seen1
        if i >= COPY_MAX - 1:
            gram = tuple(ids[i - COPY_MAX + 1: i + 1])
            copy5[i] = gram in seen5
            seen5.add(gram)
        seen1.add(tok)
    return copy5, nocopy


def prose_content_charmask(text):
    """Char-level bool mask: True where char belongs to a content word."""
    import nltk
    words, spans = [], []
    for m in WORD_RE.finditer(text):
        words.append(m.group())
        spans.append((m.start(), m.end()))
    mask = np.zeros(len(text), dtype=bool)
    B = 20000
    for k in range(0, len(words), B):
        tags = nltk.pos_tag(words[k:k + B])
        for (w, tag), (s, e) in zip(tags, spans[k:k + B]):
            if tag.startswith(CONTENT_PREFIX) and w.lower() not in AUX:
                mask[s:e] = True
    return mask


def python_content_charmask(text):
    """True where char belongs to identifier/string/comment/number."""
    import io
    import keyword
    import tokenize
    mask = np.zeros(len(text), dtype=bool)
    lines = text.splitlines(keepends=True)
    line_start = np.cumsum([0] + [len(l) for l in lines])
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            keep = (
                (tok.type == tokenize.NAME and not keyword.iskeyword(tok.string))
                or tok.type in (tokenize.STRING, tokenize.COMMENT, tokenize.NUMBER)
            )
            if keep:
                (r1, c1), (r2, c2) = tok.start, tok.end
                s = line_start[r1 - 1] + c1
                e = line_start[r2 - 1] + c2
                mask[s:e] = True
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return mask


def windows_from_doc(text, tokenizer, content_fn):
    """Tokenize one document, yield (ids[L], content_mask[L]) full windows."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) < L:
        return
    cmask_char = content_fn(text)
    for k in range(len(ids) // L):
        w_ids = ids[k * L:(k + 1) * L]
        w_off = offs[k * L:(k + 1) * L]
        cm = np.array([cmask_char[s:e].any() if e > s else False
                       for s, e in w_off], dtype=bool)
        yield np.array(w_ids, dtype=np.int32), cm


def collect(doc_iter, tokenizer, content_fn, n_target, domain):
    ids_list, content_list, copy5_list, nocopy_list = [], [], [], []
    for text in doc_iter:
        for w_ids, cm in windows_from_doc(text, tokenizer, content_fn):
            c5, nc = copy_masks(w_ids.tolist())
            ids_list.append(w_ids)
            content_list.append(cm)
            copy5_list.append(c5)
            nocopy_list.append(nc)
            if len(ids_list) >= n_target:
                break
        print(f"  [{domain}] {len(ids_list)}/{n_target} windows", flush=True)
        if len(ids_list) >= n_target:
            break
    return (np.stack(ids_list), np.stack(content_list),
            np.stack(copy5_list), np.stack(nocopy_list))


def pg19_docs():
    import datasets
    ds = datasets.load_dataset("emozilla/pg19", split="test", streaming=True)
    for ex in ds:
        yield ex["text"]


def wiki_docs():
    import datasets
    ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    yield "".join(ds["text"])


def py_docs():
    import pathlib
    root = pathlib.Path(sys.prefix) / "lib"
    files = sorted(root.rglob("*.py"), key=lambda p: p.name)
    # deterministic shuffle, keep mid-sized real source files
    rng = np.random.RandomState(0)
    files = [f for f in files if 8_000 < f.stat().st_size < 80_000]
    rng.shuffle(files)
    for f in files:
        try:
            yield f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")

    out = {}
    for domain, docs, cfn, n in [
        ("pg19", pg19_docs(), prose_content_charmask, N_PROSE),
        ("wiki", wiki_docs(), prose_content_charmask, N_WIKI),
        ("python", py_docs(), python_content_charmask, N_PY),
    ]:
        print(f"building {domain} ...", flush=True)
        ids, content, copy5, nocopy = collect(docs, tokenizer, cfn, n, domain)
        out[f"{domain}_ids"] = ids
        out[f"{domain}_content"] = content
        out[f"{domain}_copy5"] = copy5
        out[f"{domain}_nocopy"] = nocopy
        stats = {
            "windows": int(ids.shape[0]),
            "copy5_frac": float(copy5[:, 1:].mean()),
            "nocopy_frac": float(nocopy[:, 1:].mean()),
            "content_frac": float(content[:, 1:].mean()),
            "content_nocopy_frac": float((content & nocopy)[:, 1:].mean()),
        }
        print(f"  {domain}: {json.dumps(stats)}", flush=True)

    np.savez_compressed("eval_data.npz", **out)
    print("saved eval_data.npz")


if __name__ == "__main__":
    main()
