"""Eval-pack construction: freeze once per (tokenizer, window length, corpus).

A pack is a dict of numpy arrays:
    ids      (W, L)  int32   token ids, documents packed with EOS separators
    copy5    (W, L)  bool    completes a repeated 5-gram
    nocopy   (W, L)  bool    first in-window occurrence of the token type
    content  (W, L)  bool    optional content-word mask (if content_fn given)

Masks are recomputable from ids; they are stored for convenience. Packs are
TOKENIZER-SPECIFIC: rebuild when the tokenizer changes, then freeze for the
whole campaign so numbers stay comparable.
"""

import numpy as np

from .masks import copy_n_mask, nocopy_mask


def build_pack(doc_iter, tokenizer, n_windows, window_len=2048,
               content_fn=None, eos_id=None):
    """Pack documents from doc_iter (iterable of str) into full windows.

    tokenizer: HF-style, called as tokenizer(text, add_special_tokens=False,
    return_offsets_mapping=bool(content_fn)).
    content_fn: optional text -> bool char mask (True on content-word chars).
    """
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    L = window_len
    stream_ids, stream_content = [], []
    ids_list, content_list = [], []
    for text in doc_iter:
        enc = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=content_fn is not None)
        stream_ids.extend(enc["input_ids"])
        if content_fn is not None:
            cmask = content_fn(text)
            stream_content.extend(
                cmask[s:e].any() if e > s else False
                for s, e in enc["offset_mapping"])
        else:
            stream_content.extend([False] * len(enc["input_ids"]))
        stream_ids.append(eos_id)
        stream_content.append(False)
        while len(stream_ids) >= L:
            ids_list.append(np.array(stream_ids[:L], dtype=np.int32))
            content_list.append(np.array(stream_content[:L], dtype=bool))
            del stream_ids[:L], stream_content[:L]
        if len(ids_list) >= n_windows:
            break
    if len(ids_list) < n_windows:
        raise ValueError(f"corpus exhausted at {len(ids_list)}/{n_windows} windows")
    ids = np.stack(ids_list[:n_windows])
    pack = {
        "ids": ids,
        "content": np.stack(content_list[:n_windows]),
        "copy5": np.stack([copy_n_mask(r, 5) for r in ids]),
        "nocopy": np.stack([nocopy_mask(r) for r in ids]),
    }
    return pack


def save_pack(path, pack):
    np.savez_compressed(path, **pack)


def load_pack(path):
    return dict(np.load(path))
