"""Word clouds of repeated 5-grams: plain COPY5 vs HARD-COPY5 (tau=2).

Each cloud entry is a decoded 5-gram at a copy event, weighted by how often it
occurs across all four domains. HARD keeps only events whose first occurrence
cost the selector transformer (Pythia-1.4b) > 2 nats.
"""

from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import analyze_large as al
from hard_copy import N, first_occurrence_index

TAU = 2.0
DOMAINS = ["pg19", "wiki", "climbmix"]   # prose only: python code fragments
                                         # otherwise dominate both clouds
SURFACE, INK = "#fcfcfb", "#0b0b0b"


def collect():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    sel = al.load_nll("pythia14")
    all_c, hard_c = Counter(), Counter()
    for dom in DOMAINS:
        ids = al.masks_for(dom)[1]
        for w in range(ids.shape[0]):
            row = ids[w].tolist()
            fo = first_occurrence_index(row)
            for i in range(N - 1, len(row)):
                j = fo[i]
                if j < 0:
                    continue
                text = tok.decode(row[i - N + 1:i + 1]).strip()
                text = " ".join(text.split())
                # undo wikitext escapes; drop markup/punctuation-only grams
                text = text.replace("@-@", "-").replace("@.@", ".").replace("@,@", ",")
                words = [w for w in text.split() if sum(c.isalpha() for c in w) >= 2]
                if len(text) < 3 or len(words) < 2:
                    continue
                all_c[text] += 1
                if j >= 1 and sel[dom][w, j - 1] > TAU:
                    hard_c[text] += 1
    return all_c, hard_c


def shade_func(base_rgb):
    def f(word, font_size, position, orientation, random_state=None, **kw):
        rng = np.random.RandomState(abs(hash(word)) % 2**31)
        k = 0.55 + 0.45 * rng.rand()          # 55-100% intensity
        r, g, b = [int(c * k) for c in base_rgb]
        return f"rgb({r},{g},{b})"
    return f


def cloud(counter, base_rgb):
    return WordCloud(width=1200, height=700, background_color=SURFACE,
                     max_words=110, prefer_horizontal=0.95,
                     collocations=False, color_func=shade_func(base_rgb),
                     margin=4).generate_from_frequencies(counter)


def main():
    all_c, hard_c = collect()
    print(f"distinct 5-grams: all={len(all_c)}, hard={len(hard_c)}")
    print("top ALL:", all_c.most_common(8))
    print("top HARD:", hard_c.most_common(8))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.03)
    for ax, cnt, rgb, title in [
        (axes[0], all_c, (42, 120, 214), f"Plain COPY5 — every repeated 5-gram "
                                         f"({sum(all_c.values())} events)"),
        (axes[1], hard_c, (74, 58, 167), f"HARD-COPY5 (τ=2) — context-specific repeats "
                                         f"({sum(hard_c.values())} events)"),
    ]:
        ax.imshow(cloud(cnt, rgb), interpolation="bilinear")
        ax.set_title(title, fontsize=12, color=INK, pad=10)
        ax.axis("off")
    fig.suptitle("What the two filters actually select (size ∝ frequency, prose domains)",
                 y=0.97, fontsize=13, color=INK)
    fig.savefig("fig_wordclouds.png", dpi=150, facecolor=SURFACE)
    print("saved fig_wordclouds.png")


if __name__ == "__main__":
    main()
