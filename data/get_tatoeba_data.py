import os
import random
import sys

from tatoebatools import ParallelCorpus

"""
Usage:
  python get_tatoeba_data.py <out_dir> <languages>
Example:
  python get_tatoeba_data.py tatoeba/ fr it zh ja pt tr

"""

# Tatoeba uses ISO 639-3 codes, but we try to be consistent in using ISO 639-1, so we convert them here
ISO_6391_TO_6393 = {
    "bg": "bul",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "hi": "hin",
    "it": "ita",
    "zh": "cmn",
    "ja": "jpn",
    "pt": "por",
    "tr": "tur",
    "sw": "swh",
    "vi": "vie",
}
ISO_6393_TO_6391 = {v: k for k, v in ISO_6391_TO_6393.items()}


def main():
    out_dir = sys.argv[1]
    languages = sys.argv[2:]

    random.seed(42)

    language_pairs = [
        (lang_a, lang_b)
        for lang_a in languages
        for lang_b in [lang_b for lang_b in languages if lang_a < lang_b]
    ]
    print(language_pairs)

    for lang_a, lang_b in language_pairs:

        pair_data = [
            (a, b)
            for a, b in ParallelCorpus(
                ISO_6391_TO_6393[lang_a], ISO_6391_TO_6393[lang_b]
            )
        ]
        random.shuffle(pair_data)

        print(f"Num examples for ({lang_a}-{lang_b}): {len(pair_data)}")
        with open(
            os.path.join(out_dir, f"{lang_a}-{lang_b}.txt.{lang_a}"),
            "w",
            encoding="utf-8",
        ) as fa:
            with open(
                os.path.join(out_dir, f"{lang_a}-{lang_b}.txt.{lang_b}"),
                "w",
                encoding="utf-8",
            ) as fb:
                for sentence_a, sentence_b in pair_data:
                    if sentence_a.text.strip() and sentence_b.text.strip():
                        fa.write(f"{sentence_a.text.strip()}\n")
                        fb.write(f"{sentence_b.text.strip()}\n")


if __name__ == "__main__":
    main()
