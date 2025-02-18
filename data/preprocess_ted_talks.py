import csv
import os
import sys

import pandas as pd

"""
1. Download data:
wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/ted2020.tsv.gz
2. Preprocess with this script:
python preprocess_ted_talks.py ted2020.tsv.gz <out_dir> <languages>
"""


def main():
    in_file = sys.argv[1]
    out_dir = sys.argv[2]
    languages = sys.argv[3:]

    ted_data_df = pd.read_csv(
        in_file,
        sep="\t",
        keep_default_na=False,
        encoding="utf8",
        quoting=csv.QUOTE_NONE,
    )
    language_pairs = [
        (lang_a, lang_b)
        for lang_a in languages
        for lang_b in [lang_b for lang_b in languages if lang_a < lang_b]
    ]
    print(language_pairs)
    shuffled_df = ted_data_df.sample(frac=1).reset_index(drop=True)
    for pair in language_pairs:
        lang_a = pair[0]
        lang_b = pair[1]
        lang_a_data = shuffled_df[lang_a].to_list()
        lang_b_data = shuffled_df[lang_b].to_list()
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
                for line_a, line_b in zip(lang_a_data, lang_b_data):
                    if line_a.strip() and line_b.strip():
                        fa.write(f"{line_a}\n")
                        fb.write(f"{line_b}\n")


if __name__ == "__main__":
    main()
