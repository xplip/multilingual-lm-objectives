#!/bin/bash

#export LANGS="fr it zh ja pt tr"
#export LANGS="en es fr bg vi tr hi zh"

for lang_a in $LANGS; do
    for lang_b in $LANGS; do
        if [[ "$lang_a" < "$lang_b" ]]; then
            wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.${lang_a}-${lang_b}.tsv.gz -O wikimatrix/${lang_a}-${lang_b}.tsv.gz
            python extract.py \
                --tsv wikimatrix/${lang_a}-${lang_b}.tsv.gz \
                --bitext wikimatrix/${lang_a}-${lang_b}.txt \
                --src-lang ${lang_a} --trg-lang ${lang_b} \
                --threshold 1.04
            rm wikimatrix/${lang_a}-${lang_b}.tsv.gz
        fi
    done
done