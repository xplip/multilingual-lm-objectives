## POS Tagging Fine-Tuning Data

```
./get_ud_data.sh
```

## POS Tagging Evaluation

Note: We do not include Thai in Tatoeba and WikiMatrix due to lack of examples

### Get TedTalks2020 data
```
mkdir -p tedtalks2020
# wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/ted2020.tsv.gz -O tedtalks2020/ted2020.tsv 
gdown https://drive.google.com/uc?id=14pyiGRpk1rGV5B0mQp4sp_Bqd1nzyyzv -O tedtalks2020/ted2020.tsv
python preprocess_ted_talks.py tedtalks2020/ted2020.tsv tedtalks2020 fr it zh ja pt th tr
```

### Get tatoeba data
```
mkdir -p tatoeba
python get_tatoeba_data.py tatoeba/ fr it zh ja pt tr
```

### Get WikiMatrix data
```
mkdir -p wikimatrix
export LANGS="fr it zh ja pt tr"
./get_wikimatrix_data.sh
```

## XNLI Evaluation

### Get TedTalks2020 data
```
mkdir -p tedtalks2020
# wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/ted2020.tsv.gz -O tedtalks2020/ted2020.tsv 
gdown https://drive.google.com/uc?id=14pyiGRpk1rGV5B0mQp4sp_Bqd1nzyyzv -O tedtalks2020/ted2020.tsv
python preprocess_ted_talks.py tedtalks2020/ted2020.tsv tedtalks2020 bg es fr hi tr vi zh
```

### Get WikiMatrix data
```
mkdir -p wikimatrix
export LANGS="bg es fr hi tr vi zh"
./get_wikimatrix_data.sh
```