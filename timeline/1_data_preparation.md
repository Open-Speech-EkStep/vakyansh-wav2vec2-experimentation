Objective:

To create a ASR using 40 hours of competition data by IITM

# Step 1: Working on Data

## Step 1.1: Get the data
Link: https://storage.googleapis.com/vakyaansh-open-models/ieee/data/hindi_iitm_challenge.zip

```
mkdir data
cd data
wget https://storage.googleapis.com/vakyaansh-open-models/ieee/data/hindi_iitm_challenge.zip
unzip hindi_iitm_challenge.zip
rm hindi_iitm_challenge.zip

```


## Step 1.2: Analyze audio properties

```
ls | head -10
soxi filename.wav
```

## Step 1.3: Analyze transcripts

```
ls | head -10
cat filename.txt
```

## Step 1.4: Check total files
```
ls | wc -l
ls *.wav | wc -l # to check wav files
ls *.txt | wc -l # to check txt files
```

## Step 1.5: Check total duration
```
python vakyansh-wav2vec2-experimentation/utils/audio/duration.py data/raw/train
```

## Step 1.6: Analyze and clean transcripts for foreign characters
```
cd vakyansh-wav2vec2-experimentation/utils/data
python clean_hindi.py -i ../../../data/raw/train/ -o ../../../data/processed/train/
```

## Step 1.7: Running the cleaning script
Discuss the data cleaning script

Tutorial 1: Parallel Processing
```
python vakyansh-wav2vec2-experimentation/notebooks/run_single.py
python vakyansh-wav2vec2-experimentation/notebooks/run_parallel.py
```


Tutorial 2: 
[Decomposition] (https://unicode.org/reports/tr15/)

Resources:
[Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library)

## Step 1.8 Preparing data for finetuning
Update paths for train_path, valid_path

```
bash vakyansh-wav2vec2-experimentation/scripts/finetuning/prepare_data.sh
```

## Step 1.9 Discuss Steps of Data Preparation:

a) Creation of manifest. Creates TSV files with number of frames in an audio

b) Creation of labels. Read the text from text files and create ltr (letter level separated by spaces)  and wrd (lines as read from txt files) files

c) Creation of dict (containing labels with counts) and lexicon (containing unique words with space separated letters) file
