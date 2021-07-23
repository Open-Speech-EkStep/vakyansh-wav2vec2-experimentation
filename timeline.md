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
soxi filename.wav
```

## Step 1.3: Analyze transcripts

```
cat filename.txt
```

## Step 1.4 Check total files
```
ls | wc -l
ls *.wav | wc -l # to check wav files
ls *.txt | wc -l # to check txt files
```

## Step 1.5 Check total duration
```
python vakyansh-wav2vec2-experimentation/utils/audio/duration.py data/raw/train
```


