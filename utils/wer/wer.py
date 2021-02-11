## Usage: python wer.py --original-dir ./dataset/original/ --predicted-dir ./dataset/azure/ --wav-dir ./dataset/wav_16/ --save-output True --output-name test


import pandas as pd
import numpy as np
import glob
import Levenshtein as Lev
from tqdm import tqdm
import swifter
import argparse
import os

def wer( s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def calculate_wer_local(row):
    wer_local = ''
    try:
        wer_local = wer(row['cleaned_text_x'], row['cleaned_text_y'])
        #cer_local = cer(row['cleaned_text'], row['text_y'])
    except:
        print(row)
        return len(row['cleaned_text_x'].split(' '))
    return wer_local

def calculate_cer(row):
    try:
        cer_local = cer(row['cleaned_text_x'], row['cleaned_text_y'])
    except:
        return len(row['cleaned_text_x'].str.replace(' ','').str.len())
    return cer_local

def get_text(txt_filename):
    contents = ''
    if os.path.exists(txt_filename):
        with open(txt_filename, encoding='utf-8-sig') as file:
            contents = file.read()
    return contents

import re

def clean_files(original_text):
    pattern = '[^ ँ-ःअ-ऋए-ऑओ-नप-रलव-हा-ृे-ॉो-्0-9क़-य़ॅ]+'
    return (re.sub(pattern, '', original_text)).strip()

def create_dataframe(txt_dir, wav_dir, clean):
    files = glob.glob(txt_dir + '*.txt')
    df = pd.DataFrame(files)
    df.columns = ['txt_path']
    df['filename'] = df.txt_path.str.split('/').str[-1]
    df['wav_path'] = wav_dir + df['filename'].str[:-3] + 'wav'
    df['text'] = df.txt_path.swifter.apply(get_text)
    if clean:
        df['cleaned_text'] = df.text.swifter.apply(clean_files)
    else:
        df['cleaned_text'] = df.text
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--original-dir', type=str, required=True)
    parser.add_argument('--predicted-dir', type=str, required=True)
    parser.add_argument('--wav-dir', type=str, required=True)
    parser.add_argument('--save-output', type=bool)
    parser.add_argument('--output-name', type=str)
    parser.add_argument('--clean', type=bool)
    args = parser.parse_args()



    original_csv = create_dataframe(args.original_dir, args.wav_dir, args.clean)
    predicted_csv = create_dataframe(args.predicted_dir, args.wav_dir, args.clean)

    df_merged = pd.merge(original_csv, predicted_csv, on='wav_path')

    df_merged['wer'] = df_merged.swifter.apply(calculate_wer_local, axis = 1)
    df_merged['cer'] = df_merged.swifter.apply(calculate_cer, axis = 1)
    df_merged['num_tokens'] = df_merged['cleaned_text_x'].str.split().str.len()
    df_merged['num_chars'] = df_merged['cleaned_text_x'].str.replace(' ','').str.len()
    fwer = df_merged.wer.sum() / df_merged.num_tokens.sum()
    fcer = df_merged.cer.sum() / df_merged.num_chars.sum()
    print('WER: ', fwer*100)
    print('CER: ', fcer*100)

    if args.save_output:    
        df_merged.to_csv('wer_'+args.output_name+'.csv', index=False)
