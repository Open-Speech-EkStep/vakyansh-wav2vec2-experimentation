
import pandas as pd
import numpy as np
import glob
import Levenshtein as Lev
from tqdm import tqdm
import swifter
import argparse
from joblib import Parallel, delayed

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

def clean_text(row):
    return row[0][0:row.ind]

def preprocess(original_csv):
    original_csv['ind'] = original_csv['text'].str.index('(None')
    original_csv['cleaned_text'] = original_csv.swifter.apply(clean_text, axis = 1)
    return original_csv

def calculate_wer(row):
    wer_local = ''
    try:
        wer_local = wer(row['original'], row['predicted'])
        #cer_local = cer(row['cleaned_text'], row['text_y'])
    except:
        print(row)
        return len(row['original'].split(' '))
    return wer_local


def calculate_cer(row):
    try:
        cer_local = cer(row['original'], row['predicted'])
    except:
        return len(row['original'].str.replace(' ','').str.len())
    return cer_local


def run_pipeline(ground_truth):
    
    # with open(ground_truth, encoding='utf-8') as file:
    #     original_csv = file.readlines()
    
    # original_csv = [line.strip() for line in original_csv]
    # #print(original_csv)
    # # for line in original_csv:
    # #     line.

    # df_merged = pd.DataFrame(original_csv, columns=['text'])
    # df_merged = df_merged['text'].str.split('",', expand=True)
    # print(df_merged.head())

    df_merged = pd.read_csv(ground_truth, quotechar="\"")
    

    df_merged.columns = [ 'predicted', 'original']
    df_merged['predicted'].fillna(' ', inplace=True)
    df_merged['original'] = df_merged['original'].str.strip()
    print(df_merged.head())

    df_merged['wer'] = df_merged.apply(calculate_wer, axis = 1)
    df_merged['cer'] = df_merged.apply(calculate_cer, axis = 1)
    df_merged['num_tokens'] = df_merged['original'].str.split().str.len()
    df_merged['num_chars'] = df_merged['original'].str.replace(' ','').str.len()
    
    df_merged.sort_values(by = 'wer', ascending=False)
    fwer = df_merged.wer.sum() / df_merged.num_tokens.sum()
    fcer = df_merged.cer.sum() / df_merged.num_chars.sum()
    print('WER: ', fwer*100)
    print('CER: ', fcer*100)
    local_arr= ground_truth.split('/')[-1][:-4].split('_')
    lst.append([fwer, fcer, local_arr[1], local_arr[2]])
    print(lst)
    return [fwer*100, fcer*100, local_arr[1], local_arr[2]]
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CER pipeline')
    parser.add_argument('-o', '--original', required=True, help='Original File')
    
    args_local = parser.parse_args()
    #print(args_local)

    #run_pipeline(args_local.original)
    global lst
    lst = []
    csv_files = glob.glob(args_local.original+'/*.csv')

    lst.extend(Parallel(n_jobs=-1)(delayed(run_pipeline)(local_csv) for local_csv in csv_files))
    
    df = pd.DataFrame(lst, columns=['WER', 'CER', 'LM_WEIGHT','WORD_SCORE'])
    df.sort_values(by='WER', inplace=True)
    df.to_csv('final_report_nptel.csv', index=False)
    print(df)

    
