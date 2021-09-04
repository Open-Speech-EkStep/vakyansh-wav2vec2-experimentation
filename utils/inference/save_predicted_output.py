import pandas as pd
import os
from tqdm import tqdm
import argparse

def save_text_file(path, text):
    with open(path, 'w+', encoding='utf-8') as out:
        out.write(text)

def save_predicted_output(out_csv, dest):
    df = pd.read_csv(out_csv)
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)

    for ix, row in tqdm(df.iterrows()):
        text = row['predicted']
        fpath = os.path.join(dest, row['path'].strip('/')).replace('.wav','.txt')
        folder = '/'.join(fpath.split('/')[:-1])
    
        os.makedirs(folder, exist_ok=True)
        save_text_file(fpath, text)
    print(f"predicted files created at {dest}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('-f', '--csv-path', type=str, help="Sentence wer csv path")
    parser.add_argument('-d', '--dest-path', type=str, help="Path to save predicted output as text files")

    args_local = parser.parse_args()

    save_predicted_output(args_local.csv_path, args_local.dest_path)

