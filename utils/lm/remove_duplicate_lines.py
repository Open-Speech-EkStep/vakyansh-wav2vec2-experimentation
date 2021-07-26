## Usage: python remove_duplicate_lines.py --inp-text-file /path/to/duplicate/lines/file --out-text-file /path/to/unique/lines/file

import argparse
from tqdm import tqdm

def remove_duplicate(args):
    '''
        Removes duplicate lines from text file
    '''
    with open(args.inp_text_file, mode='r', encoding='UTF-8') as inp_file, open(args.out_text_file, mode='w+', encoding='UTF-8') as out_file:
        lines = [line.strip() for line in inp_file.readlines()]
        lines_seen = set()
        for line in tqdm(lines):
            if line not in lines_seen:
                print(line, file=out_file)
                lines_seen.add(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-text-file', type=str)
    parser.add_argument('--out-text-file', type=str)
    args = parser.parse_args()
    remove_duplicate(args)
