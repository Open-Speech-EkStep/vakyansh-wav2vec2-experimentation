## Usage: python clean_text.py --input /path/to/input/text/file --output /path/to/clean/text/file --dict /path/to/dict.ltr.txt
import re
import string
import argparse
from tqdm import tqdm


def get_clean_lines(line, pattern):
    '''
    Returns line if no foreign character other than pattern is present else returns empty string
    '''
    line = re.sub('[%s]' % re.escape(string.punctuation), '', line)
    if not re.search(pattern, line):
        return ' '.join([word.upper() for word in line.split() if word])
    else:
        return ''


def clean_text(args):
    '''
        Text cleaning to replace punctuations from sentences and then remove sentences containing foreign language characters.
    '''
    with open(args.input, mode='r', encoding='UTF-8') as inp_file, open(args.output, mode='w+', encoding='UTF-8') as out_file:
        lines = [line.strip() for line in inp_file.readlines()]
        dict_pattern = get_regex_from_dict(args)
        pattern = '[^ '+dict_pattern+']+'

        for line in tqdm(lines):
            clean_line = get_clean_lines(line, pattern)
            if clean_line:
                print(clean_line, file=out_file)

def get_regex_from_dict(args):
    '''
    Returns a string of characters from dict.ltr.txt
    '''
    dict_path = args.dict
    with open(dict_path, encoding='UTF-8') as f:
        dict_lines = f.readlines()
    chars = ''.join([line.split()[0].strip() for line in dict_lines])
    return chars

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--output', '-o', type=str)
    parser.add_argument('--dict', '-d', default='../../data/finetuning/dict.ltr.txt', type=str)
    args = parser.parse_args()
    clean_text(args)
