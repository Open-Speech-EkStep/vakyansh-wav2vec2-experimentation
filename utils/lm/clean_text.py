import re
import argparse

def clean_text(args):
    '''
        Text cleaning for english, replaces punctuation from sentences and then removes sentences containing foreign language characters.
    '''
    with open(args.inp_text_file, mode='r', encoding='UTF-8') as inp_file, open(args.out_text_file, mode='w+', encoding='UTF-8') as out_file:
        lines = [line.strip() for line in inp_file.readlines()]
        for line in lines:
            line = re.sub(r'[^\w\s]','',line, re.UNICODE) #replace punctuation words with ''
            if not re.search('[^ A-Za-z]+',line): # true for sentences contains only english alphabets
                print(line.upper(), file=out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-text-file', type=str)
    parser.add_argument('--out-text-file', type=str)
    args = parser.parse_args()
    clean_text(args)