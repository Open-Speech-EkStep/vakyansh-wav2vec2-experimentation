import pandas as pd
import numpy as np
import argparse
import string
def generate_report(dict_file):
    with open(dict_file) as inp_file:
        dict_lines = inp_file.readlines()
    chars = [line.split()[0] for line in dict_lines]
    count = [line.split()[-1] for line in dict_lines]
    extra_punc_idx = []
    punctuations = list(string.punctuation)
    punctuations.remove('|')
    for idx in range(len(chars)):
        if chars[idx] in punctuations:
            extra_punc_idx.append(idx)
    if extra_punc_idx:
        print("\n\n********************** WARNING **********************")
        print("Punctuations present in dictionary")
        [print(dict_lines[idx]) for idx in extra_punc_idx]
        print("*****************************************************")
    else:
        print("None of the characters from ","".join(punctuations)," are present in the dictionary: ", dict_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--dict', type=str, help="dict.ltr.txt file to analyse")
    args = parser.parse_args()
    dict_file = args.dict
    generate_report(dict_file)