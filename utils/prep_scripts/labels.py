
 # Usage: python labels.py --jobs 64 --tsv <path to train.tsv>train.tsv --output-dir <destination dir> --output-name test --txt-dir

import argparse
import os
import re
from tqdm import tqdm
from joblib import Parallel, delayed


def get_text(line,root):
    txt_path = line.split("\t")[0].replace(".wav",".txt").strip() ## implies that the text filename and wav filename should be same

    txt_path = os.path.join( root , txt_path )

    text = ''
    with open(txt_path , mode = "r", encoding="utf-8") as file_local:
        text = file_local.readline().strip()

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type = str, help = "TSV file for which labels need to be generated")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--txt-dir")
    parser.add_argument("--jobs", default=-1, type=int, help="Number of jobs to run in parallel")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tsv_file=args.tsv
    output_dir=args.output_dir
    output_name=args.output_name

    with open(tsv_file) as tsv, open(
            os.path.join(output_dir, output_name + ".ltr"), "w",encoding="utf-8"
        ) as ltr_out, open(
            os.path.join(output_dir, output_name + ".wrd"), "w",encoding="utf-8"
        ) as wrd_out:

        root = next(tsv).strip()

        if not args.txt_dir:
            args.txt_dir = root
        
        local_arr = []

        local_arr.extend(Parallel(n_jobs = args.jobs)( delayed(get_text)(line , args.txt_dir) for line in tqdm(tsv)))
    
        
        formatted_text = []
        for text in local_arr:
            local_list = list( text.replace(" ", "|") )
            final_text = " ".join(local_list) + ' |'
            formatted_text.append(final_text)


        wrd_out.writelines("\n".join(local_arr))
        ltr_out.writelines("\n".join(formatted_text))



if __name__ == "__main__":
    main()
