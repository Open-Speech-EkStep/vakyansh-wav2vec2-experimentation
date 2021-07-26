#This code has been taken from https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py

import argparse
import gzip
import io
import os
import subprocess
import itertools
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed


def convert_and_filter_topk(args):
    """ Convert to lowercase, count word occurrences and save top-k words to a file """

    
    data_lower = os.path.join(args.output_dir, "lower.txt.gz")

    with open(args.input_txt, encoding="utf-8") as file_in:
        lines = file_in.readlines()


    print("\nCalculating number of words")
    
    words = list(itertools.chain(*[line.split() for line in tqdm(lines)]))
    
#     words = []
#     words_list = [line_lower.split() for line_lower in tqdm(lines)]
#     for item in words_list:
#         for local_item in item:
#             words.append(local_item)

    print("\nCounting Word Frequencies ")
    counter = Counter(words)


    print("\nWriting File")
    with open(data_lower, mode='w+', encoding="utf-8") as file_out:
        file_out.writelines(lines)


    # Save top-k words
    print("\nSaving top {} words ...".format(args.top_k))
    top_counter = counter.most_common(args.top_k)
    vocab_str = "\n".join(word for word, count in top_counter)
    vocab_path = "vocab-{}.txt".format(args.top_k)
    vocab_path = os.path.join(args.output_dir, vocab_path)
    with open(vocab_path, "w+") as file:
        file.write(vocab_str)

    print("\nCalculating word statistics ...")
    total_words = sum(counter.values())
    print("  Your text file has {} words in total".format(total_words))
    print("  It has {} unique words".format(len(counter)))
    top_words_sum = sum(count for word, count in top_counter)
    word_fraction = (top_words_sum / total_words) * 100
    print(
        "  Your top-{} words are {:.4f} percent of all words".format(
            args.top_k, word_fraction
        )
    )
    print('  Your most common word "{}" occurred {} times'.format(*top_counter[0]))
    last_word, last_count = top_counter[-1]
    print(
        '  The least common word in your top-k is "{}" with {} times'.format(
            last_word, last_count
        )
    )
    for i, (w, c) in enumerate(reversed(top_counter)):
        if c > last_count:
            print(
                '  The first word with {} occurrences is "{}" at place {}'.format(
                    c, w, len(top_counter) - 1 - i
                )
            )
            break

    return data_lower, vocab_str


def build_lm(args, data_lower, vocab_str):
    print("\nCreating ARPA file ...")
    lm_path = os.path.join(args.output_dir, "lm.arpa")
    subargs = [
            os.path.join(args.kenlm_bins, "lmplz"),
            "--order",
            str(args.arpa_order),
            "--temp_prefix",
            args.output_dir,
            "--memory",
            args.max_arpa_memory,
            "--text",
            data_lower,
            "--arpa",
            lm_path,
            "--prune",
            *args.arpa_prune.split("|"),
        ]
    if args.discount_fallback:
        subargs += ["--discount_fallback"]
    subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = os.path.join(args.output_dir, "lm_filtered.arpa")
    subprocess.run(
        [
            os.path.join(args.kenlm_bins, "filter"),
            "single",
            "model:{}".format(lm_path),
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )

    # Quantize and produce trie binary.
    print("\nBuilding lm.binary ...")
    binary_path = os.path.join(args.output_dir, "lm.binary")
    subprocess.check_call(
        [
            os.path.join(args.kenlm_bins, "build_binary"),
            "-a",
            str(args.binary_a_bits),
            "-q",
            str(args.binary_q_bits),
            "-v",
            args.binary_type,
            filtered_path,
            binary_path,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate lm.binary and top-k vocab for DeepSpeech."
    )
    parser.add_argument(
        "--input_txt",
        help="Path to a file.txt or file.txt.gz with sample sentences",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir", help="Directory path for the output", type=str, required=True
    )
    parser.add_argument(
        "--top_k",
        help="Use top_k most frequent words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--kenlm_bins",
        help="File path to the KENLM binaries lmplz, filter and build_binary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_arpa_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--binary_a_bits",
        help="Build binary quantization value a in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_q_bits",
        help="Build binary quantization value q in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_type",
        help="Build binary data structure type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--discount_fallback",
        help="To try when such message is returned by kenlm: 'Could not calculate Kneser-Ney discounts [...] rerun with --discount_fallback'",
        action="store_true",
    )

    args = parser.parse_args()

    data_lower, vocab_str = convert_and_filter_topk(args)
    build_lm(args, data_lower, vocab_str)

    # Delete intermediate files
    os.remove(os.path.join(args.output_dir, "lower.txt.gz"))
    os.remove(os.path.join(args.output_dir, "lm.arpa"))
    os.remove(os.path.join(args.output_dir, "lm_filtered.arpa"))


if __name__ == "__main__":
    main()
