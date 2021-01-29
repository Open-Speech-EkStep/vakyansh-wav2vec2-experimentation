from glob import glob
from tqdm import tqdm
import argparse


def concatenate_text(txt_files_dir, combined_txt_file_save_path):
    txt_list = []
    all_txt = glob(txt_files_dir + '**/*.txt', recursive=True)
    print("Text files found:", len(all_txt))
    for file in tqdm(all_txt):
        with open(file) as f:
            txt_list.append(f.read())

    txt_str = ''.join(txt_list)
    with open(combined_txt_file_save_path, 'w+') as f:
        f.write(txt_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--txt-files-dir', type=str)
    parser.add_argument('--final-txt-file-save-path', type=str)
    args_local = parser.parse_args()

    txt_files_dir = args_local.txt_files_dir
    combined_txt_file_save_path = args_local.final_txt_file_save_path

    concatenate_text(txt_files_dir, combined_txt_file_save_path)
