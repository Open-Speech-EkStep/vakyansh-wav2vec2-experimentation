## Generate n-gram based LM using KenLM

**lm_name:** Name of the folder containing the lm files(eg: lm.binary etc)

To combine text files to one file: 
**txt_files_dir:** Path to directory with multiple input text files for generaing one combined text file

**combined_txt_file_save_path:** Path to the output combined text file created from multiple input text files

For kenlm : 
**top_k:** KenLM will use only top_k frequency words to make its vocab.

**input_txt_path:** Path to your clean input text file(default finetuning/train.wrd). If the input file is generated from multiple text file then set this variable equal to 'combined_txt_file_save_path'

**output_path:** Folder path to store binary file.

**kenlm_bins:** Path to bin folder inside kenlm/build.

**run_concatenate_text:** Make it 1 if text data is present in multiple files, this will concatenate data from all text files.

**run_generate_lm_vocab:** Make it 1 to generate vocab files which will be used for making lexicon file.

**run_make_lexicon_lst:** Make it 1 to generate lexicon file which will be used while decoding language model.
