### Generate n-gram based LM using KenLM

**top_k:** KenLM will use only top_k frequency words to make its vocab.
**input_txt_path:** Path to your clean input text file.
**output_path:** Folder path to store binary file.
**kenlm_bins:** Path to bin folder inside kenlm/build.

**run_concatenate_text:** Make it 1 if text data is present in multiple files, this will concatenate data from all text files.
**run_generate_lm_vocab:** Make it 1 to generate vocab files which will be used for making lexicon file.
**run_make_lexicon_lst:** Make it 1 to generate lexicon file which will be used while decoding language model.
