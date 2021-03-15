dir=$PWD/
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change,if any - start ###
lm_name=""
#for text file generation
txt_files_dir="" #path to text file dir
combined_txt_file_save_path=${parentdir}"/lm/"${lm_name}"/all_text.txt"

#For kenlm
top_k=500000
input_txt_file_path=${parentdir}"/data/finetuning/train.wrd"
output_path=${parentdir}"/lm/"${lm_name}
kenlm_bins=${parentdir}"/../kenlm/build/bin"

#For lexicon 
vocab_txt_file=${output_path}"/vocab-"${top_k}".txt"
path_to_save_lexicon=${output_path}"/lexicon.lst"

# flags
run_concatenate_text=0 # 0 for skipping, 1 for runnning
run_generate_lm_vocab=1 # 0 for skipping, 1 for running
run_make_lexicon_lst=1 # 0 for skipping, 1 for running

### Values to change - end ###
mkdir ${output_path} 

if [ "$run_concatenate_text" == 1 ]; then
	printf "** Generating Combined Text file **\n"
	python ../../utils/lm/concatenate_text.py --txt-files-dir ${txt_files_dir} --final-txt-file-save-path ${combined_txt_file_save_path}
	printf "*:File generated at: "${combined_txt_file_save_path}
fi

if [ "$run_generate_lm_vocab" == 1 ]; then
	printf "\n** Generating kenlm **\n"
	python ../../utils/lm/generate_lm.py --input_txt ${input_txt_file_path} --output_dir ${output_path} \
		--top_k ${top_k} --kenlm_bins ${kenlm_bins} \
		--arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
		--binary_a_bits 255 --binary_q_bits 8 --binary_type trie 
	printf "**Kenlm Generated at : "${output_path}
fi

if [ "$run_make_lexicon_lst" == 1 ]; then
	printf "\n** Generating Lexicon from Combined Text File **\n"
	python ../../utils/lm/make_lexicon_lst.py --txt-file ${vocab_txt_file} --path-to-save-lexicon ${path_to_save_lexicon}
	printf "*:File generated at: "${path_to_save_lexicon}
fi
