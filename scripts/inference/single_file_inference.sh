dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change, if any ###
custom_model_path=$parentdir'/checkpoints/custom_model/final_model.pt'
dictionary=$parentdir'/data/finetuning/dict.ltr.txt'
wav_file_path="" # path to single wav file
decoder="viterbi" # viterbi or kenlm
cuda=True

#If kenlm
lexicon_lst_path=${parentdir}'/lm/lexicon.lst'
lm_model_path=${parentdir}'/lm/lm.binary'

### Values to change end ###

if [ "$decoder" = "viterbi" ]
then
	python ../../utils/inference/single_file_inference.py -m ${custom_model_path} -d ${dictionary} -w ${wav_file_path} -c ${cuda} -D ${decoder}
else
	python ../../utils/inference/single_file_inference.py -m ${custom_model_path} -d ${dictionary} -w ${wav_file_path} -c ${cuda} -D ${decoder} -l ${lexicon_lst_path} -L ${lm_model_path}
fi
