dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change, if any ###
custom_model_path=$parentdir'/checkpoints/custom_model_english/final_model.pt'
dictionary=$parentdir'/checkpoints/custom_model_english/dict.ltr.txt'
wav_file_path="path to single wav file" # path to single wav file
decoder="viterbi" # viterbi or kenlm
cuda=False
half=True

#If kenlm
lm_name='english'
lm_model_path=${parentdir}'/lm/'${lm_name}'/lm.binary'
lexicon_lst_path=${parentdir}'/lm/'${lm_name}'/lexicon.lst'

### Values to change end ###

if [ "$decoder" = "viterbi" ]
then
	python ../../utils/inference/single_file_inference.py -m ${custom_model_path} -d ${dictionary} -w ${wav_file_path} -c ${cuda} -D ${decoder} -H ${half}
else
	python ../../utils/inference/single_file_inference.py -m ${custom_model_path} -d ${dictionary} -w ${wav_file_path} -c ${cuda} -D ${decoder} -H ${half} -l ${lexicon_lst_path} -L ${lm_model_path}
fi
