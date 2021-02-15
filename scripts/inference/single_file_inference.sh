dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change, if any ###
custom_model_path=$parentdir'/checkpoints/custom_model/final_model.pt'
dictionary=$parentdir'/data/finetuning/dict.ltr.txt'
wav_file_path="/home/jupyter/dev_english/dev_IITM/kol_134_long_219_eng-010000-013808-2-1.wav"
decoder="viterbi"
cuda=True

### Values to change end ###


python ../../utils/inference/single_file_inference.py -m ${custom_model_path} -d ${dictionary} -w ${wav_file_path} -c ${cuda} -D ${decoder}
