
dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

hf_output_dir=${parentdir}/utils/hf/convert_to_hf.py
checkpoint_path=${parentdir}/checkpoints/hf
finetuning_path=${parentdir}/checkpoints/finetuning/checkpoint_best.pt
dict_path=${parentdir}/data/finetuning/dict.ltr.txt
config_path=${parentdir}/config/hf/hf_config.json
is_finetuned=1

mkdir $checkpoint_path

if [[ $is_finetuned == 1 ]]; then
	python ../../utils/hf/convert_to_hf.py -o ${checkpoint_path} -m ${finetuning_path} -d ${dict_path} -c ${config_path}
else
	python ../../utils/hf/convert_to_hf.py -o ${checkpoint_path} -m ${finetuning_path} -d ${dict_path} -c ${config_path} -f True
fi
