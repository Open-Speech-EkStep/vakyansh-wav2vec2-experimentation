
dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

hf_output_dir=${parentdir}/utils/hf/convert_to_hf.py
checkpoint_path=${parent_dir}/checkpoints/hf
finetuning_path=${parent_dir}/checkpoints/finetuning/checkpoint_best.pt
dict_path=${parent_dir}/data/finetuning/dict.ltr.txt
config_path=${parent_dir}/config/hf/hf_config.json
is_finetuned=1

if [[ $is_finetuned == 1 ]]; then
	python ../../utils/hf/convert_to_hf.py -o ${checkpoint_path} -m ${finetuning_path} -d ${dict_path} -c ${config_path}
else
	python ../../utils/hf/convert_to_hf.py -o ${checkpoint_path} -m ${finetuning_path} -d ${dict_path} -c ${config_path} -f True
fi
