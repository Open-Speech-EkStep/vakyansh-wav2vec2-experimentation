dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change, if any ###

finetuned_model_path=$parentdir'/checkpoints/finetuning/checkpoint_best.pt'
custom_model_path=$parentdir'/checkpoints/custom_model/final_model.pt'
dictionary=$parentdir'/data/finetuning/dict.ltr.txt'

### Values to change end ###
printf "Generating Custom Model"
python generate_custom_model.py -f ${finetuned_model_path} -d ${dictionary} -o $custom_model_path
printf "\nModel saved at :"${custom_model_path}"\n"
