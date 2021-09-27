dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change, if any ###
custom_model_path=$parentdir'/checkpoints/hf/'
wav_file_path="path to single wav file" # path to single wav file

python $parentdir/utils/hf/single_file_inference.py -m ${custom_model_path}  -w ${wav_file_path}
