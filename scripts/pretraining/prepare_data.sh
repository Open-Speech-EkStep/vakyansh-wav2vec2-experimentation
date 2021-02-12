dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###

valid_percentage=0.04
prep_scripts="../../utils/prep_scripts"
destination_path=$parentdir'/data/pretraining'
wav_path="wav_dir_path"

### Values to change end ###

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name train --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"