dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###

valid_percentage=0.04
prep_scripts="/home/harveen.chadha/common_scripts/prep_scripts"
destination_path=$parentdir'/data/pretraining'
wav_path="/home/jupyter/IITM_NPTEL_clean_english_data/test"

### Values to change end ###

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name train --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"