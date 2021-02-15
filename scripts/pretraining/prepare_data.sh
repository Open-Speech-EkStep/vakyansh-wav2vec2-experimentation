dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###

valid_percentage=0.04
utils="../../utils"
prep_scripts=$utils"/prep_scripts/"
analysis_scripts=$utils"/analysis/"
destination_path=$parentdir"/data/pretraining"
wav_path="path_to_data"
train_name="train"
### Values to change end ###

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name ${train_name} --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"
echo "Starting analysis..."
python ${analysis_scripts}/generate_wav_report_from_tsv.py --tsv ${destination_path}/${train_name}.tsv
echo "Analysis done."
