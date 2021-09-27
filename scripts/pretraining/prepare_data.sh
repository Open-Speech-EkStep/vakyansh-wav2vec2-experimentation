dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###
wav_path="path to wav files"
valid_percentage=0.1
### Values to change -end ###

utils="../../utils"
prep_scripts=$utils"/prep_scripts/"
analysis_scripts=$utils"/analysis/"
destination_path=$parentdir"/data/pretraining"
train_name="train"
valid_name="valid"

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name ${train_name} --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"

echo "Starting analysis of train tsv..."
python ${analysis_scripts}/generate_wav_report_from_tsv.py --tsv ${destination_path}/${train_name}.tsv
echo "Starting analysis of valid tsv..."
python ${analysis_scripts}/generate_wav_report_from_tsv.py --tsv ${destination_path}/${valid_name}.tsv
echo "Analysis done."
