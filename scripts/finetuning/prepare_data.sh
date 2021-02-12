dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###

utils="../../utils"
prep_scripts=$utils"/prep_scripts/"
analysis_scripts=$utils"/analysis/"
destination_path=$parentdir'/data/finetuning'
wav_path="/home/jupyter/dev_english/dev_IITM/"
txt_path=${wav_path}
train_name="train"
valid_name="valid"
valid_percentage=0.04
### Values to change end ###

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name ${train_name} --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"

python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/${train_name}.tsv --output-dir ${destination_path} --output-name ${train_name} --txt-dir ${txt_path}
python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/${valid_name}.tsv --output-dir ${destination_path} --output-name ${valid_name} --txt-dir ${txt_path}

echo "Word file generated"

python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${destination_path}/${train_name}.wrd --lexicon ${destination_path}/lexicon.lst --dict ${destination_path}/dict.ltr.txt
echo "Dict file generated from train data"

echo "Starting analysis of train tsv..."
python ${analysis_scripts}/generate_wav_report_from_tsv.py --tsv ${destination_path}/${train_name}.tsv
echo "Starting analysis of valid tsv..."
python ${analysis_scripts}/generate_wav_report_from_tsv.py --tsv ${destination_path}/${valid_name}.tsv
echo "Analysis done."
