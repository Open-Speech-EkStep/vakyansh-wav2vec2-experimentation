dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###

valid_percentage=0.04
prep_scripts="/home/harveen.chadha/common_scripts/prep_scripts"
destination_path=$parentdir'/data/finetuning'
wav_path="/home/jupyter/IITM_NPTEL_clean_english_data/test"
txt_path=${wav_path}

### Values to change end ###

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name train --valid-percent ${valid_percentage} --jobs -1
echo "Manifest Creation Done"

python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/train.tsv --output-dir ${destination_path} --output-name train --txt-dir ${txt_path}
python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/valid.tsv --output-dir ${destination_path} --output-name valid --txt-dir ${txt_path}

echo "Word file generated"

python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${destination_path}/train.wrd --lexicon ${destination_path}/lexicon.lst --dict ${destination_path}/dict.ltr.txt
echo "Dict file generated"
