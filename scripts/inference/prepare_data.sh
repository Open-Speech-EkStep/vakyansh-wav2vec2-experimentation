
dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"
### Values to change -start ###

prep_scripts="/home/harveen.chadha/common_scripts/prep_scripts"
destination_path=$parentdir'/data/inference'
wav_path="/home/jupyter/IITM_NPTEL_clean_english_data/test"
txt_path=${wav_path}
### Values to change end ###

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name test --valid-percent 0 --jobs -1
echo "Manifest Creation Done"

python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/test.tsv --output-dir ${destination_path} --output-name test --txt-dir ${txt_path}

echo "Word file generated"

python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${destination_path}/test.wrd --lexicon ${destination_path}/lexicon.lst --dict ${destination_path}/dict.ltr.txt
echo "Dict file generated"

