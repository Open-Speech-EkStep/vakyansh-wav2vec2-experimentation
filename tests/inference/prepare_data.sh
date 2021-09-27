dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###
inference_data_name="hindi"
wav_path="path to wav files"
prep_scripts="../../utils/prep_scripts"
destination_path=$parentdir'/data/inference/'${inference_data_name}
### Values to change end ###

finetuning_dict=$parentdir'/data/finetuning/dict.ltr.txt'
txt_path=${wav_path}
analysis_scripts="../../utils/analysis"

mkdir ${destination_path}

python ${prep_scripts}/manifest.py ${wav_path} --dest ${destination_path} --ext wav --train-name test --valid-percent 0 --jobs -1
echo "Manifest Creation Done"

python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/test.tsv --output-dir ${destination_path} --output-name test --txt-dir ${txt_path}
echo "Word file generated"

python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${destination_path}/test.wrd --lexicon ${destination_path}/lexicon.lst --dict ${destination_path}/dict.ltr.txt
echo "Dict file generated"

echo "Starting dictionary analysis of inference dictionary from "${destination_path}/dict.ltr.txt" for any punctuation marks"
python ${analysis_scripts}/generate_dict_analysis.py --dict ${destination_path}/dict.ltr.txt
# echo "Renaming dict.ltr.txt made from ", ${txt_path}, "to: test_dict.ltr.txt"
# mv dict.ltr.txt test_dict.ltr.txt

echo "Copying ",${finetuning_dict}," used for training to ", ${destination_path}
cp ${finetuning_dict} ${destination_path}

echo "Starting dictionary analysis of finetuning dictionary "${destination_path}/dict.ltr.txt" for any punctuation marks"
python ${analysis_scripts}/generate_dict_analysis.py --dict ${destination_path}/dict.ltr.txt



