dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"


### Values to change -start ###

train_wav_path="path to wav files"
valid_wav_path=""
make_valid_from_train=1 # 0 if validation data is in a different folder and 1 if validation data is to be extracted from train_wav_path
valid_percentage=0.1

### Values to change end ###

train_name="train"
valid_name="valid"
# txt_path=${train_wav_path}
utils="../../utils"
prep_scripts=$utils"/prep_scripts/"
analysis_scripts=$utils"/analysis/"
destination_path=$parentdir'/data/finetuning'

if [ "${make_valid_from_train}" = 1 ]; then

    python ${prep_scripts}/manifest.py ${train_wav_path} --dest ${destination_path} --ext wav --train-name ${train_name} --valid-percent ${valid_percentage} --jobs -1
    echo "Manifest Creation Done"
    echo "Valid data extracted from train set"

    python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/${train_name}.tsv --output-dir ${destination_path} --output-name ${train_name} --txt-dir ${train_wav_path}
    python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/${valid_name}.tsv --output-dir ${destination_path} --output-name ${valid_name} --txt-dir ${train_wav_path}
    echo "Word file generated"

else 
    valid_percentage=0.0
    python ${prep_scripts}/manifest.py ${train_wav_path} --dest ${destination_path} --ext wav --train-name ${train_name} --valid-percent ${valid_percentage} --jobs -1
    python ${prep_scripts}/manifest.py ${valid_wav_path} --dest ${destination_path} --ext wav --train-name ${valid_name} --valid-percent ${valid_percentage} --jobs -1
    echo "Manifest Creation Done"

    python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/${train_name}.tsv --output-dir ${destination_path} --output-name ${train_name} --txt-dir ${train_wav_path}
    python ${prep_scripts}/labels.py --jobs 64 --tsv ${destination_path}/${valid_name}.tsv --output-dir ${destination_path} --output-name ${valid_name} --txt-dir ${valid_wav_path}
    echo "Word file generated"

fi

python ${prep_scripts}/dict_and_lexicon_maker.py --wrd ${destination_path}/${train_name}.wrd --lexicon ${destination_path}/lexicon.lst --dict ${destination_path}/dict.ltr.txt
echo "Dict file generated from train data"

echo "Starting dictionary analysis from "${destination_path}/dict.ltr.txt" for any punctuation marks"
python ${analysis_scripts}/generate_dict_analysis.py --dict ${destination_path}/dict.ltr.txt

echo "Starting analysis of train tsv..."
python ${analysis_scripts}/generate_wav_report_from_tsv.py --tsv ${destination_path}/${train_name}.tsv

echo "Starting analysis of valid tsv..."
python ${analysis_scripts}/generate_wav_report_from_tsv.py --tsv ${destination_path}/${valid_name}.tsv
echo "Analysis done."
