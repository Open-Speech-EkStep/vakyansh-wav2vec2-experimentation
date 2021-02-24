
dir=$PWD/
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###
w2l_decoder_viterbi=1 # 1 for viterbi, 0 for kenlm
data_path=${parentdir}'/data/inference/'
validation_dataset=''
subset='test'
beam=128 # 128 or 1024

# FOR LM MODEL
lm_model_path=${parentdir}'/lm/lm.binary'
lexicon_lst_path=${parentdir}'/lm/lexicon.lst'
### Values to change end ###

wav2vec_repo_path=${parentdir}'/../fairseq/'
checkpoint_path=${parentdir}'/checkpoints/finetuning/checkpoint_best.pt'
result_path=${parentdir}'/results'

if [ "${w2l_decoder_viterbi}" = 1 ]; then
  python ${wav2vec_repo_path}examples/speech_recognition/infer.py ${data_path} --task audio_pretraining \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${result_path}/${validation_dataset} --w2l-decoder viterbi \
  --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
  --post-process letter


else
  python ${wav2vec_repo_path}examples/speech_recognition/infer.py ${data_path} --task audio_pretraining \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${result_path}/${validation_dataset}_kenlm_${beam} --w2l-decoder kenlm --lm-model ${lm_model_path}\
  --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 --lexicon ${lexicon_lst_path} \
  --post-process letter --beam ${beam}

fi
