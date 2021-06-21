
dir=$PWD/
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###

w2l_decoder_viterbi=1 # 1 for viterbi, 0 for kenlm
inference_data_name='hindi'
beam=128 # 128 or 1024
subset='test'

# FOR LM MODEL
lm_name='hindi'
lm_model_path=${parentdir}'/lm/'${lm_name}'/lm.binary'
lexicon_lst_path=${parentdir}'/lm/'${lm_name}'/lexicon.lst'

# WARNING ONLY FOR OLD MODELS

#pretrained_model_path=''
pretrained_model_path=${parentdir}'/checkpoints/pretraining/checkpoint_best.pt'

# WARNING END


### Values to change end ###

wav2vec_repo_path='/opt/wav2vec/fairseq/'
checkpoint_path=${parentdir}'/checkpoints/finetuning/checkpoint_best.pt'
result_path=${parentdir}'/results/'${inference_data_name}
data_path=${parentdir}'/data/inference/'${inference_data_name}

if [[ ${pretrained_model_path} != '' ]]; then
	echo "Updating Pretrained Model path in model config"
	python ../../utils/inference/update_model.py -f ${checkpoint_path} -p ${pretrained_model_path}
fi

if [ "${w2l_decoder_viterbi}" = 1 ]; then
  mkdir -p ${result_path}
  python ../../utils/inference/infer.py ${data_path} --task audio_pretraining \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${result_path} --w2l-decoder viterbi \
  --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
  --post-process letter

  #python ../../utils/wer/wer_wav2vec.py -o ${result_path}/ref.word-checkpoint_best.pt-test.txt -p ${result_path}/hypo.word-checkpoint_best.pt-test.txt \
  #-t ${data_path}/${subset}.tsv -s save -n ${result_path}/sentence_wise_wer.csv

else

  python ../../utils/inference/infer.py ${data_path} --task audio_pretraining \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${result_path}_kenlm_${beam} --w2l-decoder kenlm --lm-model ${lm_model_path}\
  --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 --lexicon ${lexicon_lst_path} \
  --post-process letter --beam ${beam}

  #python ../../utils/wer/wer_wav2vec.py -o ${result_path}_kenlm_${beam}/ref.word-checkpoint_best.pt-test.txt -p ${result_path}_kenlm_${beam}/hypo.word-checkpoint_best.pt-test.txt \
  #-t ${data_path}/${subset}.tsv -s save -n ${result_path}_kenlm_${beam}/sentence_wise_wer.csv
fi
