
dir=$PWD/
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

### Values to change -start ###

w2l_decoder_viterbi=1 # 1 for viterbi, 0 for kenlm
inference_data_name='toy_english'
beam=128 # 128 or 1024
subset='test'

# FOR LM MODEL
lm_name=''
lm_model_path=${parentdir}'/lm/'${lm_name}'/lm.binary'
lexicon_lst_path=${parentdir}'/lm/'${lm_name}'/lexicon.lst'

# SAVE PREDICTED TEXT FILES
dest_folder='foldername'
save_predicted=0 # 1 to save, 0 default

# FOR pretrained model
pretrained_model_path='../../checkpoints/pretraining/CLSRIL-23.pt'


# WARNING ONLY FOR OLD MODELS
old_model=0
# WARNING END


### Values to change end ###

checkpoint_path=${parentdir}'/checkpoints/finetuning/checkpoint_best.pt'
result_path=${parentdir}'/results/'${inference_data_name}
data_path=${parentdir}'/data/inference/'${inference_data_name}

if [[ ${old_model} = 1 ]]; then
	echo "Updating Pretrained Model path in model config"
	python ../../utils/inference/update_model.py -f ${checkpoint_path} -p ${pretrained_model_path}
fi

if [ "${w2l_decoder_viterbi}" = 1 ]; then
  mkdir -p ${result_path}
  python ../../utils/inference/infer.py ${data_path} --task audio_pretraining \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${result_path} --w2l-decoder viterbi \
  --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 6000000 \
  --post-process letter --model-overrides "{'w2v_path':'${pretrained_model_path}'}"

  python ../../utils/wer/wer_wav2vec.py -o ${result_path}/ref.word-checkpoint_best.pt-test.txt -p ${result_path}/hypo.word-checkpoint_best.pt-test.txt \
  -t ${data_path}/${subset}.tsv -s save -n ${result_path}/sentence_wise_wer.csv -e true

  if [ "${save_predicted}" = 1 ]; then
    python ../../utils/inference/save_predicted_output.py -f ${result_path}/sentence_wise_wer.csv -d ${dest_folder}
  fi

else
  kenlm_result_path=${result_path}_${lm_name}_${beam}
  mkdir -p ${kenlm_result_path}
  python ../../utils/inference/infer.py ${data_path} --task audio_pretraining \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${kenlm_result_path} --w2l-decoder kenlm --lm-model ${lm_model_path}\
  --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 6000000 --lexicon ${lexicon_lst_path} \
  --post-process letter --beam ${beam} --model-overrides "{'w2v_path':'${pretrained_model_path}'}"


  python ../../utils/wer/wer_wav2vec.py -o ${kenlm_result_path}/ref.word-checkpoint_best.pt-test.txt -p ${kenlm_result_path}/hypo.word-checkpoint_best.pt-test.txt \
  -t ${data_path}/${subset}.tsv -s save -n ${kenlm_result_path}/sentence_wise_wer.csv -e true

  if [ "${save_predicted}" = 1 ]; then
    python ../../utils/inference/save_predicted_output.py -f ${kenlm_result_path}/sentence_wise_wer.csv -d ${dest_folder}
  fi

fi


