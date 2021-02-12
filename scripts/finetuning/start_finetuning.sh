#!/bin/bash


### Values to change - start ###

config_name='finetuning.yaml'
gpus=1
run_in_nohup=0  #0 for no, 1 for yes

### Values to change - end ###

dir=$PWD/
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"
printf "** Directory to code is: $parentdir"

config_path=${parentdir}'/config/'${config_name}
data_path=${parentdir}'/data/finetuning'
checkpoints_path=${parentdir}'/checkpoints/finetuning'
log_path=${parentdir}'/logs/finetuning'
tensorboard_path=${log_path}'/tensorboard'
pretrained_model_path=${parentdir}'/checkpoints/pretraining/checkpoint_best.pt'
update_freq=$((24/${gpus}))
wav2vec_repo_path=${parentdir}'/../fairseq/'

printf "\n** $wav2vec_repo_path"
printf "\n** Config path is: $config_path"
printf "\n** Data path is: $data_path"
printf "\n** Checkpoint will be saved at: $checkpoints_path"
printf "\n** Logs will be saved at: ${log_path}"
printf "\n** Update frequency is: ${update_freq}"

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

# importing parse_yaml.sh file
. ${parentdir}'/scripts/'parse_yaml.sh
eval $(parse_yaml $config_path)

if [ "${run_in_nohup}" = 1 ]; then

    local_timestamp=$(timestamp)
    tensorboard_path=${tensorboard_path}_${local_timestamp}
    mkdir -p ${tensorboard_path}

    printf "\n** Tensorboard is running **"
    printf "\n** Tensorboard logs path: ${tensorboard_path}"
    printf "\n"

    nohup python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port -1 $data_path \
    --save-dir ${checkpoints_path} --fp16 --post-process letter \
    --valid-subset ${valid_subset} --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers ${num_workers} \
    --max-update ${max_update} --sentence-avg --task audio_pretraining --arch wav2vec_ctc \
    --w2v-path ${pretrained_model_path}   --labels ltr \
    --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 \
    --layerdrop 0.1 --mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 \
    --mask-channel-prob 0.5 --zero-infinity  --feature-grad-mult 0.0 --freeze-finetune-updates 500 \
    --validate-after-updates 1000 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
    --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 --decay-steps 40000 \
    --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
    --attention-dropout 0.0 --max-tokens ${max_tokens} --seed 2337  --log-format json --log-interval 1 \
    --ddp-backend no_c10d --update-freq ${update_freq} \
    --tensorboard-logdir ${tensorboard_path}  &> ${log_path}/${local_timestamp}.out &

    nohup tensorboard --logdir ${tensorboard_path} --bind_all &> /dev/null &


else
    python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port -1 $data_path \
    --save-dir ${checkpoints_path} --fp16 --post-process letter \
    --valid-subset ${valid_subset} --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers ${num_workers} \
    --max-update ${max_update} --sentence-avg --task audio_pretraining --arch wav2vec_ctc \
    --w2v-path ${pretrained_model_path} --labels ltr \
    --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 \
    --layerdrop 0.1 --mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 \
    --mask-channel-prob 0.5 --zero-infinity --feature-grad-mult 0.0 --freeze-finetune-updates 500 \
    --validate-after-updates 1000 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
    --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 --decay-steps 40000 \
    --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
    --attention-dropout 0.0 --max-tokens ${max_tokens} --seed 2337  --log-format json --log-interval 1 \
    --ddp-backend no_c10d --update-freq ${update_freq} 


fi
