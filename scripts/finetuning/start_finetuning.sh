#!/bin/bash


### Values to change - start ###

config_name='finetuning.yaml'
gpus=1
run_in_nohup=1  #0 for no, 1 for yes
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
petrained_model_path=${parentdir}'/checkpoints/pretraining/checkpoint_best.pt'
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

    nohup python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port ${distributed_port} $data_path \
    --save-dir ${checkpoints_path} --fp16  --post-process ${post_process}\
    --valid-subset ${valid_subset} --no-epoch-checkpoints --best-checkpoint-metric ${best_checkpoint_metric} --num-workers ${num_workers} \
    --max-update ${max_update} --sentence-avg --task ${task} --arch ${arch} \
    --w2v-path ${petrained_model_path}   --labels ${labels} \
    --apply-mask --mask-selection ${mask_selection} --mask-other ${mask_other} --mask-length ${mask_length} --mask-prob ${mask_prob} \
    --layerdrop ${layerdrop}  --mask-channel-selection ${mask_channel_selection} --mask-channel-other ${mask_channel_other} --mask-channel-length ${mask_channel_length} \
    --mask-channel-prob ${mask_channel_prob} --zero-infinity  --feature-grad-mult ${feature_grad_mult} --freeze-finetune-updates ${freeze_finetune_updates} \
    --validate-after-updates ${validate_after_updates} --optimizer ${optimizer} --adam-betas ${adam_betas} --adam-eps ${adam_eps} \
    --lr ${lr} --lr-scheduler ${lr_scheduler} --warmup-steps ${warmup_steps} --hold-steps ${hold_steps} --decay-steps ${decay_steps} \
    --final-lr-scale ${final_lr_scale} --final-dropout ${final_dropout} --dropout ${dropout} --activation-dropout ${activation_dropout} --criterion ${criterion} \
    --attention-dropout ${attention_dropout} --max-tokens ${max_tokens} --seed ${seed}  --log-format ${log_format} --log-interval ${log_interval} \
    --ddp-backend ${ddp_backend} --update-freq ${update_freq} \
    --tensorboard-logdir ${tensorboard_path}  &> ${log_path}/${local_timestamp}.out &


else
    python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port ${distributed_port} $data_path \
    --save-dir ${checkpoints_path} --fp16  --post-process ${post_process}\
    --valid-subset ${valid_subset} --no-epoch-checkpoints --best-checkpoint-metric ${best_checkpoint_metric} --num-workers ${num_workers} \
    --max-update ${max_update} --sentence-avg --task ${task} --arch ${arch} \
    --w2v-path ${petrained_model_path}   --labels ${labels} \
    --apply-mask --mask-selection ${mask_selection} --mask-other ${mask_other} --mask-length ${mask_length} --mask-prob ${mask_prob} \
    --layerdrop ${layerdrop}  --mask-channel-selection ${mask_channel_selection} --mask-channel-other ${mask_channel_other} --mask-channel-length ${mask_channel_length} \
    --mask-channel-prob ${mask_channel_prob} --zero-infinity  --feature-grad-mult ${feature_grad_mult} --freeze-finetune-updates ${freeze_finetune_updates} \
    --validate-after-updates ${validate_after_updates} --optimizer ${optimizer} --adam-betas ${adam_betas} --adam-eps ${adam_eps} \
    --lr ${lr} --lr-scheduler ${lr_scheduler} --warmup-steps ${warmup_steps} --hold-steps ${hold_steps} --decay-steps ${decay_steps} \
    --final-lr-scale ${final_lr_scale} --final-dropout ${final_dropout} --dropout ${dropout} --activation-dropout ${activation_dropout} --criterion ${criterion} \
    --attention-dropout ${attention_dropout} --max-tokens ${max_tokens} --seed ${seed}  --log-format ${log_format} --log-interval ${log_interval} \
    --ddp-backend ${ddp_backend} --update-freq ${update_freq} \
    --tensorboard-logdir ${tensorboard_path}


fi
