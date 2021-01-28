#!/bin/bash


### Variables to change - start ###

config_name='pretraining_large'
gpus=1
run_in_nohup=0  #0 for no, 1 for yes

### Variables to change - end ###

dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

printf "** Directory to code is: $parentdir"

config_path=${parentdir}'/config'
data_path=${parentdir}'/data/pretraining'
PORT=-1
checkpoints_path=${parentdir}'/checkpoints/pretraining'
log_path=${parentdir}'/logs/pretraining'
tensorboard_path=${log_path}'/tensorboard'
wav2vec_repo_path=${parentdir}'/../fairseq/'

# importing parse_yaml.sh file
. ${parentdir}'/scripts/'parse_yaml.sh
eval $(parse_yaml ${config_path}'/'${config_name}.yaml)

update_freq=$(($gpus_to_simulate/$gpus))
#you can simulate 64/128 GPUs by using k GPUs in base/large models respectively

printf "\n** Config path is: $config_path/$config_name.yaml"
printf "\n** Data path is: $data_path"
printf "\n** Checkpoint will be saved at: $checkpoints_path"
printf "\n** Logs will be saved at: ${log_path}"
printf "\n** Update frequency is: ${update_freq}"

timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

if [ "${run_in_nohup}" = 1 ]; then

	local_timestamp=$(timestamp)
	tensorboard_path=${tensorboard_path}_${local_timestamp}
	mkdir -p ${tensorboard_path}

	printf "\n** Tensorboard is running **"
	printf "\n** Tensorboard logs path: ${tensorboard_path}"
	printf "\n"

	nohup python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port $PORT ${data_path} \
    --save-dir ${checkpoints_path} --fp16 --num-workers ${num_workers} --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
    --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
    --conv-feature-layers ${conv_feature_layers} --final-dim ${final_dim} --latent-vars ${latent_vars} \
    --latent-groups ${latent_groups} --latent-temp ${latent_temp} --infonce --optimizer ${optimizer} \
    --adam-betas ${adam_betas} --adam-eps ${adam_eps} --lr-scheduler ${lr_scheduler} --total-num-update ${total_num_update} \
    --lr ${lr} --warmup-updates ${warmup_updates} --mask-length ${mask_length} --mask-prob ${mask_prob} --mask-selection ${mask_selection} --mask-other ${mask_other} \
    --encoder-layerdrop ${encoder_layerdrop} --dropout-input ${dropout_input} --dropout-features ${dropout_features} --feature-grad-mult ${feature_grad_mult} \
    --loss-weights ${loss_weights} --conv-pos ${conv_pos} --conv-pos-groups ${conv_pos_groups} --num-negatives ${num_negatives} --cross-sample-negatives ${cross_sample_negatives} \
    --max-sample-size ${max_sample_size} --min-sample-size ${min_sample_size} --dropout ${dropout} --attention-dropout ${attention_dropout} --weight-decay ${weight_decay} \
    --max-tokens ${max_tokens} --max-update ${max_update} --skip-invalid-size-inputs-valid-test --ddp-backend ${ddp_backend} --update-freq ${update_freq} \
    --encoder-layers ${encoder_layers} --encoder-embed-dim ${encoder_embed_dim} --encoder-ffn-embed-dim ${encoder_ffn_embed_dim} --encoder-attention-heads ${encoder_attention_heads} \
    --tensorboard-logdir ${tensorboard_path}  &> ${log_path}/${local_timestamp}.out &

	nohup tensorboard --logdir ${tensorboard_path} --bind_all &> /dev/null &

else
	python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port $PORT ${data_path} \
    --save-dir ${checkpoints_path} --fp16 --num-workers ${num_workers} --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
    --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
    --conv-feature-layers ${conv_feature_layers} --final-dim ${final_dim} --latent-vars ${latent_vars} \
    --latent-groups ${latent_groups} --latent-temp ${latent_temp} --infonce --optimizer ${optimizer} \
    --adam-betas ${adam_betas} --adam-eps ${adam_eps} --lr-scheduler ${lr_scheduler} --total-num-update ${total_num_update} \
    --lr ${lr} --warmup-updates ${warmup_updates} --mask-length ${mask_length} --mask-prob ${mask_prob} --mask-selection ${mask_selection} --mask-other ${mask_other} \
    --encoder-layerdrop ${encoder_layerdrop} --dropout-input ${dropout_input} --dropout-features ${dropout_features} --feature-grad-mult ${feature_grad_mult} \
    --loss-weights ${loss_weights} --conv-pos ${conv_pos} --conv-pos-groups ${conv_pos_groups} --num-negatives ${num_negatives} --cross-sample-negatives ${cross_sample_negatives} \
    --max-sample-size ${max_sample_size} --min-sample-size ${min_sample_size} --dropout ${dropout} --attention-dropout ${attention_dropout} --weight-decay ${weight_decay} \
    --max-tokens ${max_tokens} --max-update ${max_update} --skip-invalid-size-inputs-valid-test --ddp-backend ${ddp_backend} --update-freq ${update_freq} \
    --encoder-layers ${encoder_layers} --encoder-embed-dim ${encoder_embed_dim} --encoder-ffn-embed-dim ${encoder_ffn_embed_dim} --encoder-attention-heads ${encoder_attention_heads}

fi
