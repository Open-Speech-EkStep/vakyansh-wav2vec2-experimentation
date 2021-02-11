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

	nohup python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port -1 ${data_path} \
    --save-dir ${checkpoints_path} --fp16 --no-epoch-checkpoints --num-workers ${num_workers} --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
    --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
    --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 768 --latent-vars 320 \
	--latent-groups 2 --latent-temp '(2.0,0.1,0.999995)' --infonce --optimizer adam \
	--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 600000 \
	--lr 0.0003 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
	--encoder-layerdrop 0.0 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.03 \
	--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
	--max-sample-size ${max_sample_size} --min-sample-size ${min_sample_size} --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.01 \
	--encoder-layers 24 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 \
    --max-tokens ${max_tokens} --max-update ${max_update} --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --update-freq ${update_freq} \
    --tensorboard-logdir ${tensorboard_path}  &> ${log_path}/${local_timestamp}.out &

	nohup tensorboard --logdir ${tensorboard_path} --bind_all &> /dev/null &

else
	python ${wav2vec_repo_path}train.py --distributed-world-size ${gpus} --distributed-port -1 ${data_path} \
    --save-dir ${checkpoints_path} --fp16 --no-epoch-checkpoints --num-workers ${num_workers} --task audio_pretraining --criterion wav2vec --arch wav2vec2 \
    --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --extractor-mode default \
    --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --final-dim 768 --latent-vars 320 \
	--latent-groups 2 --latent-temp '(2.0,0.1,0.999995)' --infonce --optimizer adam \
	--adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 600000 \
	--lr 0.0003 --warmup-updates 32000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 \
	--encoder-layerdrop 0.0 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.03 \
	--loss-weights '[0.1, 10]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 \
	--max-sample-size ${max_sample_size} --min-sample-size ${min_sample_size} --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.01 \
	--encoder-layers 24 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 \
    --max-tokens ${max_tokens} --max-update ${max_update} --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --update-freq ${update_freq}

fi
