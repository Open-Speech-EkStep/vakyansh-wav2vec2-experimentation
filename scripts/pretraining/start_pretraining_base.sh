#!/bin/bash


### Variables to change - start ###

config_name='wav2vec2_base_librispeech' 
gpus=4
run_in_nohup=0  #0 for no, 1 for yes
 
### Variables to change - end ###

dir=$PWD
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"

printf "** Directory to code is: $parentdir"

config_path=${parentdir}'/config/pretraining/'
data_path=${parentdir}'/data/pretraining'
PORT=-1
checkpoints_path=${parentdir}'/checkpoints/pretraining'
log_path=${parentdir}'/logs/pretraining'
tensorboard_path=${log_path}'/tensorboard'
#wav2vec_repo_path=${parentdir}'/../fairseq/'

# importing parse_yaml.sh file
#. ${parentdir}'/scripts/'parse_yaml.sh
#eval $(parse_yaml ${config_path}'/'${config_name}.yaml)

update_freq=$((64/$gpus)) 
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

	nohup fairseq-hydra-train \
	distributed_training.distributed_port=${PORT} \
	task.data=${data_path} \
	distributed_training.distributed_world_size=${gpus} \
	+optimization.update_freq=[$update_freq] \
	+common.tensorboard_logdir=${tensorboard_path} \
	checkpoint.save_dir=${checkpoints_path} \
	checkpoint.restore_file=${checkpoints_path}/checkpoint_last.pt \
	--config-dir ${config_path} \
	--config-name ${config_name} &> ${log_path}/${local_timestamp}.out & 
	
	nohup tensorboard --logdir ${tensorboard_path} --bind_all &> /dev/null &
    
else

	fairseq-hydra-train \
	distributed_training.distributed_port=${PORT} \
	task.data=${data_path} \
	distributed_training.distributed_world_size=${gpus} \
	+optimization.update_freq=[$update_freq] \
	+common.tensorboard_logdir=${tensorboard_path} \
	checkpoint.save_dir=${checkpoints_path} \
	checkpoint.restore_file=${checkpoints_path}/hindi_pretrained.pt \
	--config-dir ${config_path} \
	--config-name ${config_name} 
fi
