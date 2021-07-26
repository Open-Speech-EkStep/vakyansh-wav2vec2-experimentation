#!/bin/bash


### Values to change - start ###

config_name='base_10h.yaml'
gpus=4
run_in_nohup=0  #0 for no, 1 for yes

### Values to change - end ###

dir=$PWD/
parentdir="$(dirname "$dir")"
parentdir="$(dirname "$parentdir")"
printf "** Directory to code is: $parentdir"

config_path=${parentdir}'/config/finetuning/'
data_path=${parentdir}'/data/finetuning'
checkpoints_path=${parentdir}'/checkpoints/finetuning'
log_path=${parentdir}'/logs/finetuning'
tensorboard_path=${log_path}'/tensorboard'
pretrained_model_path=${parentdir}'/checkpoints/pretraining/wav2vec_small.pt'
update_freq=$((24/${gpus}))
#wav2vec_repo_path='/opt/fairseq/'

printf "\n** $wav2vec_repo_path"
printf "\n** Config path is: $config_path"
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
    task.data=${data_path} \
    model.w2v_path=${pretrained_model_path} \
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
    task.data=${data_path} \
    model.w2v_path=${pretrained_model_path} \
    distributed_training.distributed_world_size=${gpus} \
    +optimization.update_freq=[$update_freq] \
    +common.tensorboard_logdir=${tensorboard_path} \
    checkpoint.save_dir=${checkpoints_path} \
    checkpoint.restore_file=${checkpoints_path}/checkpoint_last.pt \
    --config-dir ${config_path} --config-name ${config_name} 

fi
