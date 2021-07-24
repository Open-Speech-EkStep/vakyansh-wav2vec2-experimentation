# Step 2: Finetuning from scratch

## Step 2.1:Get Pretrained model (CLSRIL-23) from:
https://github.com/Open-Speech-EkStep/vakyansh-models

```
cd vakyansh-wav2vec2-experimentation/checkpoints/pretraining
wget https://storage.googleapis.com/vakyaansh-open-models/hindi/pretrained/CLSRIL-23.pt
```

## Step 2.2: Decide on the configuration for finetuning
Check available configs

```
cd vakyansh-wav2vec2-experimentation/config/finetuning
ls
```

a) Discuss important paramters of config:

  &nbsp;&nbsp;**wandb_project** -> expects name of wandb project. Useful to monitor metrics

  &nbsp;&nbsp;**num_workers** -> number of workers in the dataloader to load data

  &nbsp;&nbsp;**max_tokens** -> the maximum tokens you can fit at one time into the GPU. 3200000 / 16000 ~ 200 seconds ~ 3.3 minutes

  &nbsp;&nbsp;**max_updates** -> maximum updates to which you want to run training for

  &nbsp;&nbsp;**lr** -> the learning rate to use during training

  &nbsp;&nbsp;**update_freq** -> How frequently the gradient updates happen in distributed environment. Depends on the number of GPU's to simulate and you have currently.

  &nbsp;&nbsp;**optimizer** -> optimizer to use for training purpose

  &nbsp;&nbsp;**w2v_path** -> path to pretrained model

  &nbsp;&nbsp;**mask_prob** -> probability of masking the latent speech output

  &nbsp;&nbsp;**freeze_finetune_updates** -> the updates for which you want to freeze the entire network apart from the final layer


b) Difference between different configs

## Step 2.3: Define parameters in start_finetuning.sh script

```
cd vakyansh-wav2vec2-experimentation/scripts/finetuning
ls
vi start_finetuning.sh
```
a) Define **config_name** as decided in step 2.2

b) Define number of GPU's available using **gpus** variable

c) run_in_nohup: To run the process in background


### Step 2.4 Start Finetuning

```
bash start_finetuning.sh
```

- run without nohup initially
- login with wandb
- press Ctrl + C after epoch has started to stop the training
- ```sudo pkill -9 python``` to kill all the python processes
- check ```nvidia-smi``` to confirm no process is runnning on GPU

a) Check logs path

b) Check tensorboard

c) Check W&B

### Step 2.5 Stop training after 2 epochs

Resume from checkpoint_500. Get checkpoint from here:
```
cd vakyansh-wav2vec2-experimentation/checkpoints/finetuning
rm checkpoint*
wget https://storage.googleapis.com/vakyaansh-open-models/ieee/checkpoints/warm/checkpoint_last.pt
```

### Step 2.6 Let it run till 1000 updates

### Step 2.7 Inference using viterbi

### Step 2.8 Create LM using training data

### Step 2.8 Inference using kenlm

### Step 2.9 Show results section
