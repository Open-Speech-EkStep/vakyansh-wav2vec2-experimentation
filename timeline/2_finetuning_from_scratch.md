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

Discuss important paramters of config:
**wandb_project** -> expects name of wandb project. Useful to monitor metrics

**num_workers** -> number of workers in the dataloader to load data

**max_tokens** -> the maximum tokens you can fit at one time into the GPU. 3200000 / 16000 ~ 200 seconds ~ 3.3 minutes

**max_updates** -> maximum updates to which you want to run training for

**lr** -> the learning rate to use during training

**update_freq** -> How frequently the gradient updates happen in distributed environment. Depends on the number of GPU's to simulate and you have currently.

**optimizer** -> optimizer to use for training purpose

**w2v_path** -> path to pretrained model

**mask_prob** -> probability of masking the latent speech output

**freeze_finetune_updates** -> the updates for which you want to freeze the entire network apart from the final layer

Difference between different configs
