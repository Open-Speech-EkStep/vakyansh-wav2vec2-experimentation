## Input Parameters description

---

### For prepare_data

Wav files and its corresponding text files should be present in the same folder with same name. eg - audio_id/audio.wav, audio_id/audio.txt

**make_valid_from_train** : If valid set is already present in different folder make this 0 else make it 1 to extract valid percentage from train data 

**train_wav_path:** Directory where your train data(wav files) is present, if wav files are present in multiple folders put them under one parent directory

**valid_wav_path:** Used only when 'make_valid_from_train' is 0; directory where your valid data(wav files) is present, if wav files are present in multiple folders put them under one parent directory

**valid_percentage:** Used only when 'make_valid_from_train' is 1; percentage of data to be used for validation purpose. eg - 0.04 if 4%

**prep_scripts:** Path for utility scripts

### For start_finetuning

**config_name:** This file contains configurable parameters for finetuning

**gpus:** Number of gpus to use

**run_in_nohup:** Make it 1 for running training in background

**data_path:** Contains files made from running prepare_data.sh

**checkpoints_path** Directory to save checkpoints generated after each epoch during finetuning. eg- checkpoint_best, checkpoint_last

**log_path:** nohup.out is saved here as <timestamp_of_running>.out

**tensordboard_path:** Path where tensorboard logs are to be saved

**pretrained_model_path:** Path of checkpoint_best.pt generate from pretraining

**update_freq:** To simulate n gpus by k gpus update frequency will be [n/k]

**wav2vec_repo_path:** Path of fairseq repository