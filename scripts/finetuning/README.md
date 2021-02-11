### Input Parameters description

---

#### For prepare_data

wav files and its corresponding text files should be present in the same folder with same name. eg - audio_id/audio.wav, audio_id/audio.txt

**wav_path:** Directory where your data(wav files) is present, if wav files are present in multiple folders put them under one parent directory

**txt_path:** Directory where your data(text files) is present, if text files are present in multiple folders put them under one parent directory

**prep_scripts:** path for utility scripts

**valid_percentage:** Percentage of data you want to use for validation purpose. eg - 0.04 if 4%

#### For start_finetuning

**config_file:** this file contains configurable parameters for finetuning

**gpus:** number of gpus to use

**run_in_nohup:** make it 1 for running training in background

**data_path:** contains files made from running prepare_data.sh

**checkpoints_path** Directory to save checkpoints generated after each epoch during finetuning. eg- checkpoint_best, checkpoint_last

**log_path:** nohup.out is saved here as <timestamp_of_running>.out

**tensordboard_path:** path where tensorboard logs are to be saved

**pretrained_model_path:** path of checkpoint_best.pt generate from pretraining

**update_freq:** to simulate n gpus by k gpus update frequency will be [n/k]

**wav2vec_repo_path:** path of fairseq repository