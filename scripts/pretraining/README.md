## Input Parameters description

---

### For prepare_data

**wav_path:** Directory where your data(wav files) is present, if wav files are present in multiple folders put them under one parent directory

**valid_percentage:** Percentage of data to be used for validation purpose. eg - 0.04 if 4%

### For start_pretraining_[base/large]

**config_name:** This file contains configurable parameters for pretraining

**gpus:** Number of gpus to use

**run_in_nohup:** Make it 1 for running training in background

**data_path:** Contains files made from running prepare_data.sh

**checkpoints_path** Directory to save and update best and last checkpoints after each epochs

**log_path:** nohup.out is saved here as <timestamp_of_running>.out

**tensordboard_path:** Path where tensorboard logs are saved

**update_freq:** To simulate n gpus by k gpus update frequency will be [n/k]

**wav2vec_repo_path:** Path of fairseq repositiory.