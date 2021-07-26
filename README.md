# Vakyansh Wav2Vec2 Experimentaion

## Pretrained Models
We are releasing pretrained models in various Indic Languages. Please head over to [this repo](https://github.com/Open-Speech-EkStep/vakyansh-models).

## Table of contents

<!--ts-->  
   * [Installation and Setup](#installation-and-setup)
   * [Directory Structure](#directory-structure)
   * [Data Description](#data-description)
   * [Usage](#usage)
      * [For Pretraining](#for-pretraining)
      * [For Finetuning](#for-finetuning)
      * [For Inference](#for-inferences)
      * [For Single File Inference](#for-single-file-inference)
   * [License](#license)
<!--te-->

## Installation and Setup 
### Create an Environment

```
conda create --name <env_name> python=3.7
conda activate <env_name>

sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev

pip install packaging soundfile swifter
pip install -r requirements.txt
```

### Install fairseq
```
git clone https://github.com/Open-Speech-EkStep/fairseq -b v2-hydra
cd fairseq
pip install -e .
cd ..
```

### Install KenLM
```
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16
cd ..
export KENLM_ROOT=$PWD
# export USE_CUDA=0 ## for cpu
cd ..
```

### Install Flashlight
```
git clone https://github.com/flashlight/flashlight.git
cd flashlight/bindings/python
export USE_MKL=0
python setup.py install
# python criterion_example.py  ## to test

```

## Directory Structure
```
root-directory
.
|-- ./checkpoints
|   |-- ./checkpoints/custom_model
|   |   `-- ./checkpoints/custom_model/
|   |-- ./checkpoints/finetuning
|   |   `-- ./checkpoints/finetuning/
|   `-- ./checkpoints/pretraining
|       `-- ./checkpoints/pretraining/
|-- ./data
|   |-- ./data/finetuning
|   |   `-- ./data/finetuning/
|   |-- ./data/inference
|   |   `-- ./data/inference/
|   |-- ./data/pretraining
|   |   `-- ./data/pretraining/
|   `-- ./data/processed
|       `-- ./data/processed/
|-- ./lm
|   `-- ./lm/
|-- ./logs
|   |-- ./logs/finetuning
|   |   `-- ./logs/finetuning/
|   `-- ./logs/pretraining
|       `-- ./logs/pretraining/
|-- ./notebooks
|   `-- ./notebooks/
|-- ./results
|   `-- ./results/
|-- ./scripts
|   |-- ./scripts/data
|   |   `-- ./scripts/data/
|   |-- ./scripts/parse_yaml.sh
|   |-- ./scripts/finetuning
|   |   |-- ./scripts/finetuning/start_finetuning.sh
|   |   |-- ./scripts/finetuning/prepare_data.sh
|   |   `-- ./scripts/finetuning/README.md
|   |-- ./scripts/lm
|   |   |-- ./scripts/lm/run_lm_pipeline.sh
|   |   `-- ./scripts/lm/README.md
|   |-- ./scripts/pretraining
|   |   |-- ./scripts/pretraining/start_pretraining_base.sh
|   |   |-- ./scripts/pretraining/start_pretraining_large.sh
|   |   |-- ./scripts/pretraining/prepare_data.sh
|   |   `-- ./scripts/pretraining/README.md
|   `-- ./scripts/inference
|       |-- ./scripts/inference/infer.sh
|       |-- ./scripts/inference/prepare_data.sh
|       |-- ./scripts/inference/generate_custom_model.sh
|       |-- ./scripts/inference/single_file_inference.sh
|       `-- ./scripts/inference/README.md
|-- ./config
|   |-- ./config/finetuning.yaml
|   |-- ./config/pretraining_base.yaml
|   |-- ./config/pretraining_large.yaml
|   `-- ./config/README.md
|-- ./requirements.txt
|-- ./utils
|   |-- ./utils/analysis
|   |   `-- ./utils/analysis/generate_wav_report_from_tsv.py
|   |-- ./utils/prep_scripts
|   |   |-- ./utils/prep_scripts/dict_and_lexicon_maker.py
|   |   |-- ./utils/prep_scripts/labels.py
|   |   `-- ./utils/prep_scripts/manifest.py
|   |-- ./utils/wer
|   |   |-- ./utils/wer/wer.py
|   |   `-- ./utils/wer/wer_wav2vec.py
|   |-- ./utils/inference
|   |   |-- ./utils/inference/generate_custom_model.py
|   |   `-- ./utils/inference/single_file_inference.py
|   `-- ./utils/lm
|       |-- ./utils/lm/concatenate_text.py
|       |-- ./utils/lm/make_lexicon_lst.py
|       |-- ./utils/lm/generate_lm.py
|       |-- ./utils/lm/clean_text.py
|       `-- ./utils/lm/remove_duplicate_lines.py
`-- ./README.md         
```

## Data Description

<!--ts-->
* For Audio Files.
    * Sample Rate [Hz] = 16000
    * Channels = 'mono'
    * Bit Rate [kbit/s] = 256
    * Precision [bits] = 16
    * Audio length should be less than 30 seconds otherwise it will be ignored during data preparation
    * After scripts/finetuning/prepare_data.sh is run, analysis will be generated which can be used to tune min/max_sample_size in the config files
* For Text Files
    * Corresponding text file of each audio file must be on the same directory as its audio
    * Text file should not contain any punctuation characters
    * Check dict.ltr.txt file generated after prepare_data so that it does not contain any foreign language character
* For Language Model
    * Character set of text used for language model should be same as character set used for training
    * Sample code for cleaning text file for english language is given here [clean_text.py](utils/lm/clean_text.py)
    * Sample code for removing duplicate line from text file is given here [remove_duplicate_lines.py](utils/lm/remove_duplicate_lines.py)
<!--te-->

# Usage

## For Pretraining
Edit the path to data in the scripts/pretraining/prepare_data.sh file. To prepare the data:
```
$ cd scripts/pretraining
$ bash prepare_data.sh
```
Edit the config/pretraining_base.yaml or config/pretraining_large.yaml for different parameter configurations.Check the required paths and values in start_pretraining_base.sh or start_pretraining_large.sh. Refer to [config README](config/README.md)
To start run:
```

$ bash start_pretraining_base.sh
```
Refer [this](scripts/pretraining/README.md) for pretraining parameters.

## For Finetuning
Edit the path to data in the scripts/finetuning/prepare_data.sh file. To prepare the data:
```
$ cd scripts/finetuning
$ bash prepare_data.sh
```
Edit the config/finetuning.yaml for different parameter configurations.Check the required paths and values in start_finetuning.sh. Refer to [config README](config/README.md)
To start run:
```
$ bash start_finetuning.sh
```

Refer [this](scripts/finetuning/README.md) for finetuning parameters.

## For Inference
Edit the path to data in the scripts/inference/prepare_data.sh file. To prepare the test data run:
```
$ cd scripts/inference/
$ bash prepare_data.sh
```
Edit the infer.sh file for required paths. To start inference run:
```
$ bash infer.sh
```
Refer [this](scripts/inference/README.md) for inference parameters.

## For Single File Inference
For single file inference, there are different models. Please head over to this (repo)[https://github.com/Open-Speech-EkStep/vakyansh-models#finetuned-models] to download the same

To generate custom model, run: 
```
$ cd scripts/inference
$ bash generate_custom_model.sh
```
To infer for single file, change path in single_file_inference.sh. Then run:
```
$ bash single_file_inference.sh
```

## For generating LM
Edit the run_lm_pipeline.sh variables as required, then run:
```
$ cd scripts/lm
$ bash run_lm_pipeline.sh
```
Refer [this](scripts/lm/README.md) for LM pipeline.

# License

fairseq(-py) is MIT-licensed. The license applies to the pre-trained models as well.
