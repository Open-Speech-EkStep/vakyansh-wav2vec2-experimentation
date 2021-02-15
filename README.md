# wav2vec-template-v1 repo 
## Installation and Setup (requires wav2letter as well) 
```
git clone https://github.com/Open-Speech-EkStep/wav2vec-template-v1.git

conda create --name <env_name> python=3.7
conda activate <env_name>

### For fairseq setup(fairseq should be installed outside wav2vec-old repo)

git clone -b ekstep-wav2vec2 https://github.com/Open-Speech-EkStep/fairseq.git
cd fairseq
pip install -e .

### install other libraries

### For Kenlm, openblas

cd ..
sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev


git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16
cd ..
export KENLM_ROOT_DIR=$PWD
export USE_CUDA=0 ## for cpu


### wav2letter

git clone -b v0.2 https://github.com/facebookresearch/wav2letter.git
cd wav2letter
git checkout b1d1f89f586120a978a4666cffd45c55f0a2e564
cd bindings/python
pip install -e .

```
# Usage

## For Pretraining
Edit the path to data in the scripts/pretraining/prepare_data.sh file. To prepare the data:
```bash
cd scripts/pretraining/
bash prepare_data.sh
```
Edit the config/pretraining_base.yaml or config/pretraining_large.yaml for different parameter configurations.Check the required paths and values in start_pretraining_base.sh or start_pretraining_large.sh. Refer to [config README](config/README.md)
To start run:
```bash
bash start_pretraining_base.sh
```
Refer [this](scripts/pretraining/README.md) for pretraining parameters.

## For Finetuning
Edit the path to data in the scripts/finetuning/prepare_data.sh file. To prepare the data:
```bash
cd scripts/finetuning/
bash prepare_data.sh
```
Edit the config/finetuning.yaml for different parameter configurations.Check the required paths and values in start_pretraining.sh. Refer to [config README](config/README.md)
To start run:
```bash
bash start_finetuning.sh
```

Refer [this](scripts/finetuning/README.md) for finetuning parameters.

## For Inference
Edit the path to data in the scripts/inference/prepare_data.sh file. To prepare the test data run:
```bash
cd scripts/inference/
bash prepare_data.sh
```
Edit the infer.sh file for required paths. To start inference run:
```bash
bash infer.sh
```
Refer [this](scripts/inference/README.md) for inference parameters.

## For Single File Inference
To generate custom model, run: 
```bash
cd scripts/inference/
bash generate_model.sh
```
To infer for single file, change path in single_file_inference.sh. Then run:
```bash
bash single_file_inference.sh
```
## For generating LM
Edit the run_lm_pipeline.sh variables as required, then run:
```
cd scripts/lm/
bash run_lm_pipeline.sh
```
Refer [this](scripts/lm/README.md) for LM pipeline.

## License

fairseq(-py) is MIT-licensed. The license applies to the pre-trained models as well.
