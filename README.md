# wav2vec-old repo 
## Installation and Setup (requires wav2letter as well) 
```
git clone https://github.com/Open-Speech-EkStep/wav2vec-old.git

conda create --name <env_name> python=3.7
conda activate <env_name>

### For fairseq setup(fairseq should be installed outside wav2vec-old repo)
cd ..
git clone https://github.com/pytorch/fairseq.git
git checkout cd2bba4419629ffc17eb83c669e88b0bd3af6eb9
cd wav2vec
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

### Packages
pip install packaging soundfile


### wav2letter

git clone https://github.com/Open-Speech-EkStep/wav2letter.git 
cd wav2letter/bindings/python
pip install -e .

```
# Usage

## For Pretraining
Edit the path to data in the scripts/pretraining/prepare_data.sh file.To prepare the data:
```bash
cd scripts/pretraining/
bash prepare_data.sh
```
Edit the config/pretraining_base.yaml or config/pretraining_large.yaml for different parameter configurations.Check the required paths and values in start_pretraining_base.sh or start_pretraining_large.sh.To start run:
```bash
bash start_pretraining_base.sh
```

## For Finetuning
Edit the path to data in the scripts/finetuning/prepare_data.sh file.To prepare the data:
```bash
cd scripts/finetuning/
bash prepare_data.sh
```
Edit the config/finetuning.yaml for different parameter configurations.Check the required paths and values in start_pretraining.sh.To start run:
```bash
bash start_pretraining.sh
```

## For Inference
Edit the path to data in the scripts/finetuning/prepare_data.sh file. To prepare the test data run:
```bash
cd scripts/inference/
bash prepare_data.sh
```
Edit the infer.sh file for required paths. To start inference run:
```bash
bash infer.sh
```
## For Single File Inference
To generate custom model, run: 
```bash
cd scripts/inference/
bash generate_model.sh
```
To infer for single file, change path in single_file_inference.sh.Then run:
```bash
bash single_file_inference.sh
```

## License

fairseq(-py) is MIT-licensed. The license applies to the pre-trained models as well.
