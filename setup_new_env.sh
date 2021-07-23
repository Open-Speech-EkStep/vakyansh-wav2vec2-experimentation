#!/bin/sh

#conda init
#source ~/.bashrc
#conda create -y --name f1 python=3.7
#conda activate f1

sudo apt-get -y install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev sox
sudo apt -y install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev 

pip install packaging soundfile swifter
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt



cd /opt
sudo mkdir wav2vec
sudo chmod 777 -R wav2vec
cd wav2vec
git clone https://github.com/Open-Speech-EkStep/fairseq -b v2-hydra
cd fairseq
pip install -e .
cd ..

git clone https://github.com/kpu/kenlm.git
sudo chmod 777 -R kenlm
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16
cd ..
export KENLM_ROOT=$PWD
cd ..

git clone https://github.com/flashlight/flashlight.git
sudo chmod 777 -R flashlight
cd flashlight/bindings/python
export USE_MKL=0
python setup.py install
