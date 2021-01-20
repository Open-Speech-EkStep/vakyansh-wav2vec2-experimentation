# wav2vec-old repo 
## Installation (requires wav2letter as well) 
```
conda create --name fairseq python=3.7
conda activate fairseq

### For fairseq setup

git clone https://github.com/Open-Speech-EkStep/wav2vec.git
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
cmake .. -DCMAKE_BUILD_TYPE=Release -DKENLM_MAX_ORDER=20 -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j16
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
## Usage
Making the manifest .tsv file
```bash
python manifest.py <path to wav files> --dest <path to destination tsv> --ext wav --train-name train.tsv --valid-percent 0 --jobs -1
```

Making the .wrd, .ltr file
```bash
python labels.py --jobs 64 --tsv <path to train.tsv>train.tsv --output-dir <destination dir> --output-name test --txt-dir
```

Making the lexicon.lst and dict.ltr.txt file
```bash
python dict_and_lexicon_maker.py --wrd <path to train.wrd>train.wrd --lexicon <destnation path>/lexicon.lst --dict <destnation path>/dict.ltr.txt
```



## License

fairseq(-py) is MIT-licensed. The license applies to the pre-trained models as well.
