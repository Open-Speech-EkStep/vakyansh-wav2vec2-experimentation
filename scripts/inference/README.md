### Input Parameters description

---

#### For prepare_data

Wav files and its corresponding text files should be present in the same folder with same name. eg - audio_id/audio.wav, audio_id/audio.txt

**wav_path:** Directory where your data(wav files) is present, if wav files are present in multiple folders put them under one parent directory

**txt_path:** Directory where your data(text files) is present, if text files are present in multiple folders put them under one parent directory

**prep_scripts:** Path for utility scripts

**valid_percentage:** Percentage of data to be used for validation purpose. eg - 0.04 if 4%

#### For batch infer

**w2l_decoder_viterbi:** Switch decoding method 1 for viterbi, 0 for kenlm(if you have language model)

**data_path:** Path where tsv is present after running prepare_data.sh

**result_path:** Path to store results file prouced after inference

**lm_model_path:** lm.binary path if decoding using kenlm

**lexicon_lst_path:** Lexicon file made using the vocab file generated while building language model

#### For single_file_inference

**custom_model_path:** Single custom_model generated using generate_custom_model.sh, this doesn't require pretraining checkpoint

**dictionary:** Dict file generated during finetuning, contains character set used in finetuning

**wav_file_path:** Audio file to be transcribe 

**decoder:** Choose kenlm or viterbi

**cuda:** To use gpu for inference set it True