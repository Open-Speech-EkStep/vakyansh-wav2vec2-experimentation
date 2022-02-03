## Input Parameters description

---

### For prepare_data

Wav files and its corresponding text files should be present in the same folder with same name. eg - audio_id/audio.wav, audio_id/audio.txt

**inference_data_name:** Name of the folder where results of prepare_data.sh will be saved

**wav_path:** Directory where your data(wav files) is present, if wav files are present in multiple folders put them under one parent directory

**prep_scripts:** Path for utility scripts

**valid_percentage:** Percentage of data to be used for validation purpose. eg - 0.04 if 4%

### For batch infer

**w2l_decoder_viterbi:** Switch decoding method to 1 for viterbi, 0 for kenlm(for decoding with LM)

**inference_data_name:** Name of the folder where results of prepare_data.sh were saved.
A folder with the same name will be created in the results directory as well containing predictions.

**beam:** Set beam according to need for decoding(viterbi/kenlm).

**lm_name:** Name of the folder containing the lm files(eg: lm.binary etc)

**data_path:** Path where tsv is present after running prepare_data.sh

**result_path:** Path to store results file prouced after inference

**lm_model_path:** lm.binary path if decoding using kenlm

**lexicon_lst_path:** Lexicon file made using the vocab file generated while building language model

**validation_dataset:** Name of the validation folder to be stored in the results 

**save_predicted:** To save the predicted files by the ASR model. Set it as 1 to save the files, else 0 by default 

**dest_folder:** Destination folder to save predicted files. It will save in the same folder structure order as the original inferenece data

### For single_file_inference

**custom_model_path:** Single custom_model generated using generate_custom_model.sh, this doesn't require pretraining checkpoint

**dictionary:** Dict file generated during finetuning, contains character set used in finetuning

**wav_file_path:** Audio file to be transcribe 

**decoder:** Choose kenlm or viterbi

**cuda:** To use gpu for inference set it True