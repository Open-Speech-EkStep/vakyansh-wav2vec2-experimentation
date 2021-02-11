### Input Parameters description

---

#### For prepare_data

**wav_path:** Directory where your data is present, if data is present in multiple folders put them under one parent dir

**prep_scripts:** Path where you have cloned the preparation scripts

**valid_percentage:** Percentage of data you want to use for validation purpose. eg - 0.04 if 4%

#### For batch infer

**w2l_decoder_viterbi:** switch decoding method 1 for viterbi, 0 for kenlm(if you have language model)

**data_path:** path where tsv is present after running prepare_data.sh

**result_path:** path to store results file prouced after inference

**lm_model_path:** lm.binary path if decoding using kenlm

**lexicon_lst_path:** lexicon file made using the vocab file generated while building language model

#### For single_file_inference

**custom_model_path:** single custom_model generated using generate_custom_model.sh, this doesn't require pretraining checkpoint

**dictionary:** dict file generated during finetuning, contains character set used in finetuning

**wav_file_path:** audio file to be transcribe 

**decoder:** kenlm or viterbi

**cuda:** to use gpu for inference set it True