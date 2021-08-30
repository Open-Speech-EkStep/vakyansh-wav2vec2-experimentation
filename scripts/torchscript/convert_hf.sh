input_model_name = 'test'
output_dir = '../../checkpoints/ts/'

mkdir -p ${output_dir}

python ../../utils/torchscript/convert_hf.py -i ${input_model_name} -o ${output_dir}

echo "Torchscript Model saved"
