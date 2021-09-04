import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchaudio
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from transformers import Wav2Vec2ForCTC
import json
import argparse
import os

class SpeechRecognizer(torch.nn.Module):
    def __init__(self, model, vocab):
        super().__init__()
        self.model = model
        vocab = vocab
        self.labels = list(vocab.keys())

    def forward(self, waveforms: Tensor) -> str:
        """Given a single channel speech data, return transcription.
        Args:
            waveforms (Tensor): Speech tensor. Shape `[1, num_frames]`.
        Returns:
            str: The resulting transcript
        """
        logits, _ = self.model(waveforms)  # [batch, num_seq, num_label]
        best_path = torch.argmax(logits[0], dim=-1)  # [num_seq,]
        prev = ''
        hypothesis = ''
        for i in best_path:
            char = self.labels[i]
            if char == prev:
                continue
            if char == '<s>':
                prev = ''
                continue
            hypothesis += char
            prev = char
        return hypothesis.replace('|', ' ')

def read_vocab(hf_model_name):
    vocab = f'https://huggingface.co/{hf_model_name}/raw/main/vocab.json'
    os.system('wget ' +vocab)
    with open('vocab.json', encoding='utf-8') as file:
        vocab = json.load(file)
    
    return vocab

def convert_model(hf_model_name, output_dir):
    # Load Wav2Vec2 pretrained model from Hugging Face Hub
    model = Wav2Vec2ForCTC.from_pretrained(hf_model_name)
    # Convert the model to torchaudio format, which supports TorchScript.
    model = import_huggingface_model(model)
    # Remove weight normalization which is not supported by quantization.
    model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()
    model = model.eval()
    # Attach decoder
    model = SpeechRecognizer(model, read_vocab(hf_model_name))

    # Apply quantization / script / optimize for mobile
    quantized_model = torch.quantization.quantize_dynamic(
        model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    scripted_model = torch.jit.script(quantized_model)
    optimized_model = optimize_for_mobile(scripted_model)
    quant_model_name = hf_model_name.split('/')[-1] + '_quant.pt'
    os.makedirs(output_dir, exist_ok=True)
    optimized_model.save(output_dir+ '/' + quant_model_name)
    os.system(f'mv vocab.json {output_dir}/')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-model', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str,  required=True)
    args = parser.parse_args()

    convert_model(args.hf_model, args.output)
