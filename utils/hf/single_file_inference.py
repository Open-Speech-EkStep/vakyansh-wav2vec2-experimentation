import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import argparse

def parse_transcription(checkpoint_path, wav_file):
    # load pretrained model
    processor = Wav2Vec2Processor.from_pretrained(checkpoint_path)
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_path)



    # load audio
    audio_input, sample_rate = sf.read(wav_file)

    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values

    # INFERENCE

    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribe
    transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
    print(transcription)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--checkpoint_path", default=None, type=str, help="Path to hf checkpoint")
    parser.add_argument("-w", "--wav_path", default=None, type=str, help="Path to wav")
    args = parser.parse_args()
    parse_transcription(args.checkpoint_path, args.wav_path)

