

import torch
import argparse
from fairseq import utils
from fairseq.models import BaseFairseqModel
from fairseq.data import Dictionary
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecEncoder, Wav2Vec2CtcConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, target_dictionary): ##change here
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


from argparse import Namespace
def load_model(model_path, target_dict, pretrained_model):
    w2v = torch.load(model_path) #,map_location=torch.device("cpu"))
    #args = w2v.get("args", None) or w2v["cfg"].model
    
    if w2v.get("args", None) is not None:
        #print('Here')
        args = convert_namespace_to_omegaconf(w2v["args"])["model"]
        args['w2v_args']=None
        args['w2v_path'] = pretrained_model
    else:
        #print('here2')
        args = convert_namespace_to_omegaconf(w2v["cfg"]['model'])['model']
        #args['w2v_path'] = pretrained_model
        if not args:
            #print('here3')
            args = w2v["cfg"]['model']
            args['w2v_args']=None
            args['w2v_path'] = pretrained_model
            args = Namespace(**args)
    
    #print(args)
    #args['w2v_path'] = pretrained_model
        
    #print(args)    
    model = Wav2VecCtc.build_model(args, target_dict)
    
    model.load_state_dict(w2v["model"], strict=True)

    return model


def generate_custom_model(finetuned_path,pretrained_model, dictionary_path,final_model_path):
    target_dict = Dictionary.load(dictionary_path)
    model = load_model(finetuned_path, target_dict, pretrained_model)
    #print(model)
    torch.save(model,final_model_path)

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert default models to combined model')
    parser.add_argument('-f', '--finetuned_model_path', type=str, help="Fine-tuned Model path")
    parser.add_argument('-p', '--pretrained_model_path',required=False, type=str, help="Pre-tuned Model path")
    parser.add_argument('-d', '--dict', type=str, help="Dict path")
    parser.add_argument('-o', '--output_path', type=str, default='final_model.pt', help= "Final model path")
    args_local = parser.parse_args()

    generate_custom_model(finetuned_path=args_local.finetuned_model_path, pretrained_model = args_local.pretrained_model_path,
                          dictionary_path=args_local.dict,final_model_path=args_local.output_path)
