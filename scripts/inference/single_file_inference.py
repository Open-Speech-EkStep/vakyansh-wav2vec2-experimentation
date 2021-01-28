import torch
import numpy as np
import argparse
import soundfile as sf
import torch.nn.functional as F
import itertools as it
from fairseq import utils
from fairseq.models import BaseFairseqModel
from fairseq.data import Dictionary
from fairseq.models.wav2vec.wav2vec2_asr import base_architecture, Wav2VecEncoder
from wav2letter.common import create_word_dict, load_words
from wav2letter.decoder import CriterionType,DecoderOptions,KenLM,LM,LMState,SmearingMode,Trie,LexiconDecoder
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes

class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, target_dict):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args, target_dict)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

class W2lDecoder(object):
    def __init__(self, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = 1

        self.criterion_type = CriterionType.CTC
        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        self.asg_transitions = None

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)

        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)

        return torch.LongTensor(list(idxs))

class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = list()

        if self.asg_transitions is None:
            transitions = torch.FloatTensor(N, N).zero_()
        else:
            transitions = torch.FloatTensor(self.asg_transitions).view(N, N)

        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}] for b in range(B)
        ]
 
class W2lKenLMDecoder(W2lDecoder):
    def __init__(self,args,tgt_dict):
        super().__init__(tgt_dict)

        self.silence = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )
        
        self.lexicon = load_words(args['lexicon'])
        self.word_dict = create_word_dict(self.lexicon)
        self.unk_word = self.word_dict.get_index("<unk>")

        self.lm = KenLM(args['kenlm_model'], self.word_dict)
        self.trie = Trie(self.vocab_size, self.silence)

        start_state = self.lm.start(False)
        for i, (word, spellings) in enumerate(self.lexicon.items()):
            word_idx = self.word_dict.get_index(word)
            _, score = self.lm.score(start_state, word_idx)
            for spelling in spellings:
                spelling_idxs = [tgt_dict.index(token) for token in spelling]
                assert (
                    tgt_dict.unk() not in spelling_idxs
                ), f"{spelling} {spelling_idxs}"
                self.trie.insert(spelling_idxs, word_idx, score)
        self.trie.smear(SmearingMode.MAX)

        self.decoder_opts = DecoderOptions(
            args['beam'],
            int(getattr(args, "beam_size_token", len(tgt_dict))),
            args['beam_threshold'],
            args['lm_weight'],
            args['word_score'],
            args['unk_weight'],
            args['sil_weight'],
            0,
            False,
            self.criterion_type,
        )

        if self.asg_transitions is None:
            N = 768
            # self.asg_transitions = torch.FloatTensor(N, N).zero_()
            self.asg_transitions = []

        self.decoder = LexiconDecoder(
            self.decoder_opts,
            self.trie,
            self.lm,
            self.silence,
            self.blank,
            self.unk_word,
            self.asg_transitions,
            False,
        )

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            results = self.decoder.decode(emissions_ptr, T, N)

            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "tokens": self.get_tokens(result.tokens),
                        "score": result.score,
                        "words": [
                            self.word_dict.get_entry(x) for x in result.words if x >= 0
                        ],
                    }
                    for result in nbest_results
                ]
            )
            #print(hypos)
        return hypos

def get_results(wav_path,dict_path,generator,use_cuda=False,w2v_path=None,model=None):
    sample = dict()
    net_input = dict()
    feature = get_feature(wav_path)
    target_dict = Dictionary.load(dict_path)
 
    model[0].eval()
           
    net_input["source"] = feature.unsqueeze(0)

    padding_mask = torch.BoolTensor(net_input["source"].size(1)).fill_(False).unsqueeze(0)

    net_input["padding_mask"] = padding_mask
    sample["net_input"] = net_input
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    with torch.no_grad():
        hypo = generator.generate(model, sample, prefix_tokens=None)
    hyp_pieces = target_dict.string(hypo[0][0]["tokens"].int().cpu())
    text=post_process(hyp_pieces, 'letter')

    return text

def get_feature(filepath):
    def postprocess(feats, sample_rate):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
        return feats

    wav, sample_rate = sf.read(filepath)
    feats = torch.from_numpy(wav).float()
    feats = postprocess(feats, sample_rate)
    return feats

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == 'wordpiece':
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == 'letter':
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol is not None and symbol != 'none':
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    return sentence

def load_gpu_model(model_path):
    return torch.load(model_path,map_location=torch.device("cuda"))

def load_cpu_model(model_path):
    return torch.load(model_path,map_location=torch.device("cpu"))

def get_args(lexicon_path, lm_path, BEAM=128, LM_WEIGHT=2, WORD_SCORE=-1):
    args = {}
    args['lexicon'] = lexicon_path
    args['kenlm_model'] = lm_path
    args['beam'] = BEAM
    args['beam_threshold'] = 25
    args['lm_weight'] = LM_WEIGHT
    args['word_score'] = WORD_SCORE
    args['unk_weight'] = -np.inf
    args['sil_weight'] = 0
    args['nbest'] = 1
    args['criterion'] ='ctc'
    args['labels']='ltr'
    return args

def parse_transcription(model_path, dict_path, wav_path, cuda, decoder="viterbi", lexicon_path=None, lm_path=None):
    target_dict = Dictionary.load(dict_path)
    if decoder=="viterbi":
        generator = W2lViterbiDecoder(target_dict)
    else:
        args = get_args(lexicon_path, lm_path)
        generator = W2lKenLMDecoder(args, target_dict)
    
    result = ''

    if cuda:
        gpu_model = load_gpu_model(model_path)
        result = get_results(wav_path=wav_path, dict_path=dict_path, generator=generator, use_cuda=cuda, model=gpu_model)
    else:
        cpu_model = load_cpu_model(model_path)
        result = get_results(wav_path=wav_path, dict_path=dict_path, generator=generator, use_cuda=cuda, model=cpu_model)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('-m', '--model', type=str, help="Custom model path")
    parser.add_argument('-d', '--dict', type=str, help="Dict path")
    parser.add_argument('-w', '--wav', type=str, help= "Wav file path")
    parser.add_argument('-c', '--cuda', default=False, type=bool, help="CUDA True or False")
    parser.add_argument('-D', '--decoder', type=str, help= "Which decoder to use kenlm or viterbi")
    parser.add_argument('-l', '--lexicon', default=None, type=str, help= "Lexicon path if decoder is kenlm")
    parser.add_argument('-L', '--lm-path', default=None, type=str, help= "Language mode path if decoder is kenlm")
    args_local = parser.parse_args()

    result = parse_transcription(args_local.model, args_local.dict, args_local.wav,  args_local.cuda, args_local.decoder, args_local.lexicon, args_local.lm_path)
    print(result)
