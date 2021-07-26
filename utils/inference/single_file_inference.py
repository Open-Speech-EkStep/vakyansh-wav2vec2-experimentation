import torch
import numpy as np
import argparse
import soundfile as sf
import torch.nn.functional as F
import itertools as it
from fairseq import utils
from fairseq.models import BaseFairseqModel
from fairseq.data import Dictionary
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecEncoder, Wav2Vec2CtcConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

try:
    from flashlight.lib.text.dictionary import create_word_dict, load_words
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.decoder import (
        CriterionType,
        LexiconDecoderOptions,
        KenLM,
        LM,
        LMState,
        SmearingMode,
        Trie,
        LexiconDecoder,
    )
except:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object
    
    
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


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        #print(args)
        self.nbest = args['nbest']

        # criterion-specific init
        if args['criterion'] == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = (
                tgt_dict.index("<ctc_blank>")
                if "<ctc_blank>" in tgt_dict.indices
                else tgt_dict.bos()
            )
            if "<sep>" in tgt_dict.indices:
                self.silence = tgt_dict.index("<sep>")
            elif "|" in tgt_dict.indices:
                self.silence = tgt_dict.index("|")
            else:
                self.silence = tgt_dict.eos()
            self.asg_transitions = None
        elif args.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
            self.silence = -1
            self.asg_transitions = args.asg_transitions
            self.max_replabel = args.max_replabel
            assert len(self.asg_transitions) == self.vocab_size ** 2
        else:
            raise RuntimeError(f"unknown criterion: {args.criterion}")

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
        model = models ## change here
        encoder_out = model(**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            if hasattr(model, "get_logits"):
                emissions = model.get_logits(encoder_out) # no need to normalize emissions
            else:
                emissions = model.get_normalized_probs(encoder_out, log_probs=True)
        elif self.criterion_type == CriterionType.ASG:
            emissions = encoder_out["encoder_out"]
        return emissions.transpose(0, 1).float().cpu().contiguous()

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        elif self.criterion_type == CriterionType.ASG:
            idxs = filter(lambda x: x >= 0, idxs)
            idxs = unpack_replabels(list(idxs), self.tgt_dict, self.max_replabel)
        return torch.LongTensor(list(idxs))


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
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
            [{"tokens": self.get_tokens(viterbi_path[b].tolist()), "score": 0}]
            for b in range(B)
        ]


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        self.unit_lm = getattr(args, "unit_lm", False)

        if args['lexicon']:
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

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args['beam'],
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args['beam_threshold'],
                lm_weight=args['lm_weight'],
                word_score=args['word_score'],
                unk_score=args['unk_weight'],
                sil_score=args['sil_weight'],
                log_add=False,
                criterion_type=self.criterion_type,
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
                self.unit_lm,
            )
        else:
            assert args.unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {w: [[w]] for w in tgt_dict.symbols}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(getattr(args, "beam_size_token", len(tgt_dict))),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(
                self.decoder_opts, self.lm, self.silence, self.blank, []
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
        return hypos
    
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



def get_results(wav_path,dict_path,generator,use_cuda=False,w2v_path=None,model=None, half=None):
    sample = dict()
    net_input = dict()
    feature = get_feature(wav_path)
    target_dict = Dictionary.load(dict_path)
 
    model.eval()
           
    if half:
        net_input["source"] = feature.unsqueeze(0).half()
    else:
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


def load_model(model_path):
    return torch.load(model_path)#,map_location=torch.device("cuda"))



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

def parse_transcription(model_path, dict_path, wav_path, cuda, decoder="viterbi", lexicon_path=None, lm_path=None, half=None):
    target_dict = Dictionary.load(dict_path)
    args = get_args(lexicon_path, lm_path)
    
    if decoder=="viterbi":
        generator = W2lViterbiDecoder(args, target_dict)
    else:
        generator = W2lKenLMDecoder(args, target_dict)
    
    result = ''

    if cuda:
        model = load_model(model_path)
        model.cuda()
    else:
        model = load_model(model_path)
        
        
    if half:
        model.half()
        
    result = get_results(wav_path=wav_path, dict_path=dict_path, generator=generator, use_cuda=cuda, model=model, half=half)
    
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
    parser.add_argument('-H', '--half', default=False, type=bool, help="Half True or False")
    
    args_local = parser.parse_args()

    result = parse_transcription(args_local.model, args_local.dict, args_local.wav,  args_local.cuda, args_local.decoder, args_local.lexicon, args_local.lm_path, args_local.half)
    print(result)
