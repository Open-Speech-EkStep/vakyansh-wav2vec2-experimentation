import gc
import itertools as it
import os.path as osp
import warnings
from collections import deque, namedtuple

from fairseq.data import Dictionary
import numpy as np
import torch
from examples.speech_recognition.data.replabels import unpack_replabels
from fairseq import tasks
from fairseq.utils import apply_to_sample
from omegaconf import open_dict
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import torch
from tqdm import tqdm
import sys
import pandas as pd


from wav2letter.common import create_word_dict, load_words
from wav2letter.criterion import CpuViterbiPath, get_data_ptr_as_bytes
from wav2letter.decoder import (
    CriterionType,
    DecoderOptions,
    KenLM,
    LM,
    LMState,
    SmearingMode,
    Trie,
    LexiconDecoder,
)

class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args['nbest']

        # criterion-specific init
        if args['criterion'] == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = (
                tgt_dict.index("<ctc_blank>")
                if "<ctc_blank>" in tgt_dict.indices
                else tgt_dict.bos()
            )
            self.asg_transitions = None
        elif args.criterion == "asg_loss":
            self.criterion_type = CriterionType.ASG
            self.blank = -1
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
        # encoder_out = models[0].encoder(**encoder_input)
        encoder_out = models[0](**encoder_input)
        if self.criterion_type == CriterionType.CTC:
            emissions = models[0].get_normalized_probs(encoder_out, log_probs=True)
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

class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

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
        return hypos

    

    
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

    
def get_decoder(decoder=None,BEAM=128,LM_WEIGHT=2,WORD_SCORE=-1):
  
    args_lm = {}
    args_lm['lexicon'] = '/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/data/training/lexicon.lst'
    args_lm['kenlm_model'] = '/home/harveen.chadha/_/english-asr-challenge/lm/lm.binary'
   
    args_lm['beam'] = BEAM
    args_lm['beam_threshold'] = 25
    args_lm['lm_weight'] = LM_WEIGHT
    args_lm['word_score'] = WORD_SCORE
    args_lm['unk_weight'] = -np.inf
    args_lm['sil_weight'] = 0
    args_lm['nbest'] = 1
    args_lm['criterion'] ='ctc'
    args_lm['labels']='ltr'
    
    if decoder=="kenlm":
        target_dict = Dictionary.load("/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/data/training/dict.ltr.txt")
        k = W2lKenLMDecoder(args_lm, target_dict)
    else:
        target_dict = Dictionary.load("/home/harveen.chadha/_/experiments/experiment_2/english-asr-challenge/data/training/dict.ltr.txt")
        k=W2lViterbiDecoder(args_lm, target_dict)
    return k


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


def get(i,decoder):
    return " ".join(decoder.decode(i)[0][0]["words"])


def do_processing(lm_weight, word_score, emmissions, output_folder_name, original_path):

    valid_iitm = []
    with open(original_path) as file:
        valid_iitm = file.readlines()

    valid_iitm = [ item.strip() for item in valid_iitm]

    list_results=[]
    decoder = get_decoder("kenlm",LM_WEIGHT=lm_weight,WORD_SCORE=word_score)

    for index, local_emmission in tqdm(enumerate(emmissions)):
            predicted = get(local_emmission, decoder)
            ground_truth = valid_iitm[index]
            list_results.append([predicted, ground_truth])
        
        
    df=pd.DataFrame(list_results)
    filename = str(lm_weight)+'_'+str(word_score)+'.csv'
    df.to_csv(output_folder_name + '/out_'+filename, index=False)
    
    
    
if __name__ == "__main__":
    lm_weight=sys.argv[1]
    word_score=sys.argv[2]

    do_processing(float(lm_weight), float(word_score))
