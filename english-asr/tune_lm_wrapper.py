import numpy as np
import sys
from tune_lm import *
import os
import time
import numpy as np
from multiprocessing import Pool
from multiprocessing import Process
from joblib import Parallel, delayed
from tqdm import tqdm


if __name__ == "__main__":
    
    lm_weight_values = [1,3.1]
    word_score_values = [-2,0.1]
    
    # lm_weight_values = [3,4.1]
    # word_score_values = [-3.1,-2]
    
    lm_weight_range = np.around(np.arange(lm_weight_values[0], lm_weight_values[1], 0.1) , decimals = 2)
    word_score_range = np.around(np.arange(word_score_values[0], word_score_values[1], 0.1), decimals = 2)
    
    # lm_weight_range=[2]
    # word_score_range=[-1]
    combinations = []

    for i in lm_weight_range:
        for j in word_score_range:
            
            combinations.append([i,j])
    combinations.reverse()
    
    emmission_list = ['/home/harveen.chadha/_/experiments/experiment_6/english-asr-challenge/scripts/inference/dev_IITM.npy',
    '/home/harveen.chadha/_/experiments/experiment_6/english-asr-challenge/scripts/inference/dev_NPTEL.npy'
    ]

    ground_truth_list=['/home/harveen.chadha/_/experiments/experiment_6/english-asr-challenge/data/dev/dev_IITM/valid.wrd',
    '/home/harveen.chadha/_/experiments/experiment_6/english-asr-challenge/data/dev/dev_NPTEL/valid.wrd'
    ]

    for em_path in emmission_list:
        emmissions = np.load(em_path, allow_pickle=True)
        
        emmissions = [torch.from_numpy(e) for e in emmissions]
        emmissions = [e.unsqueeze(0) for e in emmissions]

        folder_type = em_path.split('/')[-1].split('.')[0]
        folder_name = './experiment_6/'+folder_type
        os.makedirs(folder_name, exist_ok=True)

        ground_truth = ''
        if folder_type=='dev_NPTEL':
            ground_truth=ground_truth_list[1]
        else:
            ground_truth=ground_truth_list[0]

        Parallel(n_jobs=-1)(delayed(do_processing)(float(combination[0]), float(combination[1]), emmissions, folder_name, ground_truth) for index, combination in tqdm(enumerate(combinations)))
