import sox
import glob
import numpy as np
import sys
from tqdm import tqdm
import os
from joblib import Parallel, delayed



def get_duration_single_file(local_file):
    duration = sox.file_info.duration(local_file)
    clean_duration = 0    
    if 'clean' in local_file.split('/'):
        clean_duration = duration

    return [duration, clean_duration]

def get_duration(path_local):
    if not os.path.exists(path_local):
        raise Exception("Sorry this path doesn't exists")
    files = glob.glob(path_local + '/**/*.wav', recursive=True)
    print("Number of files present: ", len(files))
    total_seconds = []

    total_seconds.extend( Parallel(n_jobs=24)( delayed(get_duration_single_file)(local_file) for local_file in tqdm(files)))
    total_ = np.sum(total_seconds, axis=0)
    #print(total_)
    print("Total number of data in hours: ", total_[0]/3600)
    #print("Total number of clean data in hours: ", total_[1]/3600)

if __name__ == "__main__":
    path = sys.argv[1]
    get_duration(path)
