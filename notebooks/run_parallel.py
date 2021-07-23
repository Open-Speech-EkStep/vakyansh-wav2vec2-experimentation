import time
from tqdm import tqdm
from joblib import Parallel, delayed

def test_fn(indx, sleep_duration):
    time.sleep(sleep_duration)

#for i in tqdm(range(30)):
Parallel(n_jobs=-1)(delayed(test_fn)(i,1) for i in tqdm(range(30)))
