import time
from tqdm import tqdm

def test_fn(indx, sleep_duration):
    time.sleep(sleep_duration)

for i in tqdm(range(30)):
    test_fn(i,1)
