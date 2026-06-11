import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.meta import InsertCheapestHCARP
from time import time
from glob import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="InsertCheapestHCARP")
    
    # Add arguments
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--variant', type=str, default='P', help='Environment variant')
    parser.add_argument('--num_sample', type=int, default=1, help='num_sample')
    parser.add_argument('--path', type=str, default='/usr/local/rsa/ArcRoute/data/instances', help='path to instances')
    
    return parser.parse_args()

import concurrent.futures

def process_file(f, args):
    al = InsertCheapestHCARP()
    al.import_instance(f)
    t1 = time()
    res = al(variant=args.variant, num_sample=args.num_sample)
    return f, res, time() - t1

if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    files = sorted(glob(args.path + '/*/*.npz'))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(os.cpu_count() or 8, 8), mp_context=ctx) as pool:
        futures = [pool.submit(process_file, f, args) for f in files]
        for fut in concurrent.futures.as_completed(futures):
            f, res, dt = fut.result()
            print(f, ':::', res, ':::', dt)
