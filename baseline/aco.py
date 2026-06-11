import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline.meta import ACOHCARP
from time import time
from glob import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ACOHCARP")
    
    # Add arguments
    parser.add_argument('--seed', type=int, default=6868, help='Random seed')
    parser.add_argument('--max_epoch', type=int, default=100, help='num epoch')
    parser.add_argument('--variant', type=str, default='U', help='Environment variant')
    parser.add_argument('--n_ant', type=int, default=50, help='num epoch')
    parser.add_argument('--path', type=str, default='data/5m60', help='path to instances')
    
    return parser.parse_args()

import concurrent.futures

def process_file(f, args):
    al = ACOHCARP(n_ant=args.n_ant)
    al.import_instance(f)
    t1 = time()
    res = al(n_epoch=args.max_epoch, variant=args.variant)
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