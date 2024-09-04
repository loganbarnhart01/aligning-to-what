import pickle
import argparse

import numpy as np
# import matplotlib.pyplot as plt

def main(args):
    # checkpoints = [f"checkpoint-{i}" for i in range(500, 16500, 500)] # + ['checkpoint-16437']
    # checkpoints = checkpoints[::2] + ['checkpoint-16437'] # skipping some checkpoints for now to save time
    # rewards_path = args.rewards_path
    # for checkpoint in checkpoints:
    with open(args.rewards_path, "rb") as f:
        scores = pickle.load(f)
    print(f"Mean: {np.mean(scores)}")
    print(f"Std. Dev.: {np.std(scores)}")
    print('\n')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards_path", type=str)
    args = parser.parse_args()
    main(args)