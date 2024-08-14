import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

def main(args):
    with open(args.rewards_path, "rb") as f:
        scores = pickle.load(f)

    means = []
    stdvs = []
    for checkpoint in scores:
        checkpoint_scores = [x[0] for x in scores[checkpoint]]
        means.append(np.mean(checkpoint_scores))
        stdvs.append(np.std(checkpoint_scores))

    for i in range(len(means)):
        print(f'Checkpoint: {list(scores.keys())[i]}, Mean Reward: {means[i]:.4f}, Std: {stdvs[i]:.4f}')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewards_path", type=str)
    args = parser.parse_args()
    main(args)