import argparse
import pickle

import torch

from data import (
    ATTRIBUTES_ALL, 
    FAVORABILITIES_ALL, 
    GSS_OCCUPATIONS, 
    GSS_PRESTIGE_RATINGS,
    ATTRIBUTES_KATZ,
    ATTRIBUTES_GILBERT,
    ATTRIBUTES_KARLINS,
    ATTRIBUTES_BERGSIEKER
)

# Loads association scores stored in dictionary like in matched_guise_probing.py and 
# calculates the necessary metrics across different prompts

adjectives = ATTRIBUTES_ALL
favorabilities = FAVORABILITIES_ALL
roles = GSS_OCCUPATIONS
prestige = GSS_PRESTIGE_RATINGS
ranking_32 = ATTRIBUTES_KATZ[:5]
ranking_50 = ATTRIBUTES_GILBERT[:5]
ranking_67 = ATTRIBUTES_KARLINS[:5]
ranking_12 = ATTRIBUTES_BERGSIEKER[:5]

rankings = [(ranking_32, '1932'), (ranking_50, '1950'), (ranking_67, '1967'), (ranking_12, '2012')]

adj_to_scores = {adjectives[i] : favorabilities[i] for i in range(len(adjectives))}
role_to_scores = {roles[i] : prestige[i] for i in range(len(roles))}

def main(args):
    task = args.task    
    with open(args.scores_path, "rb") as f:
        scores = pickle.load(f)

    if task == "association":
        scores_mean = mean_scores(scores)
        sorted_scores_mean = sorted(enumerate(scores_mean), key=lambda x: x[1])
        top_5 = sorted_scores_mean[-5:]
        bottom_5 = sorted_scores_mean[:5]
        top_5_fav_score = 0
        bottom_5_fav_score = 0

        print("Top 5 adjectives (More association with AAE):")
        for idx, score in reversed(top_5):
            print(f"{adjectives[idx]}: {score:.4f}")
            top_5_fav_score += adj_to_scores[adjectives[idx]]
        print("\nBottom 5 adjectives (More association with SAE):")
        for idx, score in bottom_5:
            print(f"{adjectives[idx]}: {score:.4f}")
            bottom_5_fav_score += adj_to_scores[adjectives[idx]]
        
        top_5_fav_score /= 5
        bottom_5_fav_score /= 5
        print("Average favorability of top 5 adjectives: {top_5_fav_score:.4f}")
        print("Average favorability of bottom 5 adjectives: {bottom_5_fav_score:.4f}")

        for ranking, year in rankings:
            fav_score = sum([adj_to_scores[adj] for adj in ranking]) / len(ranking)
            print(f"Average favorability of top 5 adjectives in {year}: {top_5_fav_score:.4f}")

    if task == "agreement":
        pass

    if task == "prestige":
        pass

    if task == "conviction":
        pass

    if task == "life_death":
        pass

def average_precision(ranking, ranking_true):
    
    def precision(ranking_pred, ranking_true):
        ranking1 = set(ranking1)
        ranking2 = set(ranking2)
        intersection = sum(ranking1 & ranking2)
        return intersection / len(ranking1)
    
    precisions = []
    for i in range(len(ranking)):
        if ranking[i] in ranking_true:
            precisions.append(precision(ranking[:i+1], ranking_true))

    return sum(precisions) / len(ranking_true)

    



def mean_scores(scores_dict, sorted=False):
    """
    Calculate the mean score for one adjective (or job) across all prompts.
    Args:
        scores_dict (dict): Dictionary with scores for each adjective across prompts
            {prompt : list_of_scores}
        sorted (bool): Whether to sort the scores
    Returns:
        scores_mean (torch.Tensor): Mean score for each adjective across prompts
    """
    scores = {k: torch.tensor(v) for k, v in scores_dict.items()}
    scores_mean = [scores[key] for key in scores]
    scores_mean = torch.stack(scores_mean).mean(0)
    return scores_mean
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-path", type=str, required=True) 
    parser.add_argument("--task", type=str, default="association")
    args = parser.parse_args()
    main(args)
