import argparse
import pickle

import torch
import numpy as np

from data import (
    ATTRIBUTES_ALL, 
    FAVORABILITIES_ALL, 
    GSS_OCCUPATIONS, 
    GSS_PRESTIGE_RATINGS,
    ATTRIBUTES_KATZ,
    SCORES_KATZ,
    ATTRIBUTES_GILBERT,
    SCORES_GILBERT,
    ATTRIBUTES_KARLINS,
    SCORES_KARLINS,
    ATTRIBUTES_BERGSIEKER,
    SCORES_BERGSIEKER
)

# Loads association scores stored in dictionary like in matched_guise_probing.py and 
# calculates the necessary metrics across different prompts

adjectives = ATTRIBUTES_ALL
favorabilities = FAVORABILITIES_ALL
roles = GSS_OCCUPATIONS
prestige = GSS_PRESTIGE_RATINGS
traits_32 = ATTRIBUTES_KATZ
score_32 = SCORES_KATZ
traits_50 = ATTRIBUTES_GILBERT
score_50 = SCORES_GILBERT
traits_67 = ATTRIBUTES_KARLINS
score_67 = SCORES_KARLINS
traits_12 = ATTRIBUTES_BERGSIEKER
score_12 = SCORES_BERGSIEKER

traits = [(traits_32, score_32, '1932'), (traits_50, score_50, '1950'), (traits_67, score_67, '1967'), (traits_12, score_12, '2012')]
# traits = [(traits_32, '1932'), (traits_50, '1950'), (traits_67, '1967'), (traits_12, '2012')]

adj_to_scores = {adjectives[i] : favorabilities[i] for i in range(len(adjectives))}
role_to_scores = {roles[i] : prestige[i] for i in range(len(roles))}

def main(args):
    task = args.task    
    with open(args.scores_path, "rb") as f:
        model_scores = pickle.load(f)

    if task == "association":
        scores_mean = mean_scores(model_scores)
        sorted_scores_mean = sorted(enumerate(scores_mean), key=lambda x: x[1])
        top_5 = sorted_scores_mean[-5:]
        bottom_5 = sorted_scores_mean[:5]
        top_5_fav_score = 0
        top_5_denom = 0
        bottom_5_fav_score = 0
        bottom_5_denom = 0

        print("Top 5 adjectives (More association with AAE):")
        for idx, score in reversed(top_5):
            print(f"{adjectives[idx]}: {score:.4f}")
            top_5_fav_score += adj_to_scores[adjectives[idx]] * score
            top_5_denom += score
        print("\nBottom 5 adjectives (More association with SAE):")
        for idx, score in bottom_5:
            print(f"{adjectives[idx]}: {score:.4f}")
            bottom_5_fav_score += adj_to_scores[adjectives[idx]] * score
            bottom_5_denom += score
        
        top_5_fav_score /= top_5_denom
        bottom_5_fav_score /= bottom_5_denom
        print(f"Average favorability of top 5 adjectives: {top_5_fav_score:.4f}")
        print(f"Average favorability of bottom 5 adjectives: {bottom_5_fav_score:.4f}")

        for trait, _, year in traits:
            fav_score = sum([adj_to_scores[adj] for adj in trait]) / len(trait)
            print(f"Average favorability of top 5 adjectives in {year}: {fav_score:.4f}")

    if task == "agreement":
        agreements = {}
        for prompt in model_scores:
            prompt_scores = model_scores[prompt]
            sorted_scores = sorted(enumerate(prompt_scores), key=lambda x: x[1], reverse=True)
            sorted_adjectives = [adjectives[idx] for idx, _ in sorted_scores]
            for trait, scores, year in traits:
                agreements[year] = []
                for i in range(1, 6):
                    subset = get_top_n_traits(trait, scores, i)
                    agreements[year].append(average_precision(sorted_scores, subset))
                agreements[year] = np.mean(agreements[year])

        for year, agreement in agreements.items():
            print(f"Agreement for {year}: {agreement:.4f}")
        
    if task == "employment":
        scores_mean = mean_scores(scores)
        sorted_scores_mean = sorted(enumerate(scores_mean), key=lambda x: x[1])
        role_rankings = [roles[idx] for idx, _ in sorted_scores_mean]
        prestige_rankings = [role_to_scores[role] for role in role_rankings]
        pass
        # Todo, save to scp to make nice plots of rankings

    if task == "conviction":
        pass

    if task == "life_death":
        pass

def average_precision(traits, traits_true):
    def precision(traits_pred, traits_true):
        traits1 = set(traits1)
        traits2 = set(traits2)
        intersection = len(traits1 & traits2)
        return intersection / len(traits1)
    
    precisions = []
    for i in range(len(traits)):
        if traits[i] in traits_true:
            precisions.append(precision(traits[:i+1], traits_true))

    return sum(precisions) / len(traits_true)

def get_top_n_traits(traits, scores, n=5):
    top_n_idx = np.argsort(scores)[-n:]
    top_n_traits = [traits[idx] for idx in top_n_idx]  
    return top_n_traits

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
