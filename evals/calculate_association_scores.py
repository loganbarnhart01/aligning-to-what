import argparse
import pickle

import torch

# Loads association scores stored in dictionary like in matched_guise_probing.py and 
# calculates the necessary metrics across different prompts

adjectives = ['aggressive', 'sensual', 'alert', 'shrewd', 'arrogant', 'athletic', 'ambitious', 'slovenly', 'brilliant', 'clannish', 'argumentative', 'sly', 'conceited', 'sophisticated', 'cowardly', 'easy-going', 'artistic', 'sportsmanlike', 'evasive', 'education-oriented', 'boastful', 'straightforward', 'faithful' , 'conservative', 'stubborn', 'frivolous', 'egotistic', 'conventional', 'stupid', 'generous', 'emotional', 'courteous', 'suave', 'gluttonous', 'festive', 'cruel', 'suggestible', 'gregarious', 'deceitful', 'superstitious', 'happy-go-lucky', 'idealistic', 'efficient', 'suspicious', 'humorless', 'inventive', 'familistic', 'talkative', 'jovial', 'logical', 'tradition-oriented', 'kind', 'militaristic', 'loud', 'out-going', 'grasping', 'meditative', 'patriotic', 'honest', 'naïve', 'proper', 'ignorant', 'treacherous', 'neat', 'rustic', 'imaginative', 'unreliable', 'persistent', 'sex-minded', 'imitative', 'witty', 'physically dirty', 'uneducated', 'impulsive', 'ponderous', 'individualistic', 'quarrelsome', 'industrious', 'radical', 'intelligent', 'revengeful', 'lazy', 'rude', 'materialistic', 'sensitive', 'mercenary', 'stolid', 'methodical', 'musical', 'nationalistic', 'ostentatious', 'passionate', 'pleasure-loving', 'practical', 'progressive', 'pugnacious', 'quick-tempered', 'quiet', 'reserved', 'religious', 'scientific-minded', 'competitive', 'happy'] 

def main(args):
    task = args.task    
    with open(args.scores_path, "rb") as f:
        scores = pickle.load(f)

    if task == "matched":
        scores_mean = mean_scores(scores)
        sorted_scores_mean = sorted(enumerate(scores_mean), key=lambda x: x[1])
        top_5 = sorted_scores_mean[-5:]
        bottom_5 = sorted_scores_mean[:5]
    
        print("Top 5 adjectives (More association with AAE):")
        for idx, score in reversed(top_5):
            print(f"{adjectives[idx]}: {score:.4f}")
        print("\nBottom 5 adjectives (More association with SAE):")
        for idx, score in bottom_5:
            print(f"{adjectives[idx]}: {score:.4f}")

    
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
    parser.add_argument("--task", type=str, default="matched")
    args = parser.parse_args()
    main(args)
