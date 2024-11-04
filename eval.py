# eval.py
import argparse
import json
from tqdm import tqdm
from .infer_eval import evaluate_predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                      choices=['vqa-rad', 'path-vqa', 'mdwiratathya/SLAKE-vqa-english'])
    parser.add_argument("--pred_file_parent_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load predictions
    predictions = []
    with open(args.pred_file_parent_path) as f:
        for line in tqdm(f, desc="Loading predictions"):
            predictions.append(json.loads(line))
    
    # Calculate and print results
    results = evaluate_predictions(predictions)
    print("\nResults:")
    print(f"Total examples: {results['total']}")
    if results['closed_ended']['total'] > 0:
        print(f"Closed-ended accuracy: {results['closed_ended']['accuracy']*100:.2f}")
    if results['open_ended']['total'] > 0:
        print(f"Open-ended F1 score: {results['open_ended']['f1']*100:.2f}")

if __name__ == '__main__':
    main()