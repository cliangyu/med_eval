# eval.py
import argparse
import json
from tqdm import tqdm
from .infer_eval import evaluate_predictions, evaluate_mcq_predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True,
                      choices=['vqa-rad', 'path-vqa', 'slake', 'ubench'])
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
    if args.dataset_name == 'ubench':
        results = evaluate_mcq_predictions(predictions)
        print("\nResults:")
        print(f"Total examples: {results['total']}")
        print(f"Overall accuracy: {results['overall_accuracy']*100:.2f}%")
        print("\nResults by question type:")
        for qtype, metrics in results['by_type'].items():
            print(f"{qtype}: {metrics['accuracy']*100:.2f}% ({metrics['total']} questions)")
    else:
        results = evaluate_predictions(predictions)
        print("\nResults:")
        print(f"Total examples: {results['total']}")
        if results['closed_ended']['total'] > 0:
            print(f"Closed-ended accuracy: {results['closed_ended']['accuracy']*100:.2f}%")
        if results['open_ended']['total'] > 0:
            print(f"Open-ended F1 score: {results['open_ended']['f1']*100:.2f}%")
            print(f"Open-ended Recall: {results['open_ended']['recall']*100:.2f}%")

if __name__ == '__main__':
    main()