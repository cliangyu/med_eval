# find_buggy_ubench_questions.py

import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Find buggy questions in the uBench dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jnirschl/uBench",
        help="Name of the uBench dataset to load. Default is 'jnirschl/uBench'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to check. Default is 'test'.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="buggy_ubench_questions.jsonl",
        help="Path to the output file where buggy questions will be saved.",
    )
    return parser.parse_args()

def is_buggy_item(item):
    """
    Check if a dataset item is buggy based on the 'options' field.
    Returns True if buggy, False otherwise.
    """
    # Assuming each item has a 'questions' field which is a dictionary
    questions = item.get('questions')
    if not questions:
        return True, "Missing 'questions' field"

    for q_type, q_data in questions.items():
        if q_data is None:
            return True, f"'q_data' for question type '{q_type}' is None"

        if not isinstance(q_data, dict):
            return True, f"'q_data' for question type '{q_type}' is not a dict"

        options = q_data.get('options')
        if options is None:
            return True, f"'options' is None for question type '{q_type}'"
        if not isinstance(options, list):
            return True, f"'options' is not a list for question type '{q_type}'"
        if len(options) == 0:
            return True, f"'options' is empty for question type '{q_type}'"
    return False, ""

def main():
    args = parse_args()

    print(f"Loading dataset '{args.dataset_name}' split '{args.split}'...")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Scanning for buggy questions...")
    buggy_items = []
    for idx, item in enumerate(tqdm(dataset, desc="Checking items")):
        is_buggy, reason = is_buggy_item(item)
        if is_buggy:
            buggy_info = {
                "index": idx,
                "item_id": item.get('id', None),
                "reason": reason,
                "question_data": item.get('questions', None)
            }
            buggy_items.append(buggy_info)

    print(f"Found {len(buggy_items)} buggy items out of {len(dataset)} total items.")

    if buggy_items:
        print(f"Saving buggy items to '{args.output_file}'...")
        try:
            with open(args.output_file, 'w') as f:
                for buggy in buggy_items:
                    f.write(json.dumps(buggy) + '\n')
            print("Buggy items saved successfully.")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    else:
        print("No buggy items found.")

if __name__ == "__main__":
        main()
