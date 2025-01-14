# validate_ubench_dataset.py

import argparse
import json
from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Validate the uBench dataset for data consistency.")
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
        help="Dataset split to validate. Default is 'test'.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="ubench_validation_report.jsonl",
        help="Path to the output file where validation issues will be saved.",
    )
    return parser.parse_args()

def is_valid_q_data(q_data, q_type):
    """
    Validate a single q_data entry.
    Returns (is_valid: bool, reason: str)
    """
    if q_data is None:
        return False, f"'q_data' for question type '{q_type}' is None"

    if not isinstance(q_data, dict):
        return False, f"'q_data' for question type '{q_type}' is not a dict"

    # Required fields
    required_fields = ['question', 'options', 'answer', 'answer_idx', 'id', 'name']
    for field in required_fields:
        if field not in q_data:
            return False, f"Missing '{field}' in q_data for question type '{q_type}'"

    # Validate 'options'
    options = q_data.get('options')
    if options is None:
        return False, f"'options' is None for question type '{q_type}'"
    if not isinstance(options, list):
        return False, f"'options' is not a list for question type '{q_type}'"
    if len(options) == 0:
        return False, f"'options' is empty for question type '{q_type}'"

    # Validate 'answer_idx'
    answer_idx = q_data.get('answer_idx')
    if not isinstance(answer_idx, int):
        return False, f"'answer_idx' is not an integer for question type '{q_type}'"
    if not (0 <= answer_idx < len(options)):
        return False, f"'answer_idx' ({answer_idx}) is out of bounds for 'options' in question type '{q_type}'"

    return True, ""

def main():
    args = parse_args()

    print(f"Loading dataset '{args.dataset_name}' split '{args.split}'...")
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Validating dataset entries...")
    total_items = len(dataset)
    total_questions = 0
    valid_questions = 0
    invalid_questions = 0
    buggy_items = []

    for idx, item in enumerate(tqdm(dataset, desc="Validating items")):
        questions = item.get('questions')
        if not questions:
            # Entire 'questions' field is missing or empty
            buggy_info = {
                "index": idx,
                "item_id": item.get('id', None),
                "reason": "Missing 'questions' field",
                "question_data": item.get('questions', None)
            }
            buggy_items.append(buggy_info)
            invalid_questions += 1
            continue

        for q_type, q_data in questions.items():
            total_questions += 1
            is_valid, reason = is_valid_q_data(q_data, q_type)
            if not is_valid:
                buggy_info = {
                    "index": idx,
                    "item_id": item.get('id', None),
                    "question_type": q_type,
                    "reason": reason,
                    "question_data": q_data
                }
                buggy_items.append(buggy_info)
                invalid_questions += 1
            else:
                valid_questions += 1

    print("\nValidation Complete!")
    print(f"Total items in dataset: {total_items}")
    print(f"Total questions checked: {total_questions}")
    print(f"Valid questions: {valid_questions}")
    print(f"Invalid/buggy questions: {invalid_questions}")

    if buggy_items:
        print(f"\nSaving details of buggy questions to '{args.output_file}'...")
        try:
            with open(args.output_file, 'w') as f:
                for buggy in buggy_items:
                    f.write(json.dumps(buggy) + '\n')
            print("Buggy questions saved successfully.")
        except Exception as e:
            print(f"Error writing to output file: {e}")
    else:
        print("No buggy questions found. Dataset is clean.")

if __name__ == "__main__":
    main()
