import json
import random
import os
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Read the input JSON data
print("Reading JSON data...")
with open('/pasteur/data/jnirschl/datasets/hpa/data/processed/subcellular/captions.json', 'r') as f:
    data = json.load(f)

# Prepare the JSONL format entries
print("Converting to JSONL format...")
jsonl_entries = []
for item in tqdm(data, desc="Processing entries"):
    image_path = os.path.join('/pasteur/data/jnirschl/datasets/hpa/data/processed/subcellular', item['file_path'])
    # Split localizations into a list if it contains multiple locations
    localizations_list = [loc.strip() for loc in item['localizations'].split(';')]
    entry = {
        "query": "Provide a description of the given image.",
        "response": item['caption'],
        "images": image_path,
        "cell_line": item['cell_line'],
        "localizations": localizations_list
    }
    jsonl_entries.append(entry)

# Randomly shuffle the data
print("Shuffling data...")
random.shuffle(jsonl_entries)

# Split into train and test sets (2000 for test)
test_size = 2000
test_set = jsonl_entries[:test_size]
train_set = jsonl_entries[test_size:]

# Write train and test files
def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for entry in tqdm(data, desc=f"Writing {filename}"):
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

write_jsonl('subcellular_train.jsonl', train_set)
write_jsonl('subcellular_test.jsonl', test_set)

# Print some statistics
print(f"\nTotal entries: {len(jsonl_entries)}")
print(f"Train set size: {len(train_set)}")
print(f"Test set size: {len(test_set)}")
