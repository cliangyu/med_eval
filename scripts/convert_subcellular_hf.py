from PIL import Image
import json
from tqdm import tqdm
import numpy as np
import os
import tempfile
from multiprocessing import Pool, cpu_count
from functools import partial

# Create a temporary directory in current working directory
temp_dir = "/pasteur2/u/liangyuc/code/med_eval/temp_hf_cache"
os.makedirs(temp_dir, exist_ok=True)
tempfile.tempdir = temp_dir
from datasets import Dataset, Features, Image as ImageFeature

def load_single_image(item):
    """Load a single image and return the updated item"""
    try:
        img = Image.open(item['file_path'])
        item['image'] = img
    except Exception as e:
        print(f"Error loading image {item['file_path']}: {e}")
        item['image'] = None
    return item

def load_and_prepare_data():
    # Read JSON file
    with open('captions.json', 'r') as f:
        data = json.load(f)
    
    # Calculate optimal number of processes
    num_processes = min(cpu_count(), 16)  # limit to 16 processes max
    print(f"Loading images using {num_processes} processes...")
    
    # Use multiprocessing to load images
    with Pool(processes=num_processes) as pool:
        # Process images in parallel with progress bar
        data = list(tqdm(
            pool.imap(load_single_image, data),
            total=len(data),
            desc="Loading images"
        ))
    
    features = Features({
        'image_id': 'string',
        'ensembl_id': 'string',
        'protein': 'string',
        'uniprot_id': 'string',
        'cell_line': 'string',
        'cellosaurus_id': 'string',
        'localizations': 'string',
        'url': 'string',
        'caption': 'string',
        'file_path': 'string',
        'md5_hash': 'string',
        'image': ImageFeature()
    })
    
    # Randomly shuffle the data
    np.random.seed(42)
    np.random.shuffle(data)
    
    # Split into train and test
    test_size = 1000
    test_data = data[:test_size]
    train_data = data[test_size:]
    
    print("Creating datasets...")
    train_dataset = Dataset.from_list(train_data, features=features)
    test_dataset = Dataset.from_list(test_data, features=features)
    
    return train_dataset, test_dataset

# Create datasets
train_dataset, test_dataset = load_and_prepare_data()

# Push both splits to hub
print("Pushing to Hugging Face Hub...")
train_dataset.push_to_hub(
    "liangyuch/subcellular", 
    split="train", 
    num_proc=4,
    cache_dir=temp_dir
)
test_dataset.push_to_hub(
    "liangyuch/subcellular", 
    split="test", 
    num_proc=4,
    cache_dir=temp_dir
)

# Clean up
import shutil
shutil.rmtree(temp_dir)