import argparse
import json
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
import re
from collections import defaultdict
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id_or_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path. If not specified, will use model_id_or_path directory")
    parser.add_argument("--test_file", type=str, default="/home/ly/d/data/subcellular/subcellular_test.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    return parser.parse_args()

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_cell_line_and_localizations(text, all_cell_lines, all_localizations):
    """Extract cell line and localizations from model output using known values"""
    # Create regex patterns from the actual values
    cell_line_pattern = '|'.join(map(re.escape, all_cell_lines))
    matches = re.findall(f"(?i)({cell_line_pattern})", text)
    predicted_cell_line = matches[0].upper() if matches else ""

    predicted_locs = []
    for loc in all_localizations:
        if re.search(f"(?i){re.escape(loc)}", text):
            predicted_locs.append(loc)
    
    return predicted_cell_line, predicted_locs

def process_batch(llm, batch_items, all_cell_lines, all_localizations, model_id_or_path):
    inputs = []
    for item in batch_items:
        # Load and convert image to RGB
        try:
            image = Image.open(item['images']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {item['images']}: {e}")
            continue
        
        # Choose prompt based on model name
        if "llava" in model_id_or_path.lower():
            prompt = ("<|im_start|>user <image>\n"
                     "Describe the cell line and subcellular localizations shown in this image."
                     "<|im_end|><|im_start|>assistant\n")
        else:
            prompt = ("<|im_start|>system\nYou are a biomedical expert.<|im_end|>\n"
                     "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                     "Describe the cell line and subcellular localizations shown in this image.<|im_end|>\n"
                     "<|im_start|>assistant\n")
        
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        })

    sampling_params = SamplingParams(temperature=0.2, max_tokens=128)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    batch_predictions = []
    for i, output in enumerate(outputs):
        item = batch_items[i]
        prediction_text = output.outputs[0].text
        predicted_cell_line, predicted_locs = extract_cell_line_and_localizations(
            prediction_text, all_cell_lines, all_localizations
        )
        
        batch_predictions.append({
            'true_cell_line': item['cell_line'],
            'true_localizations': item['localizations'],
            'predicted_cell_line': predicted_cell_line,
            'predicted_localizations': predicted_locs,
            'raw_prediction': prediction_text
        })
    return batch_predictions


def calculate_recalls(predictions):
    # Cell line recall
    cell_line_correct = 0
    total = len(predictions)
    
    # Localizations recall per category
    loc_recalls = defaultdict(int)
    loc_counts = defaultdict(int)
    
    for pred in predictions:
        # Cell line evaluation
        if pred['predicted_cell_line'].upper() == pred['true_cell_line'].upper():
            cell_line_correct += 1
            
        # Localizations evaluation
        true_locs = set(pred['true_localizations'])
        pred_locs = set(pred['predicted_localizations'])
        
        for loc in true_locs:
            loc_counts[loc] += 1
            if loc in pred_locs:
                loc_recalls[loc] += 1
    
    # Calculate final metrics
    cell_line_recall = cell_line_correct / total if total > 0 else 0
    
    # Calculate macro-averaged recall for localizations
    macro_recall = 0
    if loc_counts:
        individual_recalls = [
            loc_recalls[loc] / loc_counts[loc] 
            for loc in loc_counts.keys()
        ]
        macro_recall = sum(individual_recalls) / len(individual_recalls)
    
    return {
        'cell_line_recall': cell_line_recall,
        'localization_macro_recall': macro_recall,
        'total_samples': total,
        'per_location_recalls': {
            loc: loc_recalls[loc] / loc_counts[loc] 
            for loc in loc_counts.keys()
        }
    }

def main():
    args = parse_args()
    
    # Set default output file path if not specified
    if args.output_file is None:
        import os
        base_dir = os.path.dirname(args.model_id_or_path.rstrip('/'))
        filename = f"subcellular_{args.max_pixels}.jsonl"
        args.output_file = os.path.join(args.model_id_or_path, filename)
    
    # Load test data
    print("Loading test data...")
    test_data = load_jsonl(args.test_file)
    
    # Extract all possible cell lines and localizations from the data
    all_cell_lines = set()
    all_localizations = set()
    for item in test_data:
        all_cell_lines.add(item['cell_line'])
        all_localizations.update(item['localizations'])
    
    # Convert to Dataset
    dataset = Dataset.from_list(test_data)
    
    # Initialize vLLM
    llm = LLM(
        model=args.model_id_or_path,
        max_model_len=16384,
        max_num_seqs=32,
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
    )

    # Initialize output file
    with open(args.output_file, 'w') as f:
        pass
    
    # Run batch inference
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=lambda x: x
    )
    
    predictions = []
    for batch in tqdm(dataloader, desc="Running inference"):
        responses = process_batch(llm, batch, all_cell_lines, all_localizations, args.model_id_or_path)
        predictions.extend(responses)
        # Write batch results
        with open(args.output_file, 'a') as f:
            for pred in responses:
                f.write(json.dumps(pred) + '\n')

    # Calculate and print results
    results = calculate_recalls(predictions)
    print("\nResults:")
    print(f"Total examples: {results['total_samples']}")
    print(f"Cell line recall: {results['cell_line_recall']*100:.2f}%")
    print(f"Localization macro-averaged recall: {results['localization_macro_recall']*100:.2f}%")
    print("\nPer-location recalls:")
    for loc, recall in results['per_location_recalls'].items():
        print(f"{loc}: {recall*100:.2f}%")

if __name__ == '__main__':
    main()