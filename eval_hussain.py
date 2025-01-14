import argparse
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
import os
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id_or_path", type=str, required=True)
    parser.add_argument("--hussain_file", type=str, 
                       default="/home/ly/d/data/hussain_et_al_2019/test_200_qwen.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=28 * 28)
    parser.add_argument("--max_pixels", type=int, default=12_000_000)
    return parser.parse_args()

def load_hussain_data(file_path):
    """Load questions, responses, and images from Hussain JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Load image directly from the path
            try:
                image = Image.open(item['images'])
                data.append({
                    'image': image,
                    'query': item['query'],
                    'response': item['response'],
                    'image_path': item['images']
                })
            except Exception as e:
                print(f"Error loading image {item['images']}: {e}")
    return data

def custom_collate(batch):
    """Custom collate function that preserves PIL Images"""
    return batch

def process_batch(llm, batch_items):
    inputs = []
    for item in batch_items:
        image = item['image'].convert('RGB') if item['image'].mode != 'RGB' else item['image']
        
        prompt = ("<|im_start|>system\nYou are a biomedical expert.<|im_end|>\n"
                 "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                 f"{item['query']}<|im_end|>\n"
                 "<|im_start|>assistant\n")
        
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            }
        })

    sampling_params = SamplingParams(temperature=0.2, max_tokens=128)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    batch_predictions = []
    for i, output in enumerate(outputs):
        item = batch_items[i]
        batch_predictions.append({
            'query': item['query'],
            'response': item['response'],
            'prediction': output.outputs[0].text,
            'image_path': item['image_path']
        })
    return batch_predictions

def write_results(predictions, output_file, mode='a'):
    with open(output_file, mode) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

def evaluate_predictions(predictions):
    total = len(predictions)
    correct = 0
    
    for pred in predictions:
        gt_answer = pred['response'].lower()
        pred_answer = pred['prediction'].lower()
        
        if gt_answer in pred_answer:
            correct += 1
    
    return {
        'total': total,
        'accuracy': correct/total if total > 0 else 0
    }

def main():
    args = parse_args()
    
    # Set default output file if not specified
    if args.output_file is None:
        base_dir = os.path.dirname(args.model_id_or_path.rstrip('/'))
        args.output_file = os.path.join(base_dir, "hussain_results.jsonl")
    
    # Load Hussain data
    hussain_data = load_hussain_data(args.hussain_file)
    
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
    write_results([], args.output_file, mode='w')
    
    # Create dataloader
    dataloader = DataLoader(
        hussain_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )
    
    # Run inference
    predictions = []
    for batch in tqdm(dataloader, desc="Running inference"):
        batch_predictions = process_batch(llm, batch)
        predictions.extend(batch_predictions)
        write_results(batch_predictions, args.output_file)
    
    # Calculate and print results
    results = evaluate_predictions(predictions)
    print("\nHussain Test Results:")
    print(f"Total examples: {results['total']}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")

if __name__ == '__main__':
    main()