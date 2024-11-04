# infer_eval.py
import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
from .eval_metrics import calculate_f1score
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, 
                      choices=['vqa-rad', 'path-vqa', 'slake'])
    parser.add_argument("--model_id_or_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--min_pixels", type=int, default=28 * 28,
                       help="Minimum number of pixels for image processing")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28,
                       help="Maximum number of pixels for image processing")
    return parser.parse_args()

def is_yes_no_question(answer):
    return answer.lower() in ['yes', 'no']

def evaluate_predictions(predictions):
    total = len(predictions)
    closed_ended_correct = 0
    closed_ended_total = 0
    open_ended_f1_sum = 0
    open_ended_total = 0
    
    for pred in predictions:
        gt_answer = pred['answer'].lower()
        pred_answer = pred['prediction'].lower()
        
        if is_yes_no_question(gt_answer):
            closed_ended_total += 1
            # Check if 'yes' exists in prediction when ground truth is 'yes'
            # and same for 'no'
            if (gt_answer == 'yes' and 'yes' in pred_answer) or \
               (gt_answer == 'no' and 'no' in pred_answer):
                closed_ended_correct += 1
        else:
            open_ended_total += 1
            f1, _, _ = calculate_f1score(pred_answer, gt_answer)
            open_ended_f1_sum += f1

    results = {
        'total': total,
        'closed_ended': {
            'accuracy': closed_ended_correct/closed_ended_total if closed_ended_total > 0 else 0,
            'total': closed_ended_total
        },
        'open_ended': {
            'f1': open_ended_f1_sum/open_ended_total if open_ended_total > 0 else 0,
            'total': open_ended_total
        }
    }
    return results

def process_batch(llm, batch_items):
    # Prepare batch inputs for vLLM
    inputs = []
    for item in batch_items:
        # Convert image to RGB if it isn't already
        image = item['image'].convert('RGB') if item['image'].mode != 'RGB' else item['image']
        
        prompt = ("<|im_start|>system\nYou are a biomedical expert.<|im_end|>\n"
                 "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                 f"{item['question']}<|im_end|>\n"
                 "<|im_start|>assistant\n")
        
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image  # Now using RGB image
            }
        })

    # Generate responses using vLLM
    sampling_params = SamplingParams(temperature=0.2, max_tokens=128)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    batch_predictions = []
    for i, output in enumerate(outputs):
        item = batch_items[i]
        batch_predictions.append({
            'question': item['question'],
            'answer': item['answer'],
            'prediction': output.outputs[0].text  # ✅ Properly formatted prediction
        })
    return batch_predictions

def write_results(predictions, output_file):
    with open(output_file, 'a') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

def custom_collate(batch):
    """Custom collate function that preserves PIL Images"""
    return batch

def main():
    args = parse_args()
    
    # Load dataset
    if args.dataset_name == 'vqa-rad':
        dataset = load_dataset('flaviagiammarino/vqa-rad', split='test')
    elif args.dataset_name == 'path-vqa':
        dataset = load_dataset('flaviagiammarino/path-vqa', split='test')
    else:  # SLAKE
        dataset = load_dataset('mdwiratathya/SLAKE-vqa-english', split='test')

    # Initialize vLLM
    llm = LLM(
        model=args.model_id_or_path,
        max_model_len=4096,
        max_num_seqs=32,  # Adjust based on GPU memory
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
    )

    # Run batch inference
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate
    )
    predictions = []
    for batch in tqdm(dataloader, desc="Running inference"):
        responses = process_batch(llm, batch)
        predictions.extend(responses)
        # Write results as we go
        write_results(predictions, args.output_file)

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