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
                      choices=['vqa-rad', 'path-vqa', 'slake', 'ubench'])
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
    open_ended_recall_sum = 0
    open_ended_total = 0
    
    for pred in predictions:
        gt_answer = pred['answer'].lower()
        pred_answer = pred['prediction'].lower()
        
        if is_yes_no_question(gt_answer):
            closed_ended_total += 1
            if (gt_answer == 'yes' and 'yes' in pred_answer) or \
               (gt_answer == 'no' and 'no' in pred_answer):
                closed_ended_correct += 1
        else:
            open_ended_total += 1
            f1, _, recall = calculate_f1score(pred_answer, gt_answer)
            open_ended_f1_sum += f1
            open_ended_recall_sum += recall

    results = {
        'total': total,
        'closed_ended': {
            'accuracy': closed_ended_correct/closed_ended_total if closed_ended_total > 0 else 0,
            'total': closed_ended_total
        },
        'open_ended': {
            'f1': open_ended_f1_sum/open_ended_total if open_ended_total > 0 else 0,
            'recall': open_ended_recall_sum/open_ended_total if open_ended_total > 0 else 0,
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
            'prediction': output.outputs[0].text
        })
    return batch_predictions

def write_results(predictions, output_file, mode='a'):
    """Write results to file with specified mode (append or write)"""
    with open(output_file, mode) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

def custom_collate(batch):
    """Custom collate function that preserves PIL Images"""
    return batch

def format_mcq_options(options):
    """Add alphabetical prefixes to options"""
    return [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]

def get_correct_letter(answer_idx):
    """Convert numerical index to letter answer"""
    return chr(65 + answer_idx)

def process_mcq_batch(llm, batch_items):
    """Process a batch of MCQ items from uBench dataset"""
    inputs = []
    metadata_list = []
    
    for item in batch_items:
        image = item['image'].convert('RGB') if item['image'].mode != 'RGB' else item['image']
        questions = item['questions']
        
        for q_type, q_data in questions.items():
            options = format_mcq_options(q_data['options'])
            prompt = ("<|im_start|>system\nYou are a biomedical expert.<|im_end|>\n"
                     "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
                     "Answer with a single letter, no extra details.\n"
                     f"Question: {q_data['question']}\n"
                     f"{chr(10).join(options)}<|im_end|>\n"
                     "<|im_start|>assistant\n")
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image}
            })
            metadata_list.append({
                "question_type": q_type,
                "question": q_data['question'],
                "correct_answer": q_data['answer'],
                "correct_letter": get_correct_letter(q_data['answer_idx']),
                "options": q_data['options']
            })

    # Generate responses using vLLM
    sampling_params = SamplingParams(temperature=0.2, max_tokens=128)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    batch_predictions = []
    for i, output in enumerate(outputs):
        metadata = metadata_list[i]
        prediction = output.outputs[0].text.strip().upper()
        # Extract first letter if model outputs more than one character
        predicted_letter = prediction[0] if prediction else ''
        
        batch_predictions.append({
            'question_type': metadata['question_type'],
            'question': metadata['question'],
            'options': metadata['options'],
            'correct_answer': metadata['correct_answer'],
            'correct_letter': metadata['correct_letter'],
            'prediction': prediction,
            'predicted_letter': predicted_letter
        })
    
    return batch_predictions

def evaluate_mcq_predictions(predictions):
    """Evaluate multiple choice questions predictions"""
    from collections import defaultdict
    
    total = len(predictions)
    correct = 0
    results_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for pred in predictions:
        question_type = pred['question_type']
        results_by_type[question_type]['total'] += 1
        
        if pred['predicted_letter'] == pred['correct_letter']:
            correct += 1
            results_by_type[question_type]['correct'] += 1

    overall_accuracy = correct/total if total > 0 else 0
    type_accuracies = {
        qtype: {
            "accuracy": res["correct"]/res["total"] if res["total"] > 0 else 0,
            "total": res["total"]
        }
        for qtype, res in results_by_type.items()
    }
    
    return {
        "total": total,
        "overall_accuracy": overall_accuracy,
        "by_type": type_accuracies
    }

def main():
    args = parse_args()
    
    # Load dataset
    dataset_map = {
        'ubench': ('jnirschl/uBench', True),
        'vqa-rad': ('flaviagiammarino/vqa-rad', False),
        'path-vqa': ('flaviagiammarino/path-vqa', False),
        'slake': ('mdwiratathya/SLAKE-vqa-english', False)
    }
    dataset, is_mcq = dataset_map.get(args.dataset_name, (None, None))
    if dataset is None:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    dataset = load_dataset(dataset, split='test')
    dataset = dataset.select(range(100))
    
    # Initialize vLLM
    llm = LLM(
        model=args.model_id_or_path,
        max_model_len=4096,
        max_num_seqs=32,
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
    )

    # Initialize the output file
    write_results([], args.output_file, mode='w')
    
    # Run batch inference
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=custom_collate
    )
    
    predictions = []
    for batch in tqdm(dataloader, desc="Running inference"):
        if is_mcq:
            responses = process_mcq_batch(llm, batch)
        else:
            responses = process_batch(llm, batch)
        predictions.extend(responses)
        write_results(responses, args.output_file)

    # Calculate and print results
    if is_mcq:
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
            print(f"Open-ended F1 score: {results['open_ended']['f1']*100:.2f}")
            print(f"Open-ended Recall: {results['open_ended']['recall']*100:.2f}")

if __name__ == '__main__':
    main()