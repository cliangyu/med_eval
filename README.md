# Medical QA Evaluation

This is the evaluation code for the medical QA tasks.

## Features

* Evaluate open-ended and closed-ended questions.
* Evaluate on huggingface datasets.
* Evaluate with vllm-supported LLMs.

## Requirements

```
pip install -r requirements.txt
# install latest vllm
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

## Datasets

* [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
* [Path-VQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
* [SLAKE-VQA English](https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english)
* [uBench](https://huggingface.co/datasets/jnirschl/uBench)

All are huggingface datasets. 

Except for uBench, each instance consists of an image-question-answer triplet.

```python
{
  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=CMYK size=309x272>,
  'question': 'where are liver stem cells (oval cells) located?',
  'answer': 'in the canals of hering'
}
```

For uBench, each instance is a dictionary with multiple choice questions.


## Usage

```
# output the answer to a file and then evaluate the answer file in eval.py
cd ..
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval_subcellular --model_id_or_path /mnt/disks/persistent-disk/code/swift/output/llava-onevision-qwen2-0_5b-ov/v12-20241204-034303/checkpoint-123 --max_pixels 50176
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name ubench --model_id_or_path /pasteur/u/liangyuc/code/swift/output/qwen2-vl-2b-instruct/v30-20241103-234014/checkpoint-901 --output_file answer-file-qwen2-vl-2b-instruct.jsonl --max_pixels 50176
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name vqa-rad --model_id_or_path /home/ly/.cache/modelscope/hub/qwen/Qwen2-VL-7B-Instruct --max_pixels 50176

CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name vqa-rad --model_id_or_path /mnt/disks/persistent-disk/code/output/InternVL2_5-4B/v6-20241219-052230/checkpoint-13 --batch_size 1

CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name vqa-rad --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v20-20241206-095923/checkpoint-4757 --max_pixels 1000000 
CUDA_VISIBLE_DEVICES=1 python -m med_eval.infer_eval  --dataset_name path-vqa --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v20-20241206-095923/checkpoint-4757 --max_pixels 1000000
CUDA_VISIBLE_DEVICES=2 python -m med_eval.infer_eval  --dataset_name slake --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v20-20241206-095923/checkpoint-4757 --max_pixels 1000000
CUDA_VISIBLE_DEVICES=1 python -m med_eval.infer_eval  --dataset_name path-vqa --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v14-20241206-010529/checkpoint-1000 --max_pixels 1000000
CUDA_VISIBLE_DEVICES=2 python -m med_eval.infer_eval  --dataset_name slake --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v14-20241206-010529/checkpoint-1000 --max_pixels 1000000
CUDA_VISIBLE_DEVICES=1 python -m med_eval.infer_eval  --dataset_name slake --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v14-20241206-010529/checkpoint-1000 --max_pixels 50176
'CUDA_VISIBLE_DEVICES=3 python -m med_eval.infer_eval  --dataset_name ubench --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v8-20241118-223800/checkpoint-306
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name vqa-rad --model_id_or_path /sailhome/liangyuc/.cache/modelscope/hub/qwen/Qwen2-VL-2B-Instruct --output_file answer-file-qwen2-vl-2b-instruct.jsonl

CUDA_VISIBLE_DEVICES=3 python -m med_eval.infer_eval  --dataset_name vqa-rad --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-2b-instruct/v22-20241123-222820/checkpoint-1224
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name path-vqa --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v14-20241120-013332/checkpoint-486
CUDA_VISIBLE_DEVICES=1 python -m med_eval.infer_eval  --dataset_name slake --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v14-20241120-013332/checkpoint-486
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name path-vqa --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v14-20241120-013332/checkpoint-486

CUDA_VISIBLE_DEVICES=3 python -m med_eval.eval_hussain --model_id_or_path /mnt/disks/persistent-disk/code/swift/output/qwen2-vl-7b-instruct/v12-20241119-231240/checkpoint-63 --max_pixels 50176

CUDA_VISIBLE_DEVICES=0 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v18-20241121-024730/checkpoint-105
CUDA_VISIBLE_DEVICES=1 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v18-20241121-024730/checkpoint-105 --max_pixels 50176
CUDA_VISIBLE_DEVICES=2 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v19-20241121-034437/checkpoint-105
CUDA_VISIBLE_DEVICES=3 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-7b-instruct/v19-20241121-034437/checkpoint-105 --max_pixels 50176

# eval wsi vqa
CUDA_VISIBLE_DEVICES=0 python -m med_eval.eval_hussain --model_id_or_path /pasteur2/u/liangyuc/code/output/qwen2-vl-2b-instruct/v3-20250112-174415/checkpoint-442 --max_pixels 50176 --hussain_file /pasteur2/u/liangyuc/data/brca_json/WsiVQA_test_shuffled_subtype.jsonl --output_file /pasteur2/u/liangyuc/code/output/qwen2-vl-2b-instruct/v3-20250112-174415/checkpoint-442/output_eval.jsonl
CUDA_VISIBLE_DEVICES=1 python -m med_eval.eval_hussain --model_id_or_path /pasteur2/u/liangyuc/code/output/qwen2-vl-2b-instruct/v2-20250112-150028/checkpoint-442/ --max_pixels 50176 --hussain_file /pasteur2/u/liangyuc/data/brca_json/WsiVQA_test_shuffled_subtype.jsonl --output_file /pasteur2/u/liangyuc/code/output/qwen2-vl-2b-instruct/v2-20250112-150028/checkpoint-442/output_eval.jsonl

  /pasteur2/u/liangyuc/code/output/qwen2-vl-2b-instruct/v2-20250112-150028/checkpoint-442/


CUDA_VISIBLE_DEVICES=3 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-2b-instruct/v59-20241121-073714/checkpoint-1323 --max_pixels 50176 --hussain_file /home/ly/d/data/brca_json/WsiVQA_test_shuffled.jsonl
CUDA_VISIBLE_DEVICES=2 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-2b-instruct/v59-20241121-073714/checkpoint-1323 --max_pixels 50176 --hussain_file /home/ly/d/data/brca_json/WsiVQA_test_shuffled.jsonl --max_pixels 50176
CUDA_VISIBLE_DEVICES=2 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-2b-instruct/v56-20241121-055610/checkpoint-500 --max_pixels 50176 --hussain_file /home/ly/d/data/brca_json/WsiVQA_test_shuffled.jsonl --max_pixels 50176
# bbbc
CUDA_VISIBLE_DEVICES=2 python -m med_eval.eval_hussain --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v30-20241211-224739/checkpoint-780 --hussain_file /home/ly/d/data/bbbc_rgb/bbbc021_test.jsonl --max_pixels 50176
CUDA_VISIBLE_DEVICES=3 python -m med_eval.eval_hussain --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v30-20241211-224739/checkpoint-780 --hussain_file /home/ly/d/data/bbbc_rgb/bbbc021_test.jsonl --max_pixels 50176
CUDA_VISIBLE_DEVICES=3 python -m med_eval.eval_hussain --model_id_or_path  /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v19-20241206-074333/checkpoint-1560 --hussain_file /home/ly/d/data/bbbc_rgb/bbbc021_test.jsonl --max_pixels 50176

CUDA_VISIBLE_DEVICES=2 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-2b-instruct/v1-20241204-222716/checkpoint-441 --hussain_file /home/ly/d/data/brca_json/WsiVQA_test_shuffled.jsonl --max_pixels 50176 --batch_size 1
CUDA_VISIBLE_DEVICES=3 python -m med_eval.eval_hussain --model_id_or_path /home/ly/d/code/swift/output/qwen2-vl-2b-instruct/v1-20241204-222716/checkpoint-441 --hussain_file /home/ly/d/data/brca_json/WsiVQA_test_shuffled.jsonl --batch_size 2



CUDA_VISIBLE_DEVICES=0 python -m med_eval.eval_hussain --model_id_or_path /mnt/disks/persistent-disk/code/output/Qwen2-VL-2B-Instruct/v18-20241206-071821/checkpoint-1565 --max_pixels 50176
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name path-vqa --model_id_or_path /pasteur/u/liangyuc/code/swift/output/qwen2-vl-2b-instruct/v27-20241027-115543/checkpoint-1802 --output_file answer-file-qwen2-vl-2b-instruct.jsonl --max_pixels 12845056

# evaluate the saved answer file to get the results
python -m med_eval.eval --dataset_name vqa-rad --pred_file_parent_path answer-file-qwen2-vl-2b-instruct.jsonl
python -m med_eval.eval --dataset_name ubench --pred_file_parent_path answer-file-qwen2-vl-2b-instruct.jsonl
```
The question is closed-ended if the groundtruth answer is one of the options [yes, no].
All predictions and groundtruths are in lowercase for evaluation.
For open-ended questions, the evaluation metric is recall (function `calculate_f1score`).
For closed-ended questions, the evaluation metric is accuracy (check if 'yes' or 'no' in the response).

