# Medical QA Evaluation

This is the evaluation code for the medical QA tasks.

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

All are huggingface datasets. Each instance consists of an image-question-answer triplet.

```python
{
  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=CMYK size=309x272>,
  'question': 'where are liver stem cells (oval cells) located?',
  'answer': 'in the canals of hering'
}
```


## Usage

```
# output the answer to a file and then evaluate the answer file in eval.py
cd ..
CUDA_VISIBLE_DEVICES=0 python -m med_eval.infer_eval  --dataset_name vqa-rad --model_id_or_path /pasteur/u/liangyuc/code/swift/output/qwen2-vl-2b-instruct/v27-20241027-115543/checkpoint-1802 --output_file answer-file-qwen2-vl-2b-instruct.jsonl

# evaluate the saved answer file directly
python -m med_eval.eval --dataset_name vqa-rad --pred_file_parent_path answer-file-qwen2-vl-2b-instruct.jsonl
```
The question is closed-ended if the groundtruth answer is one of the options [yes, no].
All predictions and groundtruths are in lowercase for evaluation.
For open-ended questions, the evaluation metric is recall (function `calculate_f1score`).
For closed-ended questions, the evaluation metric is accuracy (check if 'yes' or 'no' in the response).

