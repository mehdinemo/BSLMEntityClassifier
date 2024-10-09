import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

# Set your Hugging Face token
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("'HF_TOKEN' environment variable not defined.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

hf_username = "MittyN"
model_name = "google/mt5-base"
train_size = 500000

fine_tuned_model = f"bslm-entity-extraction-{model_name.split('/')[1]}-tr{train_size}"
repo_name = f"{hf_username}/{fine_tuned_model}"


def add_input(examples):
    inputs = [
        f"title: {title} description: {description or ''}"
        for title, description in zip(examples['title'], examples['description'])
    ]

    return {"input": inputs}


dataset = load_dataset(repo_name, split='test', token=hf_token)
dataset = dataset.map(add_input, batched=True)

pipe = pipeline(
    "text2text-generation",
    model=f"./save/{fine_tuned_model}",
    token=hf_token,
    device_map="auto"
)

generated_entities = [
    out[0]['generated_text'].replace("<extra_id_0>", "").replace("<extra_id_0>:", "").strip() for out in tqdm(
        pipe(
            KeyDataset(dataset, 'input'),
            max_length=64,
            batch_size=16
        ),
        total=len(dataset)
    )
]

dataset = dataset.add_column("generated_entity", generated_entities)
dataset = dataset.remove_columns('input')

try:
    print(
        dataset.push_to_hub(
            repo_name + "-results",
            token=hf_token,
            private=True
        )
    )
except Exception as e:
    print(f"Can not push dataset: {e}")
