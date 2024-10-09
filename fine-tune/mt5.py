import os

import dotenv
import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

dotenv.load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set your Hugging Face token
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("'HF_TOKEN' environment variable not defined.")

include_description = False
hf_username = "MittyN"
model_name = "google/mt5-base"
train_size = 500000
val_size = 50000
test_size = 50000

fine_tuned_model = f"bslm-entity-extraction-{'include-desc' if include_description else 'exclude-desc'}-{model_name.split('/')[1]}-tr{train_size}"
repo_name = f"{hf_username}/{fine_tuned_model}"


def load_split_dataset(train_size=50000, val_size=5000, test_size=1000):
    """
    :return: train, validation and test datasets
    """

    dataset = load_dataset("BaSalam/bslm-product-entity-cls-610k", split='train')
    dataset = dataset.train_test_split(train_size=train_size, test_size=val_size + test_size, shuffle=True)
    test_val_ds = dataset['test'].train_test_split(test_size=test_size, shuffle=True)

    return dataset["train"], test_val_ds["train"], test_val_ds["test"]


# Split the dataset into train and validation sets
train_dataset, validation_dataset, test_dataset = load_split_dataset(
    train_size=train_size,
    val_size=val_size,
    test_size=test_size
)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True, clean_up_tokenization_spaces=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Tokenization function
def tokenize_function(examples, include_desc=True):
    if include_desc:
        inputs = [
            f"title: {title} description: {desc or ''}"
            for title, desc in zip(examples['title'], examples['description'])
        ]
    else:
        inputs = examples['title']

    targets = examples['entity']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Tokenize datasets
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    fn_kwargs={"include_desc": include_description}
)
tokenized_val = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=validation_dataset.column_names,
    fn_kwargs={"include_desc": include_description}
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Metric for evaluation
metric = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 4) for k, v in result.items()}


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./results/{fine_tuned_model}",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=1000,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    push_to_hub=True,
    hub_model_id=repo_name,
    hub_token=hf_token,
    hub_private_repo=True
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Save the model
trainer.save_model(f"./save/{fine_tuned_model}")
tokenizer.save_pretrained(f"./save/{fine_tuned_model}")

try:
    print(trainer.push_to_hub())
except Exception as e:
    print(f"Can not push model: {e}")

all_dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

try:
    print(
        all_dataset.push_to_hub(
            repo_name,
            token=hf_token,
            private=True
        )
    )
except Exception as e:
    print(f"Can not push dataset: {e}")

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
