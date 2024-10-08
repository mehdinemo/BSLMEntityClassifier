import logging

import tiktoken
from datasets import load_dataset

from constants import DATASET_PATH

models = [
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo-0301",

]


def logger_function(
        name: str,
        filename: str = "bslm",
        level: int = logging.DEBUG
) -> logging.Logger:
    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a stream handler to log to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'{filename}.log')
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format the log messages
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    stream_handler.setFormatter(log_formatter)
    file_handler.setFormatter(log_formatter)

    # Add the stream handler to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def load_split_dataset(train_size=50000, val_size=5000, test_size=1000):
    """
    :return: train, validation and test datasets
    """

    dataset = load_dataset(DATASET_PATH, split='train')
    dataset = dataset.train_test_split(train_size=train_size, test_size=val_size + test_size, shuffle=True)
    test_val_ds = dataset['test'].train_test_split(test_size=test_size, shuffle=True)

    return dataset["train"], test_val_ds["train"], test_val_ds["test"]
