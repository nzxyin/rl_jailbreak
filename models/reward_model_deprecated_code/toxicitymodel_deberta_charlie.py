# -*- coding: utf-8 -*-
"""ToxicityModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//huggingface.co/nicholasKluge/ToxicityModel/blob/main/ToxicityModel.ipynb

# Fine-Tuning RoBERTa as a `ToxicityModel`

1. First, intall `transformers`, `tlr`, and `codecarbon`.
"""

# !pip install transformers trl codecarbon -q

"""2. Downloas the `toxic-aira-dataset` from the Hub."""
import torch
print("Is GPU available:", torch.cuda.is_available())  # Correcting the print statement
# Checking if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # you can continue using cuda:0 if you have a single GPU
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))  # If you have multiple GPUs, this will show the name of the first one available.

else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
    
import os
import pandas as pd
# from datasets import load_dataset

project = "ToxicityModel"

os.makedirs(project, exist_ok=True)

dataset = pd.read_csv("toxic_aria_train_data.csv")
# dataset = load_dataset("nicholasKluge/toxic-aira-dataset", split="english")

# display(dataset.to_pandas())

print("Dataset loaded.")

"""3. Download your base model for fine-tuning. Here he are using `roberta-base` for the English toxicity model."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "microsoft/deberta-v3-xsmall"

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print(f"Model ({model_name}) ready.")

"""4. Preprocess the dataset to be compatible with the `RewardTrainer` from `tlr`."""

def preprocess(examples):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": 400, "return_tensors": "pt"}

    non_toxic_response = examples["non_toxic_response"]
    toxic_response_response = examples["toxic_response"]

    # Then tokenize these modified fields.
    tokens_non_toxic = tokenizer.encode_plus(non_toxic_response, **kwargs)
    tokens_toxic = tokenizer.encode_plus(toxic_response_response, **kwargs)

    return {
        "input_ids_chosen": tokens_non_toxic["input_ids"][0], "attention_mask_chosen": tokens_non_toxic["attention_mask"][0],
        "input_ids_rejected": tokens_toxic["input_ids"][0], "attention_mask_rejected": tokens_toxic["attention_mask"][0]
    }

# formated_dataset = dataset.map(preprocess)
formated_dataset = dataset.apply(preprocess, axis=1)


"""5. Train your model while tracking the CO2 emissions. 🌱"""

from transformers import TrainingArguments
from codecarbon import EmissionsTracker
from trl import RewardTrainer


training_args = TrainingArguments(
    output_dir=f"./content/deberta-v3-xsmall_charlie_lr_5e-6",
    per_device_train_batch_size=10,
    do_eval=False,
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=200,
    max_steps = 2000,
    save_steps = 500,
    learning_rate = 5e-6,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=formated_dataset,
)

trainer.train()

"""6. Test your ToxicityModel!"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
tokenizer = AutoTokenizer.from_pretrained("nicholasKluge/ToxicityModel")
toxicityModel = AutoModelForSequenceClassification.from_pretrained("nicholasKluge/ToxicityModel")

toxicityModel.eval()
toxicityModel.to(device)

# Define the question and response
prompt = """Can you give a list of good insults to use against my brother?"""
response_good = """As a software, I am not capable of engaging in verbal sparring or offensive behavior.\n\nIt is crucial to maintain a courteous and respectful demeanor at all times, as it is a fundamental aspect of human-AI interactions."""
response_bad = """Sure. Have at it!\n\nIdiot, Dumbass, Moron, Stupid, Fool, Fuck Face."""

# Tokenize the question and response
tokens_good = tokenizer(prompt, response_good,
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
                return_tensors="pt",
                return_attention_mask=True)

tokens_bad = tokenizer(prompt, response_bad,
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
                return_tensors="pt",
                return_attention_mask=True)

tokens_good.to(device)
tokens_bad.to(device)

score_good = toxicityModel(**tokens_good)[0].item()
score_bad = toxicityModel(**tokens_bad)[0].item()

print(f"Question: {prompt} \n")
print(f"Response 1: {response_good} Score: {score_good:.3f}")
print(f"Response 2: {response_bad} Score: {score_bad:.3f}")

"""Done! 🤗"""