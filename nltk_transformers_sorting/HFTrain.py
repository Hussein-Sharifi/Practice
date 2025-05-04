# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:45:16 2024

@author: husse
"""
# %% Importing dependencies

from transformers import AutoModelForSequenceClassification, AutoTokenizer,\
DataCollatorWithPadding

from datasets import load_dataset, DatasetDict

# %% loading dataset, model, tokenizer

checkpoint = 'bert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_datasets = load_dataset('glue', 'mrpc')

raw_datasets = DatasetDict({
    'train': raw_datasets['train'].shuffle(seed=42).select(range(8)),
    'validation': raw_datasets['validation'].shuffle(seed=42).select(
                                                                  range(4)),
    'test': raw_datasets['test'].shuffle(seed=42).select(range(4))})

# %% Pre-process datasets


def tokenizer_function(datasets):
    return tokenizer(datasets['sentence1'], datasets['sentence2'],
                     truncation=True)


tokenized_datasets = raw_datasets.map(tokenizer_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% Post-Process datasets into model's format
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('labels', 'labels')
tokenized_datasets.set_format('torch')
tokenized_datasets['train'].column_names

# %% Use dataloader to create batches
from torch.utils.data import DataLoader