# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:53:39 2024

@author: husse
"""

# %% importing dependencies

from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
TrainingArguments, Trainer

from datasets import load_dataset, DatasetDict

import numpy, evaluate

# %% importing dataset, tokenizer, model

checkpoint = 'bert-base-uncased'
dataset = load_dataset('glue', 'mrpc')
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

dataset = DatasetDict({
    'train': dataset['train'].shuffle(seed=42).select(range(20)),
    'validation': dataset['validation'].shuffle(seed=42).select(range(8)),
    'test': dataset['test'].shuffle(seed=42).select(range(8)),
    })

# %% Tokenizer funciton


def tokenize_function(dataset):
    return tokenizer(dataset['sentence1'], dataset['sentence2'], truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# %% Training Arguments and compute metrics

training_args = TrainingArguments('test-trainer', eval_strategy='epoch',
                                  logging_strategy='epoch')


def compute_metrics(preds):
    logits, labels = preds
    predictions = numpy.argmax(logits, axis=-1)
    metrics = evaluate.load('glue', 'mrpc')
    return metrics.compute(predictions=predictions, references=labels)

# %% Training


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    )

# %% Training

trainer.train()
