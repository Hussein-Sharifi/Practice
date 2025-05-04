# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:42:50 2025

@author: husse
"""

# %% Importing Dependencies

from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
DataCollatorWithPadding, AdamW, get_scheduler

from datasets import DatasetDict, load_dataset

import evaluate, torch

from tqdm.auto import tqdm


# %% Loading Model, tokenizer, collator, and Datasets

checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                           num_labels=2)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
dataset = load_dataset('glue', 'sst2')

small_train = dataset['train'].shuffle(seed=3).select(range(16))
small_eval = dataset['validation'].shuffle(seed=3).select(range(8))

dataset = DatasetDict({
    'train': small_train,
    'validation': small_eval})

# %% Tokenizing dataset


def tokenize_function(dataset):
    return tokenizer(dataset['sentence'], truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %% Preprocessing datasets

tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

# %% Loading datasets

train_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets['train'], batch_size=16, collate_fn=data_collator)
eval_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)

# %% Setting optimizer and learning rate

optimizer = AdamW(model.parameters(), lr = 5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps)

# %% Training Loop

device = torch.device('cpu')
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# %% Evaluation Loop

metric = evaluate.load('glue', 'sst2')
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k,v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references = batch['labels'])

metric.compute()