# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:33:18 2024

@author: husse
"""

# %% importing dependencies

from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    DataCollatorWithPadding, AdamW, get_scheduler

import torch

from datasets import load_dataset, DatasetDict

from tqdm.auto import tqdm

# %% importing model tokenizer and dataset

checkpoint = 'bert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
dataset = load_dataset('glue', 'mrpc')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset = DatasetDict({
    'train': dataset['train'].shuffle(seed=12).select(range(16)),
    'validation': dataset['validation'].shuffle(seed=12).select(range(8)),
    'test': dataset['test'].shuffle(seed=12).select(range(8)),
    })

# %% preprocessing data


def tokenize_function(data):
    return tokenizer(data['sentence1'], data['sentence2'], truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# %% formatting tokenized_dataset for model

tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

# %% setting data loaders

train_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets['train'], batch_size=16, collate_fn=data_collator)
eval_dataloader = torch.utils.data.DataLoader(
    tokenized_datasets['validation'], batch_size=8, collate_fn=data_collator)

# %% sample batch testing

for batch in train_dataloader:
    break

print({k: v.shape for k, v in batch.items()})
output = model(**batch)
print(output.loss, output.logits.shape)


# %% setting optimizer and learning rate

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
    )

# %% Setting gpu or cpu device

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
device

# %% Training loop with progress bar

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

# %% Adding evaluation loop

import evaluate


metric = evaluate.load('glue', 'mrpc')
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch['labels'])

metric.compute()
