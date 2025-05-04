# %% Importing dependencies, dataset, model, and tokenizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
TrainingArguments, Trainer

from datasets import load_dataset, DatasetDict

import numpy, evaluate

checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_dataset = load_dataset("glue", "sst2") 

# %% Splitting Dataset

small_train = raw_dataset['train'].shuffle(seed=42).select(range(1000))
small_eval = raw_dataset['validation'].shuffle(seed=42).select(range(50))
small_test = raw_dataset['validation'].shuffle(seed=42).select(range(50))

small_datasets = DatasetDict({
    "train": small_train,
    "validation": small_eval,
    "test": small_test
})

# %% Pre-processing dataset

def tokenize_function(dataset):
    return tokenizer(dataset['sentence'], truncation = True)

tokenized_datasets = small_datasets.map(tokenize_function, batched = True)

# %% Compute metrics function

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metrics = evaluate.load("glue", "sst2")
    predictions = numpy.argmax(logits, axis = -1)
    return metrics.compute(predictions = predictions, references = labels)

# %% Fine tune

training_args = TrainingArguments("test-trainer", eval_strategy="epoch")


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    )

# %% Train model
trainer.train()
