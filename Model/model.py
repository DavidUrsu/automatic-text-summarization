import torch
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM, Trainer
import pandas as pd
from datasets import Dataset

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

# ✅ Load CSV
df = pd.read_csv("../news-article-categories.csv")  # columns: article, title
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1)

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess(inputs):
    articles = inputs["body"]
    titles = inputs["title"]
    articles = ["summarize: " + str(article) for article in articles]
    inputs = tokenizer(articles, truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(titles, truncation=True, padding="max_length", max_length=64)
    inputs["labels"] = targets["input_ids"]
    return inputs

training_args = TrainingArguments(
    output_dir="./title_model",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    fp16=False,
)

tokenized_dataset = dataset.map(preprocess, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()


