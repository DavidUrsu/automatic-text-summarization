import torch
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM, Trainer
import pandas as pd
from datasets import Dataset
import re


class ModelLLM:
    def __init__(self, model_name):
        self.model_name = ''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(model_name).lower() == 'bart':
            self.model_name = 'facebook/bart-large-cnn'
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif str(model_name).lower() == 't5':
            self.model_name = 't5-base'
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif str(model_name).lower() == 'extractive':
            pass

        self.model = model.to(self.device)


    def infer(self, text):
        if "t5" in self.model_name:
            text = "summarize: " + text
        inputs = self.tokenizer(text, truncation=True, return_tensors="pt", padding="max_length", max_length=512)
        if self.device != torch.device("mps"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=64,
            min_length=20,
            num_beams=4,
            early_stopping=True,
            length_penalty=2.0,
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("Generated summary:", summary)
        summary = self.trim_to_sentence(summary)
        return summary

    def trim_to_sentence(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        output = ""
        word_count = 0

        for sentence in sentences:
            words = sentence.split()
            if word_count + len(words) > 30:
                break
            output += sentence + " "
            word_count += len(words)

        sentences = output.split('. ')
        output = ""
        for sentence in sentences:
            output += sentence.strip().capitalize() + '.'

        return output.strip()

