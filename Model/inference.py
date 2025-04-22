from transformers import AutoTokenizer, TrainingArguments, AutoModelForSeq2SeqLM, Trainer

model_path = "./results"  # or wherever you saved it
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

article_text = "{{path}}"

input_text = "summarize: " + article_text

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

summary_ids = model.generate(
    inputs["input_ids"],
    max_length=64,
    num_beams=4,
    early_stopping=True
)

generated_title = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Generated Title:", generated_title)