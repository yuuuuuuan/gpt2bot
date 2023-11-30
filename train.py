from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import datasets

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

raw_datasets = load_dataset("HuggingFaceH4/no_robots")
raw_train_dataset = raw_datasets["train_sft"]
raw_test_dataset = raw_datasets["test_sft"]

processed_dialogues = []
for example in raw_train_dataset:
    question = example["prompt"]
    answer = example["messages"]
    dialogue = f"Question: {question} Answer: {answer}"
    processed_dialogues.append(dialogue)

train_dataset = datasets.Dataset.from_dict({"text": processed_dialogues})

tokenized_dataset = train_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True, padding="longest"))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
tokenizer.save_pretrained("./gpt2-finetuned")
trainer.save_model()