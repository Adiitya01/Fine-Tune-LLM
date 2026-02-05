# -*- coding: utf-8 -*-


!pip install datasets

from datasets import load_dataset

dataset = load_dataset("go_emotions")

print(dataset)

print(dataset["train"][1])

label_list = dataset["train"].features["labels"].feature.names

sample = dataset["train"][0]

print("Text:", sample["text"])
print("Label IDs:", sample["labels"])
print("Emotions:", [label_list[i] for i in sample["labels"]])

def format_examples(example):
  emotions =",".join([label_list[i] for i in example["labels"]])
  return {
        "prompt": f"What emotion is expressed in this sentence? '{example['text']}'",
        "response": emotions
    }

# Apply to the whole dataset
formatted_dataset = dataset.map(format_examples)

!pip install transformers

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1.5")

def tokenize_example(example):
  # Tokenize both the prompt and response
  # We'll use padding and truncation to ensure uniform input size

  encoding = tokenizer(example["promot"] , example["responce"] , padding = "max_length" , truncation = True , max_length = 128)
  return encoding

# Applied to whole datasets
tokenized_dataset = formatted_dataset.map(tokenize_example, remove_columns=["prompt", "response"])

from transformers import TrainingArguments, Trainer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1.5")

training_args = TrainingArguments(
    output_dir="./results",           # Output directory to save checkpoints
    #evaluation_strategy="epoch",      # Evaluate after each epoch
    num_train_epochs=3,               # Number of epochs to train
    per_device_train_batch_size=8,    # Batch size per device during training
    per_device_eval_batch_size=8,     # Batch size during evaluation
    weight_decay=0.01,                # Weight decay for regularization
    logging_dir="./logs",             # Directory to save logs
    logging_steps=10,                 # Log every 10 steps
    save_steps=500,                   # Save checkpoint every 500 steps
    save_total_limit=2,               # Only keep 2 most recent checkpoints
    remove_unused_columns=False,      # Keep all columns (like labels)
)

# Initialize Trainer
trainer = Trainer(
    model=model,                                        # The model we are fine-tuning
    args=training_args,                                 # Training arguments
    train_dataset=tokenized_dataset["train"],           # Training dataset
    eval_dataset=tokenized_dataset["test"],             # Evaluation dataset
    tokenizer=tokenizer,                                # The tokenizer used for encoding
)

trainer.train()

# Evaluate the model on the test dataset
results = trainer.evaluate()

# Print the evaluation results
print("Evaluation Results:", results)

# Save the model and tokenizer
model.save_pretrained("./fine_tuned_phi_1.5")
tokenizer.save_pretrained("./fine_tuned_phi_1.5")





