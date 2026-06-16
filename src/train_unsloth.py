# training the unsloth model that is already been coded
# SFTTrainer expects an object and not a list

from models.unsloth_loader import load_model, add_lora
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

model, tokenizer = load_model()
model = add_lora(model)
dataset = Dataset.from_dict({
    "text": [open("input.txt", "r", encoding="utf-8").read()]
})

trainer = SFTTrainer(
    # for training sfttrainer you need
    # model, tokenzier, train the collected dataset,
    # dataset_text_field (let know what the type is)
    # and pass the arguments
    model = model, 
    processing_class = tokenizer,
    train_dataset=dataset,
    dataset_text_field = "text",
    args=TrainingArguments(
        output_dir="outputs",
        num_train_epochs=1
    )
)

trainer.train()