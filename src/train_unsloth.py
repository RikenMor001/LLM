# training the unsloth model that is already been coded
from models.unsloth_loader import load_model, add_lora
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = load_model()
model = add_lora(model)

trainer = SFTTrainer(
    model = model, 
    train_dataset = [{"text":open("input.txt").read()}],
    args = TrainingArguments(
        output_dir = "outputs",
        num_train_epochs = 1
    )
)
trainer.train() 