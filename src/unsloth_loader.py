# unsloth in an open source library for faster LLM inference
# and for memory efficiency 
from unsloth import FastLanguageModel

BASE_MODEL = "HuggingFaceTB/SmolLM-135M"
MAX_SEQ_LENGTH = 512
SEED = 42

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_length = MAX_SEQ_LENGTH,
        load_in_4bit = True
    )
    return model, tokenizer

def add_lora(model):
    return FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules = [
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","down_proj","up_proj",
            "embed_tokens","lm_head"
        ],
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpoint = "unsloth",
        random_state = SEED,
        use_rslora = True
    )