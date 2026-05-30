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
            "q_proj","k_proj","v_proj","o_proj", # focus on words and context
            "gate_proj","down_proj","up_proj", # focus on knowledge storage and pattern recognition
            "embed_tokens","lm_head" # embed_tokens, converts words into vectors and lm_head converts vectors into words
        ],
        lora_alpha = 32, # provides stability
        lora_dropout = 0.05, # randomly drops 5% neurons to prevent overfitting
        bias = "none", # No bias term
        use_gradient_checkpoint = "unsloth", #Enables gradient checkpointing for memory efficiency
        random_state = SEED, # reproducibility
        use_rslora = True
    )

def set_inference_mode(model):
    FastLanguageModel.for_inference(model)
    return model