# config
BATCH_SIZE = 32
CONTEXT_LENGTH = 128
EVAL_INTERVAL = 200
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 2 # This is grouped query attention
FFN_HIDDEN = 680 # FFN = Feed Forward Network
DROPOUT = 0.2
MAX_SEQ_LEN = 256 # This is the maximum sequence length for the model
MAX_STEPS = 10000
HEAD_DIM = D_MODEL // N_HEADS
LABEL_SMOOTHING = 0.1
LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
EPS = 1e-8
ETA_MIN = 1e-5
CHECKPOINT_PATH = "checkpoint.pt"
DATA_PATH = "input.txt"