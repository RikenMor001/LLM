# config
BATCH_SIZE = 32
CONTEXT_LENGTH = 128
EVAL_INTERVAL = 200
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 8
N_KV_HEADS = 2 # This is grouped query attention
HEAD_DIM = D_MODEL // N_HEADS
FFN_HIDDEN = 680 # FFN = Feed Forward Network
DROPOUT = 0.2
MAX_SEQ_LEN = 256 # This is the maximum sequence length for the model