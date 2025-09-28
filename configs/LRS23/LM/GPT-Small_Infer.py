import sys
sys.path.append("../../")

# Minimal GPT-Small config for inference-only neural LM rescoring
import nnet
import torch

vocab_size = 1024
model_name = "GPT-Small"
max_pos_encoding = 2048
pad_token = 0
sos_token = vocab_size
eos_token = vocab_size

tokenizer_path = "datasets/LRS3/tokenizerbpe1024.model"
callback_path = "callbacks/LRS23/LM/GPT-Small"

# Build model (no datasets here)
model = nnet.GPT(
    vocab_size=vocab_size + 1,
    padding_idx=pad_token,
    max_pos_encoding=max_pos_encoding,
    model=model_name,
    pos_embedding=nnet.SinPosEmbedding,
)

# Optimizer not needed for inference; compile minimally to satisfy .load interface if required
# Provide a minimal dummy optimizer to satisfy compile() expectations
class _DummyScheduler:
    def __init__(self):
        self.model_step = 0

class _DummyOptim:
    def __init__(self):
        self.scheduler = _DummyScheduler()
    def zero_grad(self):
        pass
    def step(self):
        pass

model.compile(optimizer=_DummyOptim())
