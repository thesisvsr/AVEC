import os, sys, types, glob, re, torch
sys.path.append("../../")

import nnet
import torch.nn as nn
from nnet import transforms as vtf
from nnet import schedulers as _schedulers
from nnet import optimizers as _optim

"""LRW-AR Visual Efficient Conformer Inter-CTC transfer config

Transfer from LRS23 SWA checkpoint (visual-only InterCTC) loading encoder weights
(front_end/back_end) while keeping new CTC heads (main + inter) randomly initialized.

Features:
 - Curriculum hooks can be added similarly if needed (not included by default here)
 - Best+last-K checkpoint retention (optional; simple baseline here)
 - Differential LR scaling for encoder vs heads via lr_scale param groups
 - Optional initial encoder freezing (env LRWARCTC_FREEZE_EPOCHS)

Environment overrides:
  LRWARCTC_EPOCHS (default 200)
  LRWARCTC_FREEZE_EPOCHS (default 3)
  LRWARCTC_ENCODER_LR_SCALE (default 0.3)
  LRWARCTC_LR_MAX (default 1e-4)
  LRWARCTC_LR_MIN (default 1e-5)
  LRWARCTC_WARMUP_STEPS (default 400)
  RETAIN_LAST_K (fallback to 5)
"""

# ---------------------------------------------------------------------------
# Datasets (Arabish transliteration enabled)
# ---------------------------------------------------------------------------

crop_size = (88, 88)
center_crop = (96, 96)

train_video_transform = nn.Sequential(
    vtf.RandomTemporalShift(max_shift=2),
    vtf.RandomCropVideo(crop_size),
    vtf.RandomHorizontalFlipVideo(p=0.5),
)
val_video_transform = vtf.CenterCropVideo(crop_size)

collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])

training_dataset = nnet.datasets.LRWAR(
    batch_size=32,
    collate_fn=collate_fn,
    mode="train",
    video_transform=train_video_transform,
    center_crop_size=center_crop,
    fixed_frames=30,
    load_audio=False,
    indices_path="datasets/LRW-AR/indices/train.pt",
    prepared_only=True,
    use_arabish=True,
)

evaluation_dataset = nnet.datasets.LRWAR(
    batch_size=64,
    collate_fn=collate_fn,
    mode="val",
    video_transform=val_video_transform,
    center_crop_size=center_crop,
    fixed_frames=30,
    load_audio=False,
    indices_path="datasets/LRW-AR/indices/val.pt",
    prepared_only=True,
    use_arabish=True,
)

### Character vocabulary construction (Arabish) ###
_words = list(getattr(training_dataset, 'classes', []))
if not _words:
    raise RuntimeError("LRW-AR classes list empty; cannot build char vocab.")
_char_set = sorted({c for w in _words for c in w})
char2idx = {c: i+1 for i, c in enumerate(_char_set)}  # 0 reserved blank
idx2char = {i: c for c, i in char2idx.items()}
blank_idx = 0
# persist vocab
_vdir = 'datasets/LRW-AR/vocab'
os.makedirs(_vdir, exist_ok=True)
with open(os.path.join(_vdir,'char_vocab.txt'), 'w', encoding='utf-8') as f:
    f.write('<blank>\n')
    for c in _char_set:
        f.write(c+'\n')

def _lrwar_ctc_collate(samples):
    vids = []
    seqs = []  # list of 1D tensors (char ids)
    for v, _a, lbl in samples:
        vids.append(v.permute(1,2,3,0))  # (T,H,W,C)
        word = _words[int(lbl)]
        ids = [char2idx[c] for c in word if c in char2idx]
        if not ids:  # safety fallback
            ids = [next(iter(char2idx.values()))]
        seqs.append(torch.tensor(ids, dtype=torch.long))
    videos = torch.stack(vids, dim=0)
    T = videos.shape[1]
    vid_lengths = torch.full((videos.shape[0],), T, dtype=torch.long)
    y = torch.cat(seqs, dim=0)
    y_len = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    tgt = (y, y_len)
    targets = {k: tgt for k in ("outputs","ctc_3","ctc_7","ctc_11")}
    return {"inputs": [videos, vid_lengths], "targets": targets}

training_dataset.collate_fn = _lrwar_ctc_collate
evaluation_dataset.collate_fn = _lrwar_ctc_collate

# ---------------------------------------------------------------------------
# Model (char-level vocab)
# ---------------------------------------------------------------------------
vocab_size = len(char2idx) + 1
interctc_blocks = [4, 8, 12]
model = nnet.VisualEfficientConformerInterCTC(vocab_size=vocab_size, interctc_blocks=interctc_blocks)
loss_weights = [0.5/3, 0.5/3, 0.5/3, 0.5]

# ---------------------------------------------------------------------------
# Optimizer with encoder/head scaling
# ---------------------------------------------------------------------------
freeze_encoder_epochs = int(os.environ.get("LRWARCTC_FREEZE_EPOCHS", 3))
encoder_lr_scale = float(os.environ.get("LRWARCTC_ENCODER_LR_SCALE", 0.3))
_epochs = int(os.environ.get("LRWARCTC_EPOCHS", 200))
lr_max = float(os.environ.get("LRWARCTC_LR_MAX", 1e-4))
lr_min = float(os.environ.get("LRWARCTC_LR_MIN", 1e-5))
warmup_steps = int(os.environ.get("LRWARCTC_WARMUP_STEPS", 400))


def build_param_groups_with_scaling(model: torch.nn.Module, encoder_lr_scale: float):
    decay_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    no_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    decay_params = ("weight",)
    no_decay_params = ("bias",)
    enc_decay, enc_no_decay, head_decay, head_no_decay = [], [], [], []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            is_encoder = full_name.startswith("encoder.")
            tgt_d = enc_decay if is_encoder else head_decay
            tgt_nd = enc_no_decay if is_encoder else head_no_decay
            placed = False
            for nd in no_decay_params:
                if param_name.endswith(nd):
                    tgt_nd.append(param)
                    placed = True
                    break
            if placed:
                continue
            for dp in decay_params:
                if param_name.endswith(dp) and isinstance(module, decay_modules):
                    tgt_d.append(param); placed=True; break
                if param_name.endswith(dp) and isinstance(module, no_decay_modules):
                    tgt_nd.append(param); placed=True; break
            if not placed:
                tgt_nd.append(param)
    return [
        {"params": enc_decay, "weight_decay": 0.05, "lr_scale": encoder_lr_scale},
        {"params": enc_no_decay, "weight_decay": 0.0,  "lr_scale": encoder_lr_scale},
        {"params": head_decay, "weight_decay": 0.05, "lr_scale": 1.0},
        {"params": head_no_decay, "weight_decay": 0.0,  "lr_scale": 1.0},
    ]

class MultiScaleAdamW(_optim.AdamW):
    def step(self, closure=None):  # type: ignore[override]
        base_lr = self.scheduler.step()
        for group in self.param_groups:
            scale = group.get("lr_scale", 1.0)
            group['lr'] = base_lr * scale
        return super(_optim.AdamW, self).step(closure)

transfer_scheduler = _schedulers.CosineAnnealingScheduler(
    warmup_steps=warmup_steps,
    val_max=lr_max,
    val_min=lr_min,
    end_step=200000,
)

param_groups = build_param_groups_with_scaling(model, encoder_lr_scale)
optimizer = MultiScaleAdamW(
    params=param_groups,
    lr=transfer_scheduler,
    betas=(0.9, 0.98),
    eps=1e-8,
)

# CTC losses + metrics/decoders (main output only for now; inter CTC heads internal)
# Use CTCGreedySearchDecoder (or later beam) requiring a tokenizer; for LRW-AR we create a simple char tokenizer file externally.
# If you already have a BPE tokenizer path, set via env LRWARCTC_TOKENIZER_PATH.
_tokenizer_path = os.environ.get("LRWARCTC_TOKENIZER_PATH", "datasets/LRW-AR/tokenizer_char.model")

class SimpleCharGreedyDecoder(nn.Module):
    def __init__(self, idx2char, blank=0):
        super(SimpleCharGreedyDecoder, self).__init__()
        self.idx2char = idx2char
        self.blank = blank
    def forward(self, outputs, from_logits=True):
        if not isinstance(outputs, (list, tuple)) or len(outputs)!=2:
            return []
        first, second = outputs
        # logits path
        if from_logits:
            logits, lengths = first, second
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            tokens = logits.softmax(dim=-1).argmax(dim=-1)  # (B,T)
            decoded=[]
            for b in range(tokens.size(0)):
                seq=[]; prev=None
                L=int(lengths[b])
                for t in range(L):
                    k=int(tokens[b,t])
                    if k==self.blank or k==prev: continue
                    if k in self.idx2char:
                        seq.append(self.idx2char[k])
                    prev=k
                decoded.append(''.join(seq))
            return decoded
        # targets path: first = y (1D), second = y_len (B,)
        y, y_len = first, second
        decoded=[]; off=0
        for L in y_len.tolist():
            seg=y[off:off+L].tolist(); off+=L
            decoded.append(''.join(self.idx2char.get(i,'') for i in seg))
        return decoded

_char_decoder = SimpleCharGreedyDecoder(idx2char=idx2char, blank=blank_idx)

model.compile(
    losses=nnet.CTCLoss(blank=blank_idx, zero_infinity=True),
    metrics={"outputs": [nnet.WordErrorRate(), nnet.CharacterErrorRate()]},
    decoders={"outputs": _char_decoder},
    optimizer=optimizer,
    loss_weights=loss_weights,
)
# After compile: set grad clipping & EMA manually
model.grad_max_norm = 1.0
model.set_ema(0.999)

# ---------------------------------------------------------------------------
# Encoder freezing logic
# ---------------------------------------------------------------------------

def _freeze_backbone(m):
    frozen=0
    for n,p in m.encoder.named_parameters():
        if n.startswith('head.'):
            continue  # keep head trainable to allow adaptation
        p.requires_grad = False
        frozen+=1
    return frozen

if freeze_encoder_epochs>0:
    frozen = _freeze_backbone(model)
    if not hasattr(model,'rank') or model.rank==0:
        print(f"[Transfer][LRW-AR-CTC] Frozen encoder params: {frozen} for first {freeze_encoder_epochs} epochs.")

_orig_on_epoch_end = model.on_epoch_end

def _tf_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, *a, **kw):
    res = _orig_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, *a, **kw)
    if epoch == freeze_encoder_epochs:
        unf=0
        for p in model.encoder.parameters():
            if not p.requires_grad:
                p.requires_grad=True; unf+=1
        if unf and (not hasattr(model,'rank') or model.rank==0):
            print(f"[Transfer][LRW-AR-CTC] Unfroze {unf} encoder params at epoch {epoch}.")
    return res
model.on_epoch_end = _tf_on_epoch_end  # type: ignore

# ---------------------------------------------------------------------------
# Transfer weight loading from LRS23 SWA Inter-CTC checkpoint
# ---------------------------------------------------------------------------
source_ckpt = os.environ.get("LRWARCTC_SOURCE_CKPT", "callbacks/LRS23/VO/EffConfInterCTC/checkpoints_swa-equal-90-100.ckpt")
_loaded=False
try:
    ck=torch.load(source_ckpt, map_location='cpu')
    state=ck.get('model_state_dict', {})
    msd=model.state_dict()
    loadable={}
    for k,v in state.items():
        if (k.startswith('encoder.front_end') or k.startswith('encoder.back_end')) and 'head' not in k:
            if k in msd and getattr(v,'shape',None)==getattr(msd[k],'shape',None):
                loadable[k]=v
    if loadable:
        msd.update(loadable)
        model.load_state_dict(msd, strict=False)
        _loaded=True
except Exception as e:
    if not hasattr(model,'rank') or model.rank==0:
        print('[Transfer][LRW-AR-CTC] Load failed:', e)
if _loaded and (not hasattr(model,'rank') or model.rank==0):
    print(f"[Transfer][LRW-AR-CTC] Loaded {len(loadable)} encoder tensors from {source_ckpt}")

# ---------------------------------------------------------------------------
# Checkpoint retention (best CER + last K)
# ---------------------------------------------------------------------------
callback_path = "callbacks/LRW-AR/VO/EffConfInterCTC_transfer"
_best_ckpt_cer = {'cer': 1e9, 'path': None}

_orig_on_epoch_end2 = model.on_epoch_end

def _retain_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, *a, **kw):
    res = _orig_on_epoch_end2(evaluate, save, log_figure, callback_path, epoch, *a, **kw)
    try:
        # After evaluation, pick CER metric if available in self.history
        cer_val=None
        try:
            cer_val = model.history['val']['output']['cer'][-1]
        except Exception:
            pass
        if cer_val is not None and cer_val < _best_ckpt_cer['cer']:
            # discover latest epoch checkpoint
            pattern = os.path.join(callback_path, f"checkpoints_epoch_{epoch}_step_*.ckpt")
            candidates = glob.glob(pattern)
            if candidates:
                new_ckpt = sorted(candidates)[-1]
                _best_ckpt_cer['cer']=cer_val
                _best_ckpt_cer['path']=new_ckpt
                with open(os.path.join(callback_path,'best_checkpoint.txt'),'w') as f:
                    f.write(f"cer {cer_val}\n{new_ckpt}\n")
                print(f"[Checkpoint Best] CER={cer_val:.4f} -> {new_ckpt}")
        # retention
        retain_k_env = os.environ.get('RETAIN_LAST_K')
        try:
            retain_k = int(retain_k_env) if retain_k_env is not None else 5
        except ValueError:
            retain_k = 5
        if retain_k < 0: retain_k = 0
        all_ckpts = glob.glob(os.path.join(callback_path,'checkpoints_epoch_*_step_*.ckpt'))
        meta=[]
        pat=re.compile(r'checkpoints_epoch_(\d+)_step_(\d+)')
        for pth in all_ckpts:
            m=pat.search(os.path.basename(pth))
            if not m: continue
            try:
                meta.append((int(m.group(1)), int(m.group(2)), pth))
            except: pass
        meta.sort()
        last_k=set(x[2] for x in meta[-retain_k:]) if retain_k else set()
        retain=set(last_k)
        if _best_ckpt_cer['path']: retain.add(_best_ckpt_cer['path'])
        for _,_,pth in meta:
            if pth not in retain:
                try: os.remove(pth)
                except: pass
    except Exception as e:
        print('[Checkpoint Retention] skipped:', e)
    return res

model.on_epoch_end = _retain_on_epoch_end  # type: ignore

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
precision = torch.float16 if torch.cuda.is_available() else torch.float32
accumulated_steps = 1
eval_training = False
recompute_metrics = True
log_figure_period_epoch = 1
eval_period_epoch = 1
saving_period_epoch = 1
verbose_eval = 1
epochs = _epochs

# Adjust scheduler end_step after dataset length known
try:
    total_steps = epochs * len(training_dataset)
    model.optimizer.scheduler.end_step = total_steps
except Exception:
    pass
