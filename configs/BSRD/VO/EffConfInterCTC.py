"""Visual-only Efficient Conformer Inter-CTC config for BSRD (character CTC).

Prerequisites:
    Run: python scripts/prepare_bsrd.py --root datasets/BSRD --align --landmarks mp --prepare
    which produces indices under datasets/BSRD/indices and a char vocab JSON.

Recent Additions (Adaptive Bundle):
    - Auxiliary inter-CTC loss weight WARMUP (ramps early to avoid over-guiding shallow heads)
    - Entropy regularization SCHEDULE (decays from max to min over epochs)
    - Stricter LM gating: requires blank ratio metric (won't gate if metric missing)
    - Stronger early curriculum default (lower initial fraction, extended epochs)
    - Token diversity diagnostic metric (non-blank unique token ratio)
    - Expanded decoders list to align with metrics (beam kept for WER only)
"""

import json
import torch
import nnet
from nnet import datasets, collate_fn
import torch.nn as nn
import math
import os
import types
from functools import lru_cache

_use_ctcdecode = True  # Set False to force fallback Python beam
ctcdecoder = None
if _use_ctcdecode:
    try:
        from ctcdecode import CTCBeamDecoder as _NativeCTCBeamDecoder  # local ctcdecode module
        ctcdecoder = _NativeCTCBeamDecoder
    except Exception:
        ctcdecoder = None

# ---------------------------------------------------------------------------
# Paths / Artifacts
# ---------------------------------------------------------------------------
callback_path = "callbacks/BSRD/VO/EffConfInterCTC"
indices_root = "datasets/BSRD/indices"
vocab_char2idx = "datasets/BSRD/vocab/char2idx.json"

# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------
try:
    with open(vocab_char2idx, 'r', encoding='utf-8') as f:
        char2idx = json.load(f)
    vocab_size = len(char2idx) + 1  # +1 for blank at index 0
except Exception:
    char2idx = {}
    vocab_size = 1  # placeholder so import does not fail before preparation

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
batch_size = int(os.environ.get("BSRD_BATCH", 8))
ctc_collate = collate_fn.CollateCTC()

training_dataset = datasets.BSRDCTC(
    batch_size=batch_size,
    collate_fn=ctc_collate,
    root="datasets",
    mode="train",
    indices_path=f"{indices_root}/train.pt",
    vocab_path=vocab_char2idx,
    prepared_only=True
)
evaluation_dataset = datasets.BSRDCTC(
    batch_size=batch_size,
    collate_fn=ctc_collate,
    root="datasets",
    mode="val",
    indices_path=f"{indices_root}/val.pt",
    vocab_path=vocab_char2idx,
    prepared_only=True
)

# Optionally expose test dataset (not automatically used in training loop)
test_dataset = datasets.BSRDCTC(
    batch_size=batch_size,
    collate_fn=ctc_collate,
    root="datasets",
    mode="test",
    indices_path=f"{indices_root}/test.pt",
    vocab_path=vocab_char2idx,
    prepared_only=True
)

"""Bundle C adjustments:
 - Enable intermediate CTC heads to improve early supervision.
 - Switch to CosineAnnealing LR scheduler with moderate warmup.
 - Adaptive LM weight ramp (alpha/beta grow over first few epochs) by patching model.on_epoch_end.
"""

# ---------------------------------------------------------------------------
# Model (enable inter-CTC heads)
#   Added third deeper inter-CTC head (block 12) with tiny weight (feature D).
# ---------------------------------------------------------------------------
interctc_blocks = [4, 8, 12]
model = nnet.VisualEfficientConformerInterCTC(vocab_size=vocab_size, interctc_blocks=interctc_blocks)

# ---------------------------------------------------------------------------
# Curriculum: length-based subset for early epochs
#   Strategy: For first CURRIC_NUM_EPOCHS epochs, restrict to a fraction of shortest
#   samples (by video frames). Fraction linearly increases from CURRIC_MIN_FRAC to 1.0.
#   Environment overrides:
#     BSRD_CURRIC_ENABLE (default 1)
#     BSRD_CURRIC_EPOCHS (default 5)
#     BSRD_CURRIC_MIN_FRAC (default 0.3)
# ---------------------------------------------------------------------------
CURRIC_ENABLE = int(os.environ.get("BSRD_CURRIC_ENABLE", 1)) == 1
CURRIC_EPOCHS = int(os.environ.get("BSRD_CURRIC_EPOCHS", 8))  # extended for longer gradual exposure
CURRIC_MIN_FRAC = float(os.environ.get("BSRD_CURRIC_MIN_FRAC", 0.15))  # start from fewer (shorter) samples

if CURRIC_ENABLE:
    # Precompute lengths once
    lengths = []
    for item in training_dataset.items:
        # Prefer prepared frame tensor length (T) stored as 'T' else load metadata
        L = item.get('T')
        if L is None and item.get('prepared'):
            try:
                meta = torch.load(item['prepared'], map_location='cpu')
                L = meta.get('frames').shape[0] if isinstance(meta, dict) and 'frames' in meta else None
            except Exception:
                L = None
        lengths.append(L if L is not None else 0)
    # Sort indices by length
    sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])

    @lru_cache(maxsize=None)
    def curriculum_indices(epoch:int):
        if epoch >= CURRIC_EPOCHS:
            return list(range(len(training_dataset.items)))
        # Linear frac growth
        frac = CURRIC_MIN_FRAC + (1.0 - CURRIC_MIN_FRAC) * (epoch / max(1, CURRIC_EPOCHS - 1))
        keep = max(1, int(len(sorted_idx) * min(frac, 1.0)))
        return sorted_idx[:keep]

    orig_getitem = training_dataset.__getitem__
    def _curric_getitem(self, idx):
        if getattr(self, '_curric_active', None) is not None:
            idx = self._curric_active[idx]
        return orig_getitem(idx)
    training_dataset.__getitem__ = types.MethodType(_curric_getitem, training_dataset)

    def _apply_curriculum(epoch:int):
        # Preserve original full list once
        if not hasattr(training_dataset, '_orig_items'):
            training_dataset._orig_items = list(training_dataset.items)
        # After curriculum phase restore and exit
        if epoch >= CURRIC_EPOCHS:
            if hasattr(training_dataset, '_curriculum_active_items'):
                training_dataset.items = training_dataset._orig_items
                delattr(training_dataset, '_curriculum_active_items')
            return
        active_idx = curriculum_indices(epoch)
        training_dataset._curriculum_active_items = [training_dataset._orig_items[i] for i in active_idx]
        training_dataset.items = training_dataset._curriculum_active_items
        print(f"[Curriculum] Epoch {epoch}: using {len(active_idx)}/{len(training_dataset._orig_items)} samples ({100*len(active_idx)/len(training_dataset._orig_items):.1f}%)")
else:
    def _apply_curriculum(epoch:int):
        return

# Inline minimal greedy decoder for raw character vocab (no SentencePiece model)
class _GreedyCharDecoder(nn.Module):
    def __init__(self, idx2char, blank=0):
        super().__init__()
        self.idx2char = idx2char
        self.blank = blank
    def forward(self, outputs, from_logits=True):
        """Decode either model logits or raw CTC targets (greedy)."""
        with torch.no_grad():
            if from_logits:
                logits, lengths = outputs
                if 'BLANK_DECODE_SHIFT' in globals() and BLANK_DECODE_SHIFT > 0 and logits.size(-1) > 0:
                    logits = logits.clone(); logits[...,0] -= BLANK_DECODE_SHIFT
                pred_ids = logits.argmax(dim=-1)
                strings = []
                for b in range(pred_ids.size(0)):
                    seq = pred_ids[b, :lengths[b]].tolist()
                    collapsed = []
                    prev = None
                    for t in seq:
                        if t == prev:
                            continue
                        collapsed.append(t)
                        prev = t
                    chars = [self.idx2char[i] for i in collapsed if i != self.blank and i in self.idx2char]
                    strings.append("".join(chars))
                return strings
            else:
                labels, lengths = outputs
                if labels.dim() == 0:
                    return []
                strings = []
                offset = 0
                for L in lengths.tolist():
                    seq = labels[offset:offset+L].tolist()
                    offset += L
                    chars = [self.idx2char[i] for i in seq if i != self.blank and i in self.idx2char]
                    strings.append("".join(chars))
                return strings

class _FallbackPrefixBeam(nn.Module):
    """Original Python prefix beam search (fallback)."""
    def __init__(self, idx2char, blank=0, beam_size=5):
        super().__init__()
        self.idx2char = idx2char
        self.blank = blank
        self.beam_size = beam_size
    def _log_sum_exp_pair(self, a, b):
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        if a > b:
            return a + math.log1p(math.exp(b - a))
        return b + math.log1p(math.exp(a - b))
    def forward(self, outputs, from_logits=True):
        with torch.no_grad():
            if from_logits:
                logits, lengths = outputs
                log_probs = torch.log_softmax(logits, dim=-1)
                B, T, V = log_probs.size()
                out = []
                for b in range(B):
                    L = int(lengths[b])
                    beam = {(): (0.0, -math.inf)}
                    for t in range(L):
                        probs_t = log_probs[b, t]
                        next_beam = {}
                        for prefix, (pb, pnb) in beam.items():
                            p_blank = probs_t[self.blank].item()
                            nb_pb, nb_pnb = next_beam.get(prefix, (-math.inf, -math.inf))
                            nb_pb = self._log_sum_exp_pair(nb_pb, self._log_sum_exp_pair(pb + p_blank, pnb + p_blank))
                            next_beam[prefix] = (nb_pb, nb_pnb)
                            for c in range(V):
                                if c == self.blank:
                                    continue
                                p_c = probs_t[c].item()
                                last = prefix[-1] if prefix else None
                                new_prefix = prefix if c == last else prefix + (c,)
                                nb_pb2, nb_pnb2 = next_beam.get(new_prefix, (-math.inf, -math.inf))
                                if c == last:
                                    nb_pnb2 = self._log_sum_exp_pair(nb_pnb2, pb + p_c)
                                else:
                                    nb_pnb2 = self._log_sum_exp_pair(nb_pnb2, self._log_sum_exp_pair(pb + p_c, pnb + p_c))
                                next_beam[new_prefix] = (nb_pb2, nb_pnb2)
                        # prune
                        beam_items = []
                        for prefix, (pb, pnb) in next_beam.items():
                            score = self._log_sum_exp_pair(pb, pnb)
                            beam_items.append((score, prefix, pb, pnb))
                        beam_items.sort(key=lambda x: x[0], reverse=True)
                        beam = {p: (pb, pnb) for _, p, pb, pnb in beam_items[: self.beam_size]}
                    best = max(beam.items(), key=lambda kv: self._log_sum_exp_pair(kv[1][0], kv[1][1]))[0]
                    chars = [self.idx2char[i] for i in best if i != self.blank and i in self.idx2char]
                    out.append("".join(chars))
                return out
            else:
                labels, lengths = outputs
                if labels.dim() == 0:
                    return []
                res = []
                off = 0
                for L in lengths.tolist():
                    seq = labels[off:off+L].tolist(); off += L
                    chars = [self.idx2char[i] for i in seq if i != self.blank and i in self.idx2char]
                    res.append("".join(chars))
                return res

class _CTCDecodeWrapper(nn.Module):
    """Wrapper that uses native ctcdecode if available, else fallback prefix beam.

    Args:
        idx2char: mapping id->char
        blank: blank index
        beam_width: beam width for native decoder
        lm_path: optional KenLM path
        alpha,beta: LM + word insertion weights for native decoder
        cutoff_top_n, cutoff_prob: pruning knobs
    """
    def __init__(self, idx2char, blank=0, beam_width=25, lm_path=None, alpha=0.2, beta=0.5, cutoff_top_n=40, cutoff_prob=1.0, num_processes=4, fallback_beam_size=10,
                 max_alpha=0.6, max_beta=1.2, ramp_epochs=5,
                 gate_after_epoch:int=0, gate_blank_threshold:float=None,
                 beam_start_epoch:int=0, blank_decode_shift:float=0.0):
        super().__init__()
        self.idx2char = idx2char
        self.blank = blank
        self.native = None
        self.use_native = False
        self._base_alpha = alpha
        self._base_beta = beta
        self._max_alpha = max_alpha
        self._max_beta = max_beta
        self._ramp_epochs = max(1, ramp_epochs)
        self._current_epoch = 0
        self._lm_path = lm_path
        self._beam_width = beam_width
        self._cutoff_top_n = cutoff_top_n
        self._cutoff_prob = cutoff_prob
        self._num_processes = num_processes
        self._fallback_beam_size = fallback_beam_size
        # Gating params (feature A)
        self._gate_after_epoch = gate_after_epoch
        self._gate_blank_threshold = gate_blank_threshold
        self._gated_active = False
        # Force greedy decoding until this (1-based) epoch reached
        self._beam_start_epoch = beam_start_epoch
        # Decode-time blank logit suppression (does not affect training loss)
        self._blank_decode_shift = blank_decode_shift
        self._build_native(alpha=alpha, beta=beta)
        if not self.use_native:
            self.fallback = _FallbackPrefixBeam(idx2char=idx2char, blank=blank, beam_size=fallback_beam_size)

    def _compute_alpha_beta(self):
        # Linear ramp over epochs
        factor = min(1.0, self._current_epoch / self._ramp_epochs)
        a = self._base_alpha + (self._max_alpha - self._base_alpha) * factor
        b = self._base_beta + (self._max_beta - self._base_beta) * factor
        return a, b

    def _build_native(self, alpha, beta):
        if ctcdecoder is not None:
            try:
                labels = ["<blank>"] + [self.idx2char[i] for i in sorted(self.idx2char.keys()) if i != 0]
                self.native = ctcdecoder(labels=labels, model_path=self._lm_path if self._lm_path else None, alpha=alpha, beta=beta, beam_width=self._beam_width, cutoff_top_n=self._cutoff_top_n, cutoff_prob=self._cutoff_prob, num_processes=self._num_processes, blank_id=0, log_probs_input=False)
                self.use_native = True
            except Exception:
                self.native = None
        else:
            self.native = None

    def update_epoch(self, epoch: int, latest_blank_ratio:float=None):
        if epoch == self._current_epoch:
            return
        self._current_epoch = epoch
        # Evaluate gating logic
        if not self._gated_active:
            gate_epoch_ok = (self._gate_after_epoch is None) or (epoch >= self._gate_after_epoch)
            # Require blank ratio measurement if threshold provided (avoid gating with unknown metric)
            if self._gate_blank_threshold is not None:
                if latest_blank_ratio is None:
                    gate_blank_ok = False
                else:
                    gate_blank_ok = latest_blank_ratio <= self._gate_blank_threshold
            else:
                gate_blank_ok = True
            if gate_epoch_ok and gate_blank_ok:
                self._gated_active = True
                try:
                    msg_blank = f"{latest_blank_ratio:.4f}" if latest_blank_ratio is not None else "n/a"
                    print(f"[LM Gate] Activating LM/beam at epoch {epoch} (blank_ratio={msg_blank})")
                except Exception:
                    print(f"[LM Gate] Activating LM/beam at epoch {epoch}")
        if (self._lm_path and ctcdecoder is not None) and self._gated_active:
            # Rebuild decoder with new alpha/beta
            a, b = self._compute_alpha_beta()
            self._build_native(alpha=a, beta=b)

    def forward(self, outputs, from_logits=True):
        if not from_logits:
            # Ground-truth decode (just map labels)
            labels, lengths = outputs
            if labels.dim() == 0:
                return []
            res = []
            off = 0
            for L in lengths.tolist():
                seq = labels[off:off+L].tolist(); off += L
                chars = [self.idx2char[i] for i in seq if i != self.blank and i in self.idx2char]
                res.append("".join(chars))
            return res
        logits, lengths = outputs
        with torch.no_grad():
            # Apply decode-time blank suppression
            if self._blank_decode_shift > 0 and logits.size(-1) > 0:
                logits = logits.clone(); logits[...,0] -= self._blank_decode_shift
            # Force greedy decoding until beam start epoch
            if self._current_epoch < self._beam_start_epoch:
                pred_ids = logits.argmax(dim=-1)
                batch_strings = []
                for b in range(pred_ids.size(0)):
                    L = int(lengths[b]); seq = pred_ids[b, :L].tolist()
                    collapsed = []
                    prev = None
                    for t in seq:
                        if t == prev:
                            continue
                        if t != 0:
                            collapsed.append(t)
                        prev = t
                    chars = [self.idx2char[i] for i in collapsed if i in self.idx2char]
                    batch_strings.append("".join(chars))
                return batch_strings
            if self.use_native and self.native is not None:
                probs = torch.softmax(logits, dim=-1)  # native expects probs by default (log_probs_input=False)
                beam_results, beam_scores, timesteps, out_lens = self.native.decode(probs, seq_lens=lengths)
                batch_strings = []
                for b in range(beam_results.size(0)):
                    L = out_lens[b,0].item()
                    ids = beam_results[b,0,:L].tolist()
                    chars = []
                    for i in ids:
                        # native blank index 0 corresponds to our blank 0 shift for labels list
                        if i == 0:
                            continue
                        # Our idx2char starts from 1; we added <blank> at 0, so i maps directly if char2idx built that way
                        if i in self.idx2char:
                            chars.append(self.idx2char[i])
                    batch_strings.append("".join(chars))
                return batch_strings
            else:
                return self.fallback((logits, lengths), from_logits=True)

idx2char = {v: k for k, v in char2idx.items()}
greedy_decoder = _GreedyCharDecoder(idx2char=idx2char, blank=0)

# Optional KenLM path (set to actual .arpa/.binary path if available)
kenlm_path = os.environ.get("BSRD_KENLM_PATH", None)
BEAM_START_EPOCH = int(os.environ.get("BSRD_BEAM_START_EPOCH", 5))
BLANK_DECODE_SHIFT = float(os.environ.get("BSRD_DECODE_BLANK_SHIFT", 0.0))
beam_decoder = _CTCDecodeWrapper(
    idx2char=idx2char,
    blank=0,
    beam_width=25,
    lm_path=kenlm_path,
    alpha=0.2,   # start low
    beta=0.5,    # start low
    max_alpha=0.6,
    max_beta=1.2,
    ramp_epochs=int(os.environ.get("BSRD_LM_RAMP_EPOCHS", 5)),
    fallback_beam_size=5,
    gate_after_epoch=int(os.environ.get("BSRD_LM_GATE_EPOCH", 5)),
    gate_blank_threshold=float(os.environ.get("BSRD_LM_GATE_BLANK", 0.92)) if os.environ.get("BSRD_LM_GATE_BLANK") is not None else None,
    beam_start_epoch=BEAM_START_EPOCH,
    blank_decode_shift=BLANK_DECODE_SHIFT
)

# ---------------------------------------------------------------------------
# Metrics / Schedulers
# ---------------------------------------------------------------------------

class _BlankRatioMetric(nn.Module):
    """Blank token proportion before CTC collapse.
    Called as metric(targets, outputs); outputs = [logits, lengths]."""
    def __init__(self, blank=0):
        super().__init__(); self.blank = blank; self.name = "blank_ratio"
    def forward(self, targets, outputs):
        try:
            logits, lengths = outputs
        except Exception:
            return torch.tensor(0.0)
        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1)
            blank_count = 0; total = 0
            for b in range(pred_ids.size(0)):
                L = int(lengths[b]); seq = pred_ids[b, :L]
                total += L; blank_count += (seq == self.blank).sum().item()
            return torch.tensor(0.0 if total == 0 else blank_count / total, device=logits.device)

class _LengthStatsMetric(nn.Module):
    """Mean predicted length / target length ratio.
    Called as metric(targets, outputs)."""
    def __init__(self):
        super().__init__(); self.name = "len_ratio"
    def forward(self, targets, outputs):
        try:
            logits, logits_len = outputs
            y, y_len = targets
        except Exception:
            return torch.tensor(0.0)
        with torch.no_grad():
            y_len_f = y_len.to(logits_len.device).float().clamp(min=1)
            return (logits_len.float() / y_len_f).mean()

class _EntropyRegularizedCTCLoss(nn.Module):
    """Wrap CTCLoss adding negative entropy penalty to encourage exploration (feature C).

    loss_total = ctc_loss + lambda * (-H(mean_prob)) where H is entropy averaged across time/batch.
    Set BSRD_ENTROPY_LAMBDA env to weight (default 0 disables)."""
    def __init__(self, base_loss, entropy_lambda:float=0.0):
        super().__init__()
        self.base_loss = base_loss
        self.entropy_lambda = entropy_lambda
    def forward(self, targets, outputs):
        base = self.base_loss(targets, outputs)
        if self.entropy_lambda <= 0:
            return base
        logits, logits_len = outputs
        with torch.no_grad():
            # mask to valid timesteps per sample
            max_T = logits.size(1)
            mask = torch.arange(max_T, device=logits.device).unsqueeze(0) < logits_len.unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)
        probs_masked = probs * mask.unsqueeze(-1)
        # average distribution over all valid positions
        denom = mask.sum().clamp(min=1).float()
        mean_dist = probs_masked.sum(dim=(0,1)) / denom
        entropy = - (mean_dist * (mean_dist.clamp_min(1e-8).log())).sum()
        return base + self.entropy_lambda * (-entropy)

# Allow overriding warmup via env for experimentation
# Further lower LR peak & extend warmup (feature E)
warmup_steps = int(os.environ.get("BSRD_WARMUP", 12000))  # extended further
val_max = float(os.environ.get("BSRD_LR_MAX", 3.5e-5))     # lower peak LR
val_min = float(os.environ.get("BSRD_LR_MIN", 8e-6))       # slightly lower floor
lr_sched = nnet.schedulers.CosineAnnealingScheduler(warmup_steps=warmup_steps, val_max=val_max, val_min=val_min, end_step=200000)

# Adjust loss weighting: give stronger weight to final head (encourages primary sequence quality)
# Inter-CTC heads still provide gradients but at reduced influence. Order follows internal loss ordering.
# Prepare loss with entropy regularization (feature C)
_entropy_lambda_env = os.environ.get("BSRD_ENTROPY_LAMBDA", None)
_entropy_lambda = float(_entropy_lambda_env) if _entropy_lambda_env is not None else 0.0
base_ctc = nnet.CTCLoss(blank=0, zero_infinity=True, assert_shorter=False)
loss_main = _EntropyRegularizedCTCLoss(base_ctc, entropy_lambda=_entropy_lambda)

# Entropy lambda schedule parameters (epoch-based) — disabled by default for early peaking
ENT_LAMBDA_MAX = float(os.environ.get("BSRD_ENTROPY_LAMBDA_MAX", 0.0))
ENT_LAMBDA_MIN = float(os.environ.get("BSRD_ENTROPY_LAMBDA_MIN", 0.0))
ENT_LAMBDA_DECAY_EPOCHS = int(os.environ.get("BSRD_ENTROPY_LAMBDA_DECAY_EPOCHS", 1))

# Auxiliary loss weight warmup (step-based) for inter-CTC heads
approx_total_items = len(training_dataset.items)
approx_steps_per_epoch = max(1, approx_total_items // batch_size)
AUX_WARMUP_EPOCHS = int(os.environ.get("BSRD_AUX_WARMUP_EPOCHS", 5))
aux_warmup_steps = max(1, AUX_WARMUP_EPOCHS * approx_steps_per_epoch)

# Target weights after warmup
_w_ctc_3, _w_ctc_7, _w_ctc_11, _w_final = 0.12, 0.12, 0.06, 0.70
loss_weights = {
    "outputs": nnet.schedulers.ConstantScheduler(_w_final),
    "ctc_3": nnet.schedulers.LinearDecayScheduler(0.0, _w_ctc_3, aux_warmup_steps),
    "ctc_7": nnet.schedulers.LinearDecayScheduler(0.0, _w_ctc_7, aux_warmup_steps),
    "ctc_11": nnet.schedulers.LinearDecayScheduler(0.0, _w_ctc_11, aux_warmup_steps)
}

# Optional epoch-based delay (AUX_DELAY_EPOCHS) before auxiliary heads contribute.
AUX_DELAY_EPOCHS = int(os.environ.get("BSRD_AUX_DELAY_EPOCHS", 0))
if AUX_DELAY_EPOCHS > 0:
    def _wrap_delay(sched):
        if not hasattr(sched, 'get_val_step'):
            return sched
        orig = sched.get_val_step
        def _gvs(step):
            ce = getattr(model, '_current_epoch_int', 1)  # 1-based epoch
            if (ce - 1) < AUX_DELAY_EPOCHS:
                return 0.0
            return orig(step)
        sched.get_val_step = _gvs
        return sched
    for _k in ['ctc_3','ctc_7','ctc_11']:
        loss_weights[_k] = _wrap_delay(loss_weights[_k])

# Token diversity metric (unique non-blank tokens / non-blank count) for diagnostic
class _TokenDiversityMetric(nn.Module):
    def __init__(self, blank=0):
        super().__init__(); self.blank = blank; self.name = "token_diversity"
    def forward(self, targets, outputs):
        try:
            logits, lengths = outputs
        except Exception:
            return torch.tensor(0.0)
        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1)
            uniq_total = 0; nonblank_total = 0
            for b in range(pred_ids.size(0)):
                L = int(lengths[b]); seq = pred_ids[b, :L]
                nb = seq[seq != self.blank]
                nonblank_total += nb.numel()
                if nb.numel() > 0:
                    uniq_total += nb.unique().numel()
            return torch.tensor(0.0 if nonblank_total == 0 else uniq_total / nonblank_total, device=logits.device)

metrics_outputs = [
    nnet.CharacterErrorRate(),
    nnet.WordErrorRate(),
    _BlankRatioMetric(),
    _LengthStatsMetric(),
    _TokenDiversityMetric()
]

# Decoders list aligned with metrics (beam only for WER); others use greedy
decoders_outputs = [
    greedy_decoder,  # CER
    beam_decoder,    # WER
    None,            # blank ratio (raw outputs)
    None,            # length stats
    None             # token diversity
]

HEAD_LR_MULT = float(os.environ.get("BSRD_HEAD_LR_MULT", 3.0))
# Split params into encoder vs head for LR scaling
encoder_params, head_params = [], []
for n,p in model.named_parameters():
    if n.startswith('encoder.head.'):
        head_params.append(p)
    else:
        encoder_params.append(p)
base_lr_sched = lr_sched
optimizer = nnet.optimizers.Adam(
    params=[
        {"params": encoder_params},
        {"params": head_params, "lr": 0.0}  # will be scaled after scheduler step
    ],
    lr=base_lr_sched, betas=(0.9,0.98), eps=1e-9, weight_decay=1e-6
)
model.compile(
    losses=loss_main,
    loss_weights=loss_weights,
    optimizer=optimizer,
    metrics={"outputs": metrics_outputs},
    decoders={"outputs": decoders_outputs}
)

# Initialize head bias to discourage blank dominance (blank index 0 gets negative bias)
try:
    if hasattr(model.encoder, 'head') and hasattr(model.encoder.head, 'bias') and model.encoder.head.bias is not None:
        with torch.no_grad():
            model.encoder.head.bias.data.fill_(0.0)
            if model.encoder.head.bias.data.shape[0] > 0:
                model.encoder.head.bias.data[0] = -1.5
        print('[Init] Head bias initialized (blank=-1.5, others=0)')
except Exception as _e:
    print('[Init] Head bias init skipped:', _e)

# Wrap optimizer step to apply head LR multiplier each step
_orig_step = model.optimizer.step
def _scaled_step(closure=None):
    r = _orig_step(closure)
    try:
        if len(model.optimizer.param_groups) > 1:
            base_lr = model.optimizer.param_groups[0]['lr']
            model.optimizer.param_groups[1]['lr'] = base_lr * HEAD_LR_MULT
    except Exception:
        pass
    return r
model.optimizer.step = _scaled_step

# Optional: log current loss weight schedule values each epoch
_LOG_AUX = int(os.environ.get("BSRD_LOG_AUX_WEIGHTS", "0")) == 1
def _log_current_loss_weights(epoch:int):
    if not _LOG_AUX:
        return
    try:
        lw = model.loss_weights
        vals = {k: (v.get_val_step(model.model_step + 1) if hasattr(v, 'get_val_step') else str(v)) for k,v in lw.items()}
        print(f"[AuxWeights] Epoch {epoch}: " + ", ".join(f"{k}={vals[k]:.4f}" for k in sorted(vals)))
    except Exception as e:
        print('[AuxWeights] logging failed:', e)

# Enable gradient clipping (global norm) via model attribute used in training loop
model.grad_max_norm = float(os.environ.get("BSRD_GRAD_CLIP", 5.0))

# Diagnostics: enable logits debug (blank vs non-blank) when BSRD_LOGITS_DEBUG=1
if int(os.environ.get("BSRD_LOGITS_DEBUG", "0")) == 1:
    def _logits_debug_hook(module, inputs, output):
        try:
            logits = output
            if isinstance(output, (list, tuple)):
                logits = output[0]
            if logits.dim() != 3:
                return
            with torch.no_grad():
                # Compute softmax on a small random subset to reduce cost
                B,T,V = logits.shape
                probs = torch.softmax(logits[:min(2,B)], dim=-1)
                blank_mean = probs[...,0].mean().item()
                nonblank_mean = probs[...,1:].mean().item() if V>1 else 0.0
                unique_tokens = logits.argmax(-1)[:min(2,B)].unique().numel()
                print(f"[LogitsDbg] blank_mean={blank_mean:.4f} nonblank_mean={nonblank_mean:.4f} uniq_tokens={unique_tokens}")
        except Exception:
            pass
    try:
        model.encoder.head.register_forward_hook(_logits_debug_hook)
        print('[LogitsDbg] Hook registered on encoder head.')
    except Exception as _e:
        print('[LogitsDbg] Hook registration failed:', _e)

# Patch on_epoch_end to update LM alpha/beta ramp
_orig_on_epoch_end = model.on_epoch_end
_best_ckpt_cer = {"value": None, "path": None}
def _patched_on_epoch_end(self, evaluate, save, log_figure, callback_path, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics):
    # 1. Update LM gating (blank ratio from latest training metrics if present)
    latest_blank = None
    try:
        if hasattr(self, 'metrics_history'):
            hist = self.metrics_history.get('outputs', []) if isinstance(self.metrics_history, dict) else []
            if hist:
                last = hist[-1]
                if isinstance(last, dict) and 'blank_ratio' in last:
                    latest_blank = float(last['blank_ratio'])
    except Exception:
        pass
    if hasattr(beam_decoder, 'update_epoch'):
        try:
            beam_decoder.update_epoch(epoch, latest_blank_ratio=latest_blank)
        except Exception as e:
            print('[Bundle C] beam alpha/beta update failed:', e)

    # 2. Delegate to original hook (this performs eval + save)
    eval_results = _orig_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics)

    # 3. Retention policy: keep best CER + last K chronological checkpoints
    if evaluate and save and callback_path is not None and self.rank == 0:
        try:
            import glob, os, re
            # Initialize best from file if unknown
            global _best_ckpt_cer
            if _best_ckpt_cer['value'] is None:
                best_file = os.path.join(callback_path, 'best_checkpoint.txt')
                if os.path.isfile(best_file):
                    try:
                        with open(best_file, 'r') as fbf:
                            lines = [l.strip() for l in fbf.readlines() if l.strip()]
                        if len(lines) >= 2 and lines[0].startswith('cer'):
                            parts = lines[0].split()
                            if len(parts) >= 2:
                                _best_ckpt_cer['value'] = float(parts[1])
                                _best_ckpt_cer['path'] = lines[1]
                                print(f"[Checkpoint Init] Loaded prior best CER={_best_ckpt_cer['value']:.4f}")
                    except Exception:
                        pass

            # Extract CER from eval results this epoch (if present)
            cer_val = None
            if isinstance(eval_results, list) and eval_results:
                first = eval_results[0]
                metrics_dict = first.get('metrics', {}) if isinstance(first, dict) else {}
                cer_raw = metrics_dict.get('cer', None)
                if cer_raw is not None:
                    cer_val = float(cer_raw.item() if hasattr(cer_raw, 'item') else cer_raw)

            # Collect checkpoints for this epoch to identify the newly created one
            new_ckpt = None
            epoch_pattern = os.path.join(callback_path, f"checkpoints_epoch_{epoch}_step_*.ckpt")
            epoch_ckpts = sorted(glob.glob(epoch_pattern), key=os.path.getmtime)
            if epoch_ckpts:
                new_ckpt = epoch_ckpts[-1]

            # Update best if improved
            if cer_val is not None and new_ckpt is not None:
                best_val = _best_ckpt_cer['value']
                improved = (best_val is None) or (cer_val < best_val)
                if improved:
                    _best_ckpt_cer['value'] = cer_val
                    _best_ckpt_cer['path'] = new_ckpt
                    try:
                        with open(os.path.join(callback_path, 'best_checkpoint.txt'), 'w') as f:
                            f.write(f"cer {cer_val}\n{new_ckpt}\n")
                    except Exception:
                        pass
                    print(f"[Checkpoint Best] CER={cer_val:.4f} -> {new_ckpt}")

            # Apply retention (best + last K)
            retain_k_env = os.environ.get('RETAIN_LAST_K') or os.environ.get('BSRD_RETAIN_LAST_CKPTS')
            try:
                retain_k = int(retain_k_env) if retain_k_env is not None else 5
            except ValueError:
                retain_k = 5
            if retain_k < 0:
                retain_k = 0

            # Build chronological ordering by (epoch, step)
            all_ckpts = glob.glob(os.path.join(callback_path, 'checkpoints_epoch_*_step_*.ckpt'))
            meta = []
            pat = re.compile(r'checkpoints_epoch_(\d+)_step_(\d+)')
            for p in all_ckpts:
                m = pat.search(os.path.basename(p))
                if not m:
                    continue
                try:
                    meta.append((int(m.group(1)), int(m.group(2)), p))
                except Exception:
                    continue
            meta.sort()  # ascending chronological
            last_k = set(x[2] for x in meta[-retain_k:]) if retain_k else set()
            retain = set(last_k)
            if _best_ckpt_cer['path']:
                retain.add(_best_ckpt_cer['path'])
            deleted = 0
            for _,_,p in meta:
                if p not in retain:
                    try:
                        os.remove(p)
                        deleted += 1
                    except Exception:
                        pass
            if deleted > 0:
                print(f"[Checkpoint Prune] Removed {deleted} old checkpoints; retained {len(retain)} (best + last {retain_k}).")
        except Exception as e:
            print(f"[Checkpoint Prune] Retention skipped: {e}")

    return eval_results
model.on_epoch_end = types.MethodType(_patched_on_epoch_end, model)

# Patch on_epoch_begin to apply curriculum subset before each epoch (epoch numbering starts at 1 in training loop usage)
_orig_on_epoch_begin = getattr(model, 'on_epoch_begin', None)
def _patched_on_epoch_begin(self, epoch):
    try:
        self._current_epoch_int = epoch
        _apply_curriculum(epoch-1)  # use zero-based for schedule
        # Update entropy lambda (epoch-1 is zero-based training progress before entering this epoch)
        zero_based = epoch - 1
        if ENT_LAMBDA_MAX > 0 and hasattr(loss_main, 'entropy_lambda'):
            if zero_based < ENT_LAMBDA_DECAY_EPOCHS:
                cur_lambda = ENT_LAMBDA_MAX - (ENT_LAMBDA_MAX - ENT_LAMBDA_MIN) * (zero_based / max(1, ENT_LAMBDA_DECAY_EPOCHS))
            else:
                cur_lambda = ENT_LAMBDA_MIN
            # Only update & optionally print when changed noticeably
            if abs(getattr(loss_main, 'entropy_lambda', 0.0) - cur_lambda) > 1e-6:
                loss_main.entropy_lambda = cur_lambda
                if epoch == 1 or zero_based % 2 == 0:
                    print(f"[EntropySched] Epoch {epoch} set entropy_lambda={cur_lambda:.4f}")
        _log_current_loss_weights(epoch)
    except Exception as e:
        print('[Curriculum] failed to apply:', e)
    if callable(_orig_on_epoch_begin):
        return _orig_on_epoch_begin(epoch)
model.on_epoch_begin = types.MethodType(_patched_on_epoch_begin, model)

# ---------------------------------------------------------------------------
# Transfer Learning: initialize encoder from LRS23 Inter-CTC SWA checkpoint
# Provided checkpoint: callbacks/LRS23/VO/EffConfInterCTC/checkpoints_swa-equal-90-100.ckpt
# Strategy: load all matching 'encoder.front_end.' and 'encoder.back_end.' weights, skip head.
# ---------------------------------------------------------------------------
lrs23_checkpoint = "callbacks/LRS23/VO/EffConfInterCTC/checkpoints_swa-equal-90-100.ckpt"
_transfer_loaded = False
try:
    ck = torch.load(lrs23_checkpoint, map_location="cpu")
    state = ck.get("model_state_dict", {})
    msd = model.state_dict()
    loadable = {}
    for k,v in state.items():
        if k.startswith("encoder.front_end") or k.startswith("encoder.back_end"):
            if 'head' in k:
                continue
            if k in msd and getattr(v,'shape',None)==getattr(msd[k],'shape',None):
                loadable[k] = v
    if loadable:
        msd.update(loadable)
        model.load_state_dict(msd, strict=False)
        _transfer_loaded = True
except Exception:
    _transfer_loaded = False


# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
epochs = 200  # extended training
accumulated_steps = 1
precision = torch.float16 if torch.cuda.is_available() else torch.float32
eval_training = False
recompute_metrics = True
early_stopping_metric = None  # can set to 'cer'
seed = 42

# Logging / evaluation cadence (fallback to CLI overrides in main.py)
log_figure_period_epoch = 1
eval_period_epoch = 1
saving_period_epoch = 1

