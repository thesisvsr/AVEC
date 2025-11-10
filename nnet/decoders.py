# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn

# Other
import os
import sentencepiece as spm
import importlib

# Neural Nets
from nnet.module import Module

# CTC Decode (optional)
try:
    from ctcdecode import CTCBeamDecoder
except Exception:
    # Leave unavailable silently; we'll raise a clear error if a CTC beam search is actually requested.
    CTCBeamDecoder = None  # type: ignore

###############################################################################
# Decoders
###############################################################################

class IdentityDecoder(nn.Module):

    def __init__(self):
        super(IdentityDecoder, self).__init__()

    def forward(self, outputs, from_logits=True):

        return outputs.tolist()

class ThresholdDecoder(nn.Module):

    def __init__(self, threshold=0.5):
        super(ThresholdDecoder, self).__init__()
        self.threshold = threshold

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = torch.where(outputs >= self.threshold, 1, 0).squeeze(dim=-1).tolist()
        else:
            tokens = outputs.tolist()

        return tokens

class ArgMaxDecoder(nn.Module):

    def __init__(self, axis=-1):
        super(ArgMaxDecoder, self).__init__()
        self.axis = axis

    def forward(self, outputs, from_logits=True):
        # Accept (logits, lengths) or [logits, lengths] structures
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 0:
                return []
            # Take the first element as logits when decoding predictions
            primary = outputs[0]
        else:
            primary = outputs

        if from_logits:
            # Softmax -> argmax along given axis
            if not torch.is_tensor(primary):
                # If already a list of indices just return it
                try:
                    return list(primary)
                except Exception:
                    return []
            tokens = primary.softmax(dim=self.axis).argmax(axis=self.axis).tolist()
        else:
            # Targets could be (labels, lengths) tuple; accept either raw tensor or container
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 1 and torch.is_tensor(outputs[0]):
                tgt = outputs[0]
            else:
                tgt = primary
            if torch.is_tensor(tgt):
                tokens = tgt.tolist()
            elif isinstance(tgt, (list, tuple)):
                tokens = list(tgt)
            else:
                try:
                    tokens = list(tgt)
                except Exception:
                    tokens = []
        return tokens

class ClassLabelToWordDecoder(nn.Module):

    """
    Decoder for classification (CE) models where outputs are class logits.
    - When from_logits=True: take argmax over classes and map indices to word strings.
    - When from_logits=False: inputs are integer class indices; map directly to word strings.

    Returns a Python list of strings (one per batch element), suitable for jiwer WER.
    """

    def __init__(self, words):
        super(ClassLabelToWordDecoder, self).__init__()
        # words: list[str] of length V where index aligns with class id
        self.words = list(words)

    def forward(self, outputs, from_logits=True):
        # outputs: (B, V) logits if from_logits else (B,) integer class ids
        if from_logits:
            if isinstance(outputs, (list, tuple)):
                # Safety: accept a single tensor in a list/tuple
                outputs = outputs[0]
            idx = outputs.softmax(dim=-1).argmax(dim=-1).detach().cpu().tolist()
        else:
            # Targets should be integer class indices tensor (B,)
            if torch.is_tensor(outputs):
                idx = outputs.detach().cpu().tolist()
            else:
                idx = list(outputs)

        # Map indices to words with bounds checking
        decoded = []
        V = len(self.words)
        for i in idx:
            if isinstance(i, (list, tuple)):
                # Unexpected shape; take first
                i = i[0]
            try:
                if 0 <= int(i) < V:
                    decoded.append(self.words[int(i)])
                else:
                    decoded.append("")
            except Exception:
                decoded.append("")
        return decoded

class CTCGreedySearchDecoder(nn.Module):

    def __init__(self, tokenizer_path, blank_token=0):
        super(CTCGreedySearchDecoder, self).__init__()

        # Load Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

        # Blank Token
        self.blank_token = blank_token

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = self.greedy_search(*outputs)
        else:
            tokens = outputs[0].tolist()

        return self.tokenizer.decode(tokens)

    def greedy_search(self, logits, logits_len):

        # Argmax (B, T, V) -> (B, T)
        preds = logits.argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):

            # Slice Preds
            preds_b = preds[b, :logits_len[b]]

            # Remove Consecutives
            preds_b = preds_b.unique_consecutive(dim=-1)

            # Remove Blanks
            preds_b = [token.item() for token in preds_b if token != self.blank_token]

            # Append Prediction
            batch_pred_list.append(preds_b)

        return batch_pred_list

class CTCBeamSearchDecoder(Module):

    """
    
        Parameter beam_alpha specifies amount of importance to place on the N-gram language model, 
        and beam_beta is a penalty term to consider the sequence length in the scores. 
        Larger alpha means more importance on the LM and less importance on the acoustic model. 
        Negative values for beta will give penalty to longer sequences and make the decoder to prefer shorter predictions, 
        while positive values would result in longer candidates.
    
    """

    def __init__(self, tokenizer_path, beam_size=16, ngram_path=None, ngram_tmp=1.0, ngram_alpha=0.6, ngram_beta=1.0, ngram_offset=100, neural_config_path=None, neural_checkpoint=None, neural_alpha=0.6, neural_beta=1.0, num_processes=8, test_time_aug=False):
        super(CTCBeamSearchDecoder, self).__init__()

        # Load Tokenizer
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

        # Params
        self.beam_size = beam_size
        self.test_time_aug = test_time_aug

        # Ngram
        self.ngram_path = ngram_path
        self.ngram_tmp = ngram_tmp
        self.ngram_alpha = ngram_alpha
        self.ngram_beta = ngram_beta
        self.ngram_offset = ngram_offset
        self.num_processes = num_processes

        # Neural Rescorer
        self.neural_alpha = neural_alpha
        self.neural_beta = neural_beta
        if neural_config_path is not None:
            neural_config = importlib.import_module(neural_config_path.replace(".py", "").replace("/", "."))
            self.register_module_buffer("neural_rescorer", neural_config.model)
            self.neural_rescorer.load(os.path.join(neural_config.callback_path, neural_checkpoint))
            self.neural_tokenizer = spm.SentencePieceProcessor(neural_config.tokenizer_path)
            self.neural_pad_token = neural_config.pad_token
            self.neural_bos_token = torch.tensor([neural_config.sos_token], dtype=torch.long)
            self.neural_eos_token = torch.tensor([neural_config.eos_token], dtype=torch.long)
        else:
            self.neural_rescorer = None

    def forward(self, outputs, from_logits=True):

        if from_logits:
            tokens = self.beam_search(*outputs)
        else:
            tokens = outputs[0].tolist()

        return self.tokenizer.decode(tokens)

    def beam_search(self, logits, logits_len, verbose=False):

        if CTCBeamDecoder is None:
            raise RuntimeError(
                "CTCBeamDecoder is unavailable. Install/build ctcdecode or switch to greedy decoding."
            )

        # test_time_aug
        if self.test_time_aug:
            batch_size, num_augments = logits.shape[0], logits.shape[1] # b, Naug
            logits = logits.flatten(start_dim=0, end_dim=1)
            logits_len = logits_len.flatten(start_dim=0, end_dim=1)
        else:
            batch_size, num_augments = logits.shape[0], 1

        # Beam Search Decoder
        decoder = CTCBeamDecoder(
            [chr(idx + self.ngram_offset) for idx in range(self.tokenizer.vocab_size())],
            model_path=self.ngram_path,
            alpha=self.ngram_alpha,
            beta=self.ngram_beta,
            cutoff_top_n=self.tokenizer.vocab_size(),
            cutoff_prob=1.0,
            beam_width=self.beam_size,
            num_processes=self.num_processes,
            blank_id=0,
            log_probs_input=True
        )

        # Apply Temperature
        logits = logits / self.ngram_tmp

        # Softmax -> Log
        logP = logits.log_softmax(dim=-1)

        # Beam Search Decoding: (B, Beam, N), (B, Beam), (B, Beam, N), (B, Beam)
        beam_results, beam_scores, timesteps, out_lens = decoder.decode(logP, logits_len)

        # Neural Rescoring
        if self.neural_rescorer is not None:

            # Convert to neural LM ids (B, Beam, sos + N + eos)
            preds_ids_neural = [[torch.cat([self.neural_bos_token, torch.tensor(self.neural_tokenizer.encode(self.tokenizer.decode(beam_results[b][beam][:out_lens[b][beam]].tolist())), dtype=torch.long), self.neural_eos_token], dim=0) for beam in range(self.beam_size)] for b in range(logits.size(0))]

            # Pad ids (B, Beam, sos + N + eos)
            preds_ids_neural_pad = [torch.nn.utils.rnn.pad_sequence(preds_ids_neural[b], batch_first=True, padding_value=self.neural_pad_token) for b in range(logits.size(0))]

            # Forward Ids (B, Beam, sos + N + eos, V)
            neural_results = [self.neural_rescorer(preds_ids_neural_pad[b].to(self.device)).cpu() for b in range(logits.size(0))]

            # Softmax -> Log -> Neg (B, Beam, sos + N + eos, V)
            neural_results = [- neural_results[b].log_softmax(dim=-1) for b in range(logits.size(0))]

            # Compute Neural Scores and Lengths (B, beam)
            neural_scores = torch.zeros(logits.size(0), self.beam_size)
            neural_lengths = torch.zeros(logits.size(0), self.beam_size)
            for b in range(logits.size(0)):
                for beam in range(self.beam_size):
                    length_pred = len(preds_ids_neural[b][beam][1:]) # N + eos
                    for t in range(length_pred):
                        neural_scores[b][beam] += neural_results[b][beam][t][preds_ids_neural[b][beam][t+1]]
                    neural_lengths[b][beam] += self.neural_beta * length_pred

            # Rescore Predictions
            total_scores = beam_scores + self.neural_alpha * neural_scores - self.neural_beta * neural_lengths

            # (B, Beam) -> (b, Naug * Beam)
            total_scores = total_scores.reshape(batch_size, num_augments * self.beam_size)
            beam_results = beam_results.reshape(batch_size, num_augments * self.beam_size, -1)
            out_lens = out_lens.reshape(batch_size, num_augments * self.beam_size)

            # Best Ids (b,)
            best_ids = total_scores.argmin(dim=-1)

        else:

            # (B, Beam) -> (b, Naug)
            beam_scores = beam_scores.reshape(batch_size, num_augments, self.beam_size)[:, :, 0]
            beam_results = beam_results.reshape(batch_size, num_augments, self.beam_size, -1)[:, :, 0]
            out_lens = out_lens.reshape(batch_size, num_augments, self.beam_size)[:, :, 0]

            # Best Ids (b,)
            best_ids = beam_scores.argmin(dim=-1)

        # Batch Pred List
        batch_pred_list = [beam_results[b][best_ids[b]][:out_lens[b][best_ids[b]]].tolist() for b in range(batch_size)]

        return batch_pred_list

###############################################################################
# Decoder Dictionary
###############################################################################

decoder_dict = {
    "Threshold": ThresholdDecoder,
    "ArgMax": ArgMaxDecoder,
    "CTCGreedySearchDecoder": CTCGreedySearchDecoder,
    "CTCBeamSearch": CTCBeamSearchDecoder
}

###############################################################################
# Adaptive decoding: confidence-gated greedy/beam/LLM
###############################################################################

class AdaptiveCTCDecoder(nn.Module):
    """
    Confidence-adaptive CTC decoding with optional LLM (neural LM) rescoring.

    Strategy per utterance (batch element):
      - Compute an entropy-based confidence score from AV logits
      - If low entropy (high confidence): use greedy
      - If medium entropy: use small beam (no LLM)
      - If high entropy: use large beam and optionally enable neural LM (distilled LLM) rescoring

    This reduces LM usage and latency by invoking the LLM only when needed.

    Notes/assumptions:
      - Expects CTC outputs (logits, logits_len)
      - Entropy computed over non-blank distribution if exclude_blank=True, averaged over valid frames
      - For mixed decisions in a batch, decoding is run per group and merged back preserving order
    """

    def __init__(
        self,
        tokenizer_path: str,
        entropy_low: float = 1.2,
        entropy_high: float = 2.0,
        small_beam: int = 8,
        large_beam: int = 32,
        exclude_blank: bool = True,
        use_visual_gating: bool = True,
        # Optional n-gram LM
        ngram_path: str | None = None,
        ngram_tmp: float = 1.0,
        ngram_alpha: float = 0.6,
        ngram_beta: float = 1.0,
        ngram_offset: int = 100,
        num_processes: int = 8,
        # Optional neural LM (distilled LLM) for large-beam tier only
        neural_config_path: str | None = None,
        neural_checkpoint: str | None = None,
        neural_alpha: float = 0.6,
        neural_beta: float = 1.0,
        test_time_aug: bool = False,
    ):
        super().__init__()
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        self.blank_id = 0
        self.use_visual_gating = bool(use_visual_gating)

        # thresholds and beams
        self.entropy_low = float(entropy_low)
        self.entropy_high = float(entropy_high)
        assert self.entropy_low <= self.entropy_high, "entropy_low should be <= entropy_high"
        self.small_beam = int(small_beam)
        self.large_beam = int(large_beam)
        self.exclude_blank = bool(exclude_blank)

        # Greedy and beam decoders (instantiate two CTC beam decoders for small/large)
        self.greedy = CTCGreedySearchDecoder(tokenizer_path=tokenizer_path, blank_token=self.blank_id)
        # Beam decoders (if ctcdecode available); otherwise we'll fallback to greedy
        self._beam_available = CTCBeamDecoder is not None
        if self._beam_available:
            self.beam_small = CTCBeamSearchDecoder(
                tokenizer_path=tokenizer_path,
                beam_size=self.small_beam,
                ngram_path=ngram_path,
                ngram_tmp=ngram_tmp,
                ngram_alpha=ngram_alpha,
                ngram_beta=ngram_beta,
                ngram_offset=ngram_offset,
                num_processes=num_processes,
                test_time_aug=test_time_aug,
            )
            # Large beam optionally with neural rescorer
            self.beam_large = CTCBeamSearchDecoder(
                tokenizer_path=tokenizer_path,
                beam_size=self.large_beam,
                ngram_path=ngram_path,
                ngram_tmp=ngram_tmp,
                ngram_alpha=ngram_alpha,
                ngram_beta=ngram_beta,
                ngram_offset=ngram_offset,
                num_processes=num_processes,
                test_time_aug=test_time_aug,
                neural_config_path=neural_config_path,
                neural_checkpoint=neural_checkpoint,
                neural_alpha=neural_alpha,
                neural_beta=neural_beta,
            )
        else:
            self.beam_small = None
            self.beam_large = None

    @staticmethod
    def _avg_entropy(logits: torch.Tensor, lengths: torch.Tensor, blank_id: int = 0, exclude_blank: bool = True) -> torch.Tensor:
        """Compute average per-frame entropy per batch element.

        logits: (B, T, V)
        lengths: (B,)
        Returns: (B,) entropies
        """
        with torch.no_grad():
            probs = logits.softmax(dim=-1)  # (B,T,V)
            if exclude_blank:
                # zero-out blank prob and renormalize over non-blank
                probs = probs.clone()
                probs[..., blank_id] = 0.0
                Z = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                probs = probs / Z
            # entropy per frame: -sum p log p
            ent = -(probs.clamp_min(1e-8).log() * probs).sum(dim=-1)  # (B,T)
            B, T = ent.shape
            mask = torch.arange(T, device=ent.device).unsqueeze(0) < lengths.view(B, 1)
            ent_sum = (ent * mask).sum(dim=1)
            lengths_f = lengths.to(ent.dtype).clamp_min(1.0)
            return ent_sum / lengths_f

    def forward(self, outputs, from_logits: bool = True):
        # Expect (logits, lengths) or (logits, lengths, gating_logits, gating_lengths)
        if not from_logits:
            # Already token ids: pass-through decode using tokenizer
            return self.tokenizer.decode(outputs[0].tolist())

        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            logits, lengths = outputs[0], outputs[1]
            gating_logits = outputs[2] if len(outputs) >= 3 else None
            gating_lengths = outputs[3] if len(outputs) >= 4 else None
        else:
            logits, lengths = outputs
            gating_logits, gating_lengths = None, None

        # If test-time augmentation present (B, Naug, T, V), collapse aug dimension by averaging logits
        if logits.dim() == 4:
            # (B,N,T,V) -> (B,T,V)
            logits = logits.mean(dim=1)

        # Compute entropy-based confidence (prefer separate gating source if provided)
        if gating_logits is not None and gating_lengths is not None:
            ent = self._avg_entropy(gating_logits, gating_lengths, blank_id=self.blank_id, exclude_blank=self.exclude_blank)
        else:
            ent = self._avg_entropy(logits, lengths, blank_id=self.blank_id, exclude_blank=self.exclude_blank)

    # Partition indices
        idx_greedy = (ent <= self.entropy_low).nonzero(as_tuple=False).view(-1)
        idx_small = ((ent > self.entropy_low) & (ent <= self.entropy_high)).nonzero(as_tuple=False).view(-1)
        idx_large = (ent > self.entropy_high).nonzero(as_tuple=False).view(-1)

        # Helper to slice tensors by indices preserving device
        def _gather(t: torch.Tensor, idx: torch.Tensor):
            if idx.numel() == 0:
                return None
            return t.index_select(0, idx)

        decoded_text = [None] * logits.size(0)

        # Routing/latency stats
        import time
        B = int(logits.size(0))
        stats = {
            "adaptive/B": B,
            "adaptive/entropy_mean": float(ent.mean().item()),
            "adaptive/entropy_low": float(self.entropy_low),
            "adaptive/entropy_high": float(self.entropy_high),
            "adaptive/n_greedy": int(idx_greedy.numel()),
            "adaptive/n_small": int(idx_small.numel()),
            "adaptive/n_large": int(idx_large.numel()),
            "adaptive/latency_ms_greedy": 0.0,
            "adaptive/latency_ms_small": 0.0,
            "adaptive/latency_ms_large": 0.0,
            "adaptive/latency_ms_total": 0.0,
        }
        t_all = time.perf_counter()

        # Run greedy on low-entropy group
        if idx_greedy.numel() > 0:
            t0 = time.perf_counter()
            out = self.greedy.forward((_gather(logits, idx_greedy), _gather(lengths, idx_greedy)), from_logits=True)
            for i, s in zip(idx_greedy.tolist(), out):
                decoded_text[i] = s
            stats["adaptive/latency_ms_greedy"] = (time.perf_counter() - t0) * 1000.0

        # Run small-beam (no neural LM) on medium-entropy group
        if idx_small.numel() > 0:
            t0 = time.perf_counter()
            if self._beam_available and self.beam_small is not None:
                out = self.beam_small.forward((_gather(logits, idx_small), _gather(lengths, idx_small)), from_logits=True)
            else:
                out = self.greedy.forward((_gather(logits, idx_small), _gather(lengths, idx_small)), from_logits=True)
            for i, s in zip(idx_small.tolist(), out):
                decoded_text[i] = s
            stats["adaptive/latency_ms_small"] = (time.perf_counter() - t0) * 1000.0

        # Run large-beam (with optional neural LM) on high-entropy group
        if idx_large.numel() > 0:
            t0 = time.perf_counter()
            if self._beam_available and self.beam_large is not None:
                out = self.beam_large.forward((_gather(logits, idx_large), _gather(lengths, idx_large)), from_logits=True)
            else:
                out = self.greedy.forward((_gather(logits, idx_large), _gather(lengths, idx_large)), from_logits=True)
            for i, s in zip(idx_large.tolist(), out):
                decoded_text[i] = s
            stats["adaptive/latency_ms_large"] = (time.perf_counter() - t0) * 1000.0

        stats["adaptive/latency_ms_total"] = (time.perf_counter() - t_all) * 1000.0
        # expose last stats for external consumers (e.g., model logging)
        try:
            self.last_stats = stats
        except Exception:
            pass

        # Try to attach stats to outer model infos if available
        try:
            # Walk up through potential attributes sometimes set by frameworks
            parent = getattr(self, "_parent_model", None)
            if parent is None:
                # Some models might have attached decoders directly on the model
                # which is a Module with add_info; attempt to find a Module ancestor
                pass
            if parent is not None and hasattr(parent, "add_info"):
                for k, v in stats.items():
                    parent.add_info(k, v)
        except Exception:
            pass

        # All should be filled; fallback to greedy if any None
        for i in range(len(decoded_text)):
            if decoded_text[i] is None:
                decoded_text[i] = self.greedy.forward((logits[i:i+1], lengths[i:i+1]), from_logits=True)[0]

        return decoded_text

