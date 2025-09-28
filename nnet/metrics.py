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
import jiwer
from typing import List, Union

###############################################################################
# Metrics
###############################################################################

class Mean(nn.Module):

    def __init__(self, name="mean"):
        super(Mean, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):

        # Compute mean
        mean = y_pred.mean()

        return mean

class CategoricalAccuracy(nn.Module):

    def __init__(self, ignore_index=-1, dim_argmax=-1, name="acc"):
        super(CategoricalAccuracy, self).__init__()
        self.name = name
        self.dim_argmax = dim_argmax
        self.ignore_index = ignore_index

    def forward(self, y_true, y_pred):

        # ArgMax
        if self.dim_argmax != None:
            y_pred = y_pred.argmax(dim=self.dim_argmax)

        # Compute Mask
        mask = torch.where(y_true==self.ignore_index, 0.0, 1.0)

        # Reduction
        n = torch.count_nonzero(mask)

        # Element Wise Accuracy
        acc = torch.where(y_true==y_pred, 1.0, 0.0)

        # Mask Accuracy
        acc = acc * mask

        # Categorical Accuracy
        acc = 100 * acc.sum() / n

        return acc

class CategoricalAccuracyTopK(nn.Module):

    def __init__(self, ignore_index=-1, dim_topk=-1, topk=5, name=None):
        super(CategoricalAccuracyTopK, self).__init__()
        self.name = name if name != None else "topk{}".format(topk)
        self.ignore_index = ignore_index
        self.topk = topk
        self.dim_topk = dim_topk

    def forward(self, y_true, y_pred):

        # Compute Mask
        mask = torch.where(y_true==self.ignore_index, 0.0, 1.0)

        # Reduction
        n = torch.count_nonzero(mask)

        # Element Wise Topk Accuracy
        values, indices = y_pred.topk(self.topk, dim=self.dim_topk, largest=True, sorted=True)
        y_true = y_true.unsqueeze(dim=-1).repeat(1, 1, self.topk)
        acc = torch.where(y_true==indices, 1.0, 0.0).sum(dim=-1)

        # Mask Accuracy
        acc = acc * mask

        # Categorical Accuracy
        acc = 100 * acc.sum() / n

        return acc

class WordErrorRate(nn.Module):

    def __init__(self, name="wer"):
        super(WordErrorRate, self).__init__()
        self.name = name

    def forward(self, targets, outputs):
        # Word Error Rate
        # Some jiwer versions don't support the 'standardize' argument
        return torch.tensor(100 * jiwer.wer(targets, outputs))

class CharacterErrorRate(nn.Module):

    def __init__(self, name="cer"):
        super(CharacterErrorRate, self).__init__()
        self.name = name

    def _levenshtein(self, ref: str, hyp: str) -> int:
        # Simple DP edit distance at character level
        n, m = len(ref), len(hyp)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + cost  # substitution
                )
        return dp[n][m]

    def forward(self, targets: Union[str, List[str]], outputs: Union[str, List[str]]):
        # Prefer jiwer.cer if available
        try:
            cer_value = 100 * jiwer.cer(targets, outputs)  # type: ignore[attr-defined]
            return torch.tensor(cer_value)
        except Exception:
            pass

        # Fallback manual CER computation
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(outputs, str):
            outputs = [outputs]

        total_dist = 0
        total_chars = 0
        for t, o in zip(targets, outputs):
            total_dist += self._levenshtein(t, o)
            total_chars += max(len(t), 1)

        cer = 100.0 * (total_dist / max(total_chars, 1))
        return torch.tensor(cer)

###############################################################################
# Metric Dictionary
###############################################################################

metric_dict = {
    "CategoricalAccuracy": CategoricalAccuracy,
    "WordErrorRate": WordErrorRate,
    "CharacterErrorRate": CharacterErrorRate
}