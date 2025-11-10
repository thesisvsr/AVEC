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

###############################################################################
# Collate Functions
###############################################################################

class Collate(nn.Module):

    def __init__(self):
        super(Collate, self).__init__()

    def forward(self, samples):
        return samples

class CollateFn(nn.Module):

    """ Collate samples to List / Dict

    Args:
        - inputs_params_: List / Dict of collate param for inputs
        - targets_params: List / Dict of collate param for targets

    Collate Params Dict:
        - axis: axis to select samples
        - padding: whether to pad samples
        - padding_value: padding token, default 0

    """

    def __init__(self, inputs_params=[{"axis": 0}], targets_params=[{"axis": 1}]):
        super(CollateFn, self).__init__()

        assert isinstance(inputs_params, dict) or isinstance(inputs_params, list) or isinstance(inputs_params, tuple)
        self.inputs_params = inputs_params
        assert isinstance(targets_params, dict) or isinstance(targets_params, list) or isinstance(targets_params, tuple)
        self.targets_params = targets_params

        # Default Params
        if  isinstance(inputs_params, dict):
            for params in self.inputs_params.values():
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

            for params in self.targets_params.values():
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

        else:
            for params in self.inputs_params:
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

            for params in self.targets_params:
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

    def forward(self, samples):
        return {"inputs": self.collate(samples, self.inputs_params), "targets": self.collate(samples, self.targets_params)}

    def collate(self, samples, collate_params):

        # Dict
        if isinstance(collate_params, dict):
            collates = {}
            for name, params in collate_params.items():

                # Select
                collate = [sample[params["axis"]] for sample in samples]

                # Start Token
                if params["start_token"]:
                    collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]

                # End Token
                if params["end_token"]:
                    collate = [torch.cat([item, params["end_token"] * item.new_ones(1)]) for item in collate]

                # Padding
                if params["padding"]:
                    collate = torch.nn.utils.rnn.pad_sequence(collate, batch_first=True, padding_value=params["padding_value"])
                else:
                    collate = torch.stack(collate, axis=0)

                # Append
                collates[name] = collate
        # List
        elif isinstance(collate_params, list):
            collates = []
            for params in collate_params:

                # Select
                collate = [sample[params["axis"]] for sample in samples]

                # Start Token
                if params["start_token"]:
                    collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]

                # End Token
                if params["end_token"]:
                    collate = [torch.cat([item, params["end_token"] * item.new_ones(1)]) for item in collate]

                # Padding
                if params["padding"]:
                    collate = torch.nn.utils.rnn.pad_sequence(collate, batch_first=True, padding_value=params["padding_value"])
                else:
                    collate = torch.stack(collate, axis=0)

                # Append
                collates.append(collate)
        # Tuple
        elif isinstance(collate_params, tuple):
            collates = []
            for params in collate_params:

                # Select
                collate = [sample[params["axis"]] for sample in samples]

                # Start Token
                if params["start_token"]:
                    collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]

                # End Token
                if params["end_token"]:
                    collate = [torch.cat([item, params["end_token"] * item.new_ones(1)]) for item in collate]

                # Padding
                if params["padding"]:
                    collate = torch.nn.utils.rnn.pad_sequence(collate, batch_first=True, padding_value=params["padding_value"])
                else:
                    collate = torch.stack(collate, axis=0)

                # Append
                collates.append(collate)
            collates = tuple(collates)

        # For targets: if exactly two items (labels, lengths), return a tuple to preserve (y, y_len)
        # This helps CTC loss which expects both labels and target_lengths.
        # If the targets_params was provided as a tuple (e.g., (labels, lengths)),
        # return a tuple to preserve (y, y_len) structure expected by CTC.
        if isinstance(self.targets_params, tuple) and collate_params is self.targets_params:
            # Already a tuple when collated with tuple spec; ensure it's a tuple
            collates = tuple(collates) if not isinstance(collates, tuple) else collates
        else:
            collates = collates[0] if len(collates) == 1 else collates

        return collates


class CollateCTC(nn.Module):
    """Collate function for samples shaped like LRS/BSRDCTC outputs:
    (video, audio, label, video_len, audio_len, label_len)
    Returns a dict with keys: video, audio, label, video_len, audio_len, label_len
    Label is a 1D concatenated tensor for CTC; label_len provides lengths.
    Video is stacked (B, C, T, H, W) by padding to max T in batch (repeat last frame).
    """
    def __init__(self, pad_video_last=True):
        super().__init__()
        self.pad_video_last = pad_video_last

    def forward(self, samples):
        videos = []
        video_lens = []
        labels = []
        label_lens = []
        audios = []
        audio_lens = []
        max_T = 0
        for (video, audio, label, vlen, alen, llen) in samples:
            videos.append(video)
            video_lens.append(vlen)
            labels.append(label)
            label_lens.append(llen)
            audios.append(audio)
            audio_lens.append(alen)
            if video.shape[1] > max_T:
                max_T = video.shape[1]
        # Pad videos by repeating last frame
        padded_videos = []
        for v in videos:
            C,T,H,W = v.shape
            if T < max_T and T>0:
                pad = v[:, -1:, :, :].expand(C, max_T-T, H, W)
                v = torch.cat([v, pad], dim=1)
            padded_videos.append(v)
        video_batch = torch.stack(padded_videos, dim=0)  # (B,C,T,H,W)
        # Concatenate labels
        if labels and labels[0].ndim==1:
            label_batch = torch.cat(labels, dim=0)
        else:
            label_batch = torch.zeros(0, dtype=torch.long)
        video_len_tensor = torch.stack(video_lens) if len(video_lens)>0 else torch.zeros(0, dtype=torch.long)
        label_len_tensor = torch.stack(label_lens) if len(label_lens)>0 else torch.zeros(0, dtype=torch.long)
        batch_dict = {
            'video': video_batch,
            'audio': None,  # audio currently unused for BSRD visual-only
            'label': label_batch,
            'video_len': video_len_tensor,
            'audio_len': torch.stack(audio_lens) if len(audio_lens)>0 else torch.zeros(0, dtype=torch.long),
            'label_len': label_len_tensor
        }
        # Add generic keys expected by training loop: inputs (video, video_len) and targets (label, label_len)
        # Model expects (B, T, H, W, C) so convert from (B, C, T, H, W)
        video_cl = video_batch.permute(0, 2, 3, 4, 1).contiguous()
        batch_dict['inputs'] = (video_cl, video_len_tensor)
        batch_dict['targets'] = (label_batch, label_len_tensor)
        return batch_dict