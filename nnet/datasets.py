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
import torchvision
import torchaudio
from torchvision.datasets.utils import extract_archive

# Other
import os
import glob
from tqdm import tqdm
import sentencepiece as spm
import numpy as np
import requests
import pickle
import gdown
import multiprocessing
import hashlib
from nnet import layers
from nnet import transforms
from nnet import collate_fn

###############################################################################
# Datasets
###############################################################################

class Dataset(torch.utils.data.Dataset):

    def __init__(self, batch_size=8, collate_fn=collate_fn.Collate(), root="datasets", shuffle=True):
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.root = root
        self.shuffle = shuffle

class MultiDataset(Dataset):

    def __init__(self, batch_size, collate_fn, datasets, shuffle=True):
        super(MultiDataset, self).__init__(batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle, root=None)

        self.datasets = datasets

    def __len__(self):

        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, n):

        ctr = 0
        for dataset in self.datasets:
            ctr_prev = ctr
            ctr += len(dataset)
            if n < ctr:
                return dataset.__getitem__(n - ctr_prev)
            
class LRS(Dataset):

    """ LRS2 and LRS3 datasets
    
    Lip Reading Sentences 2 (LRS2) Dataset : https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html

    The dataset consists of thousands of spoken sentences from BBC television. Each sentences is up to 100 characters in length. 
    The training, validation and test sets are divided according to broadcast date. The dataset statistics are given in the table below.
    The utterances in the pre-training set correspond to part-sentences as well as multiple sentences, whereas the training set only consists of single full sentences or phrases. 
    There is some overlap between the pre-training and the training sets.
    Although there might be some label noise in the pre-training and the training sets, the test set has undergone additional verification; so, to the best of our knowledge, there are no errors in the test set.

    Infos:
        37 characters: 26 (a-z) letters + apostrophe (') + 10 (0-9) numbers
        total = 144482 samples
        pretrain + train = 142,157 training samples, 224 hours
        160 x 160, 25 fps videos

        - 96,318 pretrain samples, pretrain folder, 195 hours
        - 45,839 train samples, main folder, 28 hours
        - 1,082 val samples, main folder, 0.6 hours
        - 1,243 test samples, main folder, 0.5 hours

    Reference: "Deep Audio-Visual Speech Recognition", Afouras et al.
    https://arxiv.org/abs/1809.02108


    --------------------------------------------------------------------------------------------------------------------
    

    Lip Reading Sentences 3 (LRS3) Dataset : https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html

    The dataset consists of thousands of spoken sentences from TED and TEDx videos. 
    There is no overlap between the videos used to create the test set and the ones used for the pre-train and trainval sets.

    Reference: "LRS3-TED: a large-scale dataset for visual speech recognition", Afouras et al.
    https://arxiv.org/abs/1809.00496

    151,819 total samples
    150,498 pretrain + trainval samples, 438 hours
    118,516 pretrain samples from 5,090 videos, 407 hours
    31,982 trainval samples from 4,004 videos, 30 hours
    1,321 test samples from 412 videos, 1 hour

    37 characters: 26 (a-z) letters + apostrophe (') + 10 (0-9) numbers
    16kHz audio
    25fps 224x224pixel video
    
    """

    def __init__(self, batch_size, collate_fn, version="LRS2", img_mean=(0.5,), img_std=(0.5,), crop_mouth=True, root="datasets", shuffle=True, ascending=False, mode="pretrain+train+val", load_audio=True, load_video=True, video_transform=None, audio_transform=None, download=False, prepare=False, workers_prepare=-1, video_max_length=None, audio_max_length=None, label_max_length=None, tokenizer_path="datasets/LRS3/tokenizerbpe256.model", mean_face_path="media/20words_mean_face.npy", align=False,
                 subset_fraction=0.3, subset_seed=0, prepare_subset_fraction=None, prepare_subset_seed=0):
        super(LRS, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle and not ascending)

        assert version in ["LRS2"]

        # Params
        self.version = version
        self.mode = mode
        self.ascending = ascending
        self.load_audio = load_audio
        self.load_video = load_video
        self.video_max_length = video_max_length
        self.audio_max_length = audio_max_length
        self.label_max_length = label_max_length
        self.workers_prepare = multiprocessing.cpu_count() if workers_prepare==-1 else workers_prepare
        self.tokenizer_path = tokenizer_path
        self.crop_mouth = crop_mouth
        self.mean_face_path = mean_face_path
        self.align = align
        # Optional subsetting for quicker experiments (can also be set via env vars)
        # If subset_fraction is None, try environment variable AVEC_SUBSET_FRACTION; default to 1.0 (use all)
        env_subset = os.getenv("AVEC_SUBSET_FRACTION")
        self.subset_fraction = (float(subset_fraction) if subset_fraction is not None
                    else (float(env_subset) if env_subset not in (None, "") else 1.0))
        env_subset_seed = os.getenv("AVEC_SUBSET_SEED")
        self.subset_seed = int(subset_seed if subset_seed is not None else (int(env_subset_seed) if env_subset_seed not in (None, "") else 0))
        env_prep_subset = os.getenv("AVEC_PREPARE_SUBSET_FRACTION")
        self.prepare_subset_fraction = (float(prepare_subset_fraction) if prepare_subset_fraction is not None
                        else (float(env_prep_subset) if env_prep_subset not in (None, "") else None))
        env_prep_subset_seed = os.getenv("AVEC_PREPARE_SUBSET_SEED")
        self.prepare_subset_seed = int(prepare_subset_seed if prepare_subset_seed is not None else (int(env_prep_subset_seed) if env_prep_subset_seed not in (None, "") else 0))

        # Download Dataset
        if download:
            self.download()

        # Prepare Dataset
        if prepare:
            self.prepare()

        # LRS2
        if version == "LRS2":

            # Mode
            assert mode in ["pretrain+train+val", "pretrain+train", "pretrain", "train", "val", "test"]

            # Paths
            self.paths = []
            if "pretrain" in mode:
                with open(os.path.join(root, "LRS2", "pretrain.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "LRS2", "mvlrs_v1", "pretrain", line.replace("\n", "")))
            if "train" in mode:
                with open(os.path.join(root, "LRS2", "train.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "LRS2", "mvlrs_v1", "main", line.replace("\n", "")))
            if "val" in mode:
                with open(os.path.join(root, "LRS2", "val.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "LRS2", "mvlrs_v1", "main", line.replace("\n", "")))
            if "test" in mode:
                with open(os.path.join(root, "LRS2", "test.txt")) as f:
                    for line in f.readlines():
                        self.paths.append(os.path.join(root, "LRS2", "mvlrs_v1", "main", line.split()[0]))

        # Keep only prepared samples (require .pt to exist)
        prepared = [p for p in self.paths if os.path.isfile(p + ".pt")]
        if len(prepared) == 0:
            print("Warning: no prepared LRS2 samples found. Run prepare or adjust subset/seed.")
        self.paths = prepared

    # LRS3 (disabled)
    # elif version == "LRS3":
    #     # Mode
    #     assert mode in ["pretrain+trainval", "pretrain", "trainval", "test"]
    #     # Paths
    #     self.paths = []
    #     for m in mode.split("+"):
    #         self.paths += glob.glob(os.path.join(root, "LRS3", m, "*", "*.txt"))
    #     for i, path in enumerate(self.paths):
    #         self.paths[i] = path[:-4]

        # Reduce to subset if requested (before filtering/sorting)
        if self.subset_fraction is not None and self.subset_fraction < 1.0:
            total = len(self.paths)
            keep = max(1, int(round(total * self.subset_fraction)))
            if keep < total:
                rng = np.random.RandomState(self.subset_seed)
                idx = rng.permutation(total)[:keep]
                idx.sort()  # stable order
                self.paths = [self.paths[i] for i in idx]
                try:
                    print(f"Using subset of {keep}/{total} samples (~{self.subset_fraction*100:.1f}%) for {self.version} [{self.mode}]")
                except Exception:
                    pass

        # Video Transforms
        self.video_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            layers.Permute(dims=(1, 0, 2, 3)),
            torchvision.transforms.Grayscale(),
            layers.Permute(dims=(1, 0, 2, 3)),
            transforms.NormalizeVideo(mean=img_mean, std=img_std),
            video_transform if video_transform != None else nn.Identity()
        ])

        # Audio Transforms
        self.audio_preprocessing = audio_transform if audio_transform != None else nn.Identity() 

        # Filter Dataset
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.filter()
                n_elt = [len(self.paths)]
            else:
                n_elt = [None]

            # Broadcast number of elements
            torch.distributed.barrier()
            torch.distributed.broadcast_object_list(n_elt, src=0)

            # Broadcast path list
            torch.distributed.barrier()
            if torch.distributed.get_rank() != 0:
                self.paths = [None for _ in range(n_elt[0])]
            torch.distributed.broadcast_object_list(self.paths, src=0)
        else:
            self.filter()

    def create_corpus(self, mode):

        corpus_path = os.path.join(self.root, self.version, "corpus_{}.txt".format(mode))
        
        if not os.path.isfile(corpus_path):

            print("Create Corpus File: {} {}".format(self.version, mode))
            corpus_file = open(corpus_path, "w")

            # LRS2
            if self.version == "LRS2":

                if "pretrain" == mode:
                    with open(os.path.join(self.root, "LRS2", "pretrain.txt")) as f:
                        for line in tqdm(f.readlines()):
                            with open(os.path.join(self.root, "LRS2", "mvlrs_v1", "pretrain", line.replace("\n", "") + ".txt"), "r") as f:
                                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                                corpus_file.write(line)

                if "train" == mode:
                    with open(os.path.join(self.root, "LRS2", "train.txt")) as f:
                        for line in tqdm(f.readlines()):
                            with open(os.path.join(self.root, "LRS2", "mvlrs_v1", "main", line.replace("\n", "") + ".txt"), "r") as f:
                                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                                corpus_file.write(line)

                if "val" == mode:
                    with open(os.path.join(self.root, "LRS2", "val.txt")) as f:
                        for line in tqdm(f.readlines()):
                            with open(os.path.join(self.root, "LRS2", "mvlrs_v1", "main", line.replace("\n", "") + ".txt"), "r") as f:
                                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                                corpus_file.write(line)

                if "test" == mode:
                    with open(os.path.join(self.root, "LRS2", "test.txt")) as f:
                        for line in tqdm(f.readlines()):
                            with open(os.path.join(self.root, "LRS2", "mvlrs_v1", "main", line.split()[0] + ".txt"), "r") as f:
                                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                                corpus_file.write(line)

            # LRS3
            elif self.version == "LRS3":

                for file_path in tqdm(glob.glob(os.path.join(self.root, "LRS3", mode, "*", "*.txt"))):
                    with open(file_path, "r") as f:
                        line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                        corpus_file.write(line)

    class FilterDataset:

        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            file_path = self.paths[idx]
            return file_path, torch.load(file_path + ".pt")
      
    def filter(self):

        if self.video_max_length==None and self.audio_max_length==None and self.label_max_length==None:
            return
        else:
            video_max_length = self.video_max_length if self.video_max_length != None else float("inf")
            audio_max_length = self.audio_max_length if self.audio_max_length != None else float("inf")
            label_max_length = self.label_max_length if self.label_max_length != None else float("inf")
            print("Dataset Filtering")
            print("Video maximum length : {} / Audio maximum length : {} / Label sequence maximum length : {}".format(video_max_length, audio_max_length, label_max_length))
            
            filename = os.path.join(self.root, self.version, "mode_{}_video_max_length_{}_audio_max_length_{}_label_max_length_{}_paths.pt".format(self.mode, video_max_length, audio_max_length, label_max_length))
            if not os.path.isfile(filename):
                # Robust pre-scan: skip unreadable/corrupted .pt files without crashing
                paths = []
                lengths = []
                skipped = 0
                for path in tqdm(self.paths):
                    try:
                        infos = torch.load(path + ".pt", map_location="cpu")
                    except Exception:
                        skipped += 1
                        continue
                    if infos["video_len"] <= video_max_length and infos["audio_len"] <= audio_max_length and infos["label_len"] <= label_max_length:
                        paths.append(path)
                        lengths.append(infos["audio_len"])
                if skipped:
                    print(f"Skipped {skipped} unreadable items during filter().")
                self.paths = paths
                torch.save(self.paths, filename)

            else:
                self.paths = torch.load(filename)

            # sort_by_length
            if self.ascending:
                paths = [elt[1] for elt in sorted(zip(lengths, paths))]

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, n):

        # Load Video
        if self.load_video:
            if self.crop_mouth:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + "_mouth.mp4")
            else:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + ".mp4")
        else:
            video = None

        # Load Audio
        if self.load_audio:
            audio = torchaudio.load(self.paths[n] + ".flac")[0]
        else:
            audio = None

        # Load Infos
        infos = torch.load(self.paths[n] + ".pt")
        label, video_len, audio_len, label_len = infos["label"], infos["video_len"], infos["audio_len"], infos["label_len"]

        # Audio Preprocessing
        if self.load_audio:
            audio = self.audio_preprocessing(audio[:1]).squeeze(dim=0)

        # Video Preprocessing
        if self.load_video: 
            video = video.permute(3, 0, 1, 2)
            video = self.video_preprocessing(video)
            if self.align:
                video = transforms.align_video_to_audio(video.permute(1, 2, 3, 0), audio)
                video_len = video.shape[0]
            else:
                video = video.permute(1, 2, 3, 0)

        # Infos Preprocessing
        video_len = torch.tensor(video_len, dtype=torch.long)
        audio_len = torch.tensor(audio_len, dtype=torch.long)
        label_len = torch.tensor(label_len, dtype=torch.long)

        return video, audio, label, video_len, audio_len, label_len

    def download_lrs2(self):

        # Download Pretrain
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partaa",
        #     path=os.path.join(self.root, "LRS2", "lrs2_v1_partaa")
        # )
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partab",
        #     path=os.path.join(self.root, "LRS2", "lrs2_v1_partab")
        # )
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partac",
        #     path=os.path.join(self.root, "LRS2", "lrs2_v1_partac")
        # )
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partad",
        #     path=os.path.join(self.root, "LRS2", "lrs2_v1_partad")
        # )
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partae",
        #     path=os.path.join(self.root, "LRS2", "lrs2_v1_partae")
        # )
        # os.system("cat " + os.path.join(self.root, "LRS2", "lrs2_v1_parta*") + " > " +  os.path.join(self.root, "LRS2", "lrs2_v1.tar"))
        # extract_archive(
        #     from_path=os.path.join(self.root, "LRS2", "lrs2_v1.tar"),
        #     to_path=os.path.join(self.root, "LRS2")
        # )   

        # Filelist
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/pretrain.txt",
        #     path=os.path.join(self.root, "LRS2", "pretrain.txt")
        # ) 
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/train.txt",
        #     path=os.path.join(self.root, "LRS2", "train.txt")
        # )  
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/val.txt",
        #     path=os.path.join(self.root, "LRS2", "val.txt")
        # )  
        # self.download_file(
        #     url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data2/test.txt",
        #     path=os.path.join(self.root, "LRS2", "test.txt")
        # )   

        # # Download Landmarks from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
        # gdown.download("https://drive.google.com/uc?id=1G2-rEUNeGotJ9EtTIj0UzqbvCSbn6CJy", os.path.join(self.root, "LRS2", "LRS2_landmarks.zip"), quiet=False)
        # extract_archive(
        #     from_path=os.path.join(self.root, "LRS2", "LRS2_landmarks.zip"),
        #     to_path=os.path.join(self.root, "LRS2")
        # )  
        pass 

    def download_lrs3(self):

        # Download Pretrain
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partaa",
            path=os.path.join(self.root, "LRS3", "lrs3_pretrain_partaa")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partab",
            path=os.path.join(self.root, "LRS3", "lrs3_pretrain_partab")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partac",
            path=os.path.join(self.root, "LRS3", "lrs3_pretrain_partac")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partad",
            path=os.path.join(self.root, "LRS3", "lrs3_pretrain_partad")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partae",
            path=os.path.join(self.root, "LRS3", "lrs3_pretrain_partae")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partaf",
            path=os.path.join(self.root, "LRS3", "lrs3_pretrain_partaf")
        )
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partag",
            path=os.path.join(self.root, "LRS3", "lrs3_pretrain_partag")
        )
        os.system("cat " + os.path.join(self.root, "LRS3", "lrs3_pretrain_part*") + " > " +  os.path.join(self.root, "LRS3", "lrs3_pretrain.zip"))
        extract_archive(
            from_path=os.path.join(self.root, "LRS3", "lrs3_pretrain.zip"),
            to_path=os.path.join(self.root, "LRS3")
        )   

        # Download Trainval
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_trainval.zip",
            path=os.path.join(self.root, "LRS3", "lrs3_trainval.zip")
        )  
        extract_archive(
            from_path=os.path.join(self.root, "LRS3", "lrs3_trainval.zip"),
            to_path=os.path.join(self.root, "LRS3")
        )      

        # Download Test
        self.download_file(
            url="https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_test_v0.4.zip",
            path=os.path.join(self.root, "LRS3", "lrs3_test_v0.4.zip")
        )  
        extract_archive(
            from_path=os.path.join(self.root, "LRS3", "lrs3_test_v0.4.zip"),
            to_path=os.path.join(self.root, "LRS3")
        )  

        # Download Landmarks from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages
        gdown.download("https://drive.google.com/uc?id=1QRdOgeHvmKK8t4hsceFVf_BSpidQfUyW", os.path.join(self.root, "LRS3", "LRS3_landmarks.zip"), quiet=False)
        extract_archive(
            from_path=os.path.join(self.root, "LRS3", "LRS3_landmarks.zip"),
            to_path=os.path.join(self.root, "LRS3")
        )

    def download(self):

        # Print
        print("Download Dataset")
        os.makedirs(os.path.join(self.root, self.version), exist_ok=True)

        # LRS2
        if self.version == "LRS2":
            self.download_lrs2()
        
        # LRS3
        elif self.version == "LRS3":
            self.download_lrs3()

    def download_file(self, url, path):

        # Download, Open and Write
        with requests.get(url, auth=(os.getenv("{}_USERNAME".format(self.version)), os.getenv("{}_PASSWORD".format(self.version))), stream=True) as r:
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)

    class PrepareDataset:

        def __init__(self, paths, tokenizer, mean_face_path, version):
            self.paths = paths
            self.tokenizer = tokenizer
            self.lip_crop = transforms.LipDetectCrop(mean_face_landmarks_path=mean_face_path)
            self.version = version

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):

            file_path = self.paths[idx]

            # Read and Encode
            with open(file_path, "r") as f:
                line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower().replace("\n", "")
                label = torch.LongTensor(self.tokenizer.encode(line))

            # Load Video
            video, audio, info = torchvision.io.read_video(file_path.replace(".txt", ".mp4"))

            # Save Audio
            torchaudio.save(file_path.replace(".txt", ".flac"), audio, sample_rate=16000)

            # Extract Landmarks
            if self.version == "LRS2":
                landmarks_pathname = file_path.replace(".txt", ".pkl").replace("mvlrs_v1", "LRS2_landmarks")
            elif self.version == "LRS3":
                landmarks_pathname = file_path.replace(".txt", ".pkl").replace("LRS3", "LRS3/LRS3_landmarks")
            with open(landmarks_pathname, "br") as f:
                landmarks = pickle.load(f)

            # Interpolate Landmarks
            preprocessed_landmarks = self.lip_crop.landmarks_interpolate(landmarks)

            # Crop
            if not preprocessed_landmarks:
                video = torchvision.transforms.functional.resize(video.permute(3, 0, 1, 2), size=(self.lip_crop.crop_height, self.lip_crop.crop_width)).permute(1, 2, 3, 0)
            else:
                video = self.lip_crop.crop_patch(video.numpy(), preprocessed_landmarks)
                assert video is not None
                video = torch.tensor(video)
         
            # Save Video
            torchvision.io.write_video(filename=file_path.replace(".txt", "_mouth.mp4"), video_array=video, fps=info["video_fps"], video_codec="libx264")

            # Save Infos
            infos = {"label": label, "video_len": video.shape[0], "audio_len": audio.shape[1], "label_len": label.shape[0]}
            torch.save(infos, file_path.replace(".txt", ".pt"))
            
            return file_path, infos

    def prepare(self):

        # Remove from corpus
        # {NS} ~ non scripted
        # {LG} ~ Laughter

        if self.version == "LRS2":
            paths_txt = glob.glob(os.path.join(self.root, "LRS2", "*", "*", "*", "*.txt"))
        elif self.version == "LRS3":
            paths_txt = glob.glob(os.path.join(self.root, "LRS3", "*", "*", "*.txt"))

        # Create Corpus File
        corpus_path = os.path.join(self.root, self.version, "corpus.txt")
        if not os.path.isfile(corpus_path):
            print("Create Corpus File")
            corpus_file = open(corpus_path, "w")
            for file_path in tqdm(paths_txt):
                with open(file_path, "r") as f:
                    line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
                    corpus_file.write(line)

        # Load Tokenizer
        tokenizer = spm.SentencePieceProcessor(self.tokenizer_path)

        # Optionally restrict how many files we preprocess for a quick first run
        if self.prepare_subset_fraction is not None and self.prepare_subset_fraction < 1.0:
            total = len(paths_txt)
            keep = max(1, int(round(total * self.prepare_subset_fraction)))
            if keep < total:
                rng = np.random.RandomState(self.prepare_subset_seed)
                idx = rng.permutation(total)[:keep]
                idx.sort()
                paths_txt = [paths_txt[i] for i in idx]
                try:
                    print(f"Preparing subset of {keep}/{total} files (~{self.prepare_subset_fraction*100:.1f}%) for {self.version}")
                except Exception:
                    pass

        # Prepare
        print("Prepare Dataset")
        dataloader = torch.utils.data.DataLoader(
            self.PrepareDataset(
                paths=paths_txt,
                tokenizer=tokenizer,
                mean_face_path=self.mean_face_path,
                version=self.version
            ),
            batch_size=1,
            num_workers=self.workers_prepare,
            collate_fn=collate_fn.Collate(),
        )
        for batch in tqdm(dataloader):
            pass

class CorpusLM(Dataset):

    def __init__(self, batch_size, collate_fn, root="datasets", shuffle=True, download=False, tokenizer_path="datasets/LRS3/tokenizerbpe1024.model", max_length=None, corpus_path="datasets/LibriSpeechCorpus/librispeech-lm-norm.txt"):
        super(CorpusLM, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle)

        # Params
        self.root = root
        self.max_len = max_length

        if isinstance(tokenizer_path, str):
            self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        else:
            self.tokenizer = tokenizer_path
        self.corpus = open(corpus_path, 'r').readlines()

    def __getitem__(self, i):

        if self.max_len:
            while len(self.tokenizer.encode(self.corpus[i].replace("\n", "").lower())) > self.max_len:
                i = torch.randint(0, self.__len__(), [])

        label = torch.LongTensor(self.tokenizer.encode(self.corpus[i].replace("\n", "").lower()))

        return label,

    def __len__(self):
        return len(self.corpus)

class LRW(Dataset):

    """ Lip Reading in the Wild (LRW) Dataset : https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html

    The dataset consists of up to 1000 utterances of 500 different words, spoken by hundreds of different speakers. 
    All videos are 29 frames (1.16 seconds) in length, and the word occurs in the middle of the video.

    Infos:
        488,766 train samples
        25,000 val samples
        test samples
        (29, 256, 256, 3) videos
        (1, 19456) audios
    
    """

    def __init__(self, batch_size, collate_fn, root="datasets", shuffle=True, mode="train", img_mean=(0.5,), img_std=(0.5,), crop_mouth=True, load_audio=True, load_video=True, video_transform=None, download=False, prepare=False, mean_face_path="media/20words_mean_face.npy", workers_prepare=-1):
        super(LRW, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle)

        # Params
        self.workers_prepare = multiprocessing.cpu_count() if workers_prepare==-1 else workers_prepare
        self.crop_mouth = crop_mouth
        self.mean_face_path = mean_face_path
        self.load_audio = load_audio
        self.load_video = load_video

        # Download Dataset
        if download:
            self.download()

        # Prepare Dataset
        if prepare:
            self.prepare()

        # Mode
        assert mode in ["train", "val", "test"]

        # Class Dict
        self.class_dict = {}
        for i, path in enumerate(sorted(glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*")))):
            c = path.split("/")[-1]
            self.class_dict[i] = c
            self.class_dict[c] = i

        # Paths
        self.paths = glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*", mode, "*[0-9].mp4"))
        for i, path in enumerate(self.paths):
                self.paths[i] = path[:-4]

        # Video Transforms
        self.video_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            layers.Permute(dims=(1, 0, 2, 3)),
            torchvision.transforms.Grayscale(),
            layers.Permute(dims=(1, 0, 2, 3)),
            transforms.NormalizeVideo(mean=img_mean, std=img_std),
            video_transform if video_transform != None else nn.Identity()
        ])

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, n):

        # Load Video
        if self.load_video:
            if self.crop_mouth:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + "_mouth.mp4")
            else:
                video, audio, infos = torchvision.io.read_video(self.paths[n] + ".mp4")
        else:
            video = None

        # Load Audio
        if self.load_audio:
            audio = torchaudio.load(self.paths[n] + ".flac")[0]
        else:
            audio = None

        # Label
        c = self.paths[n].split("/")[-1].split("_")[0]
        label = self.class_dict[c]

        # Preprocessing
        video = self.video_preprocessing(video.permute(3, 0, 1, 2))
        audio = audio.squeeze(dim=0)
        label = torch.tensor(label)

        return video, audio, label

    class PrepareDataset:

        def __init__(self, paths, mean_face_path):
            self.paths = paths
            self.lip_crop = transforms.LipDetectCrop(mean_face_landmarks_path=mean_face_path)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):

            file_path = self.paths[idx]

            # Load Video
            video, audio , info = torchvision.io.read_video(file_path.replace(".txt", ".mp4"))

            # Save Audio
            torchaudio.save(file_path.replace(".txt", ".flac"), audio, sample_rate=16000)

            # Extract Landmarks
            landmarks_pathname = file_path.replace(".txt", ".npz").replace("lipread_mp4", "LRW_landmarks")
            person_id = 0
            multi_sub_landmarks = np.load(landmarks_pathname, allow_pickle=True)['data']
            landmarks = [None] * len(multi_sub_landmarks)
            for frame_idx in range(len(landmarks)):
                try:
                    landmarks[frame_idx] = multi_sub_landmarks[frame_idx][int(person_id)]['facial_landmarks']
                except IndexError:
                    continue

            # Interpolate Landmarks
            preprocessed_landmarks = self.lip_crop.landmarks_interpolate(landmarks)

            # Crop
            if not preprocessed_landmarks:
                video = torchvision.transforms.functional.resize(video.permute(3, 0, 1, 2), size=(self.lip_crop.crop_height, self.lip_crop.crop_width)).permute(1, 2, 3, 0)
            else:
                video = self.lip_crop.crop_patch(video.numpy(), preprocessed_landmarks)
                assert video is not None
                video = torch.tensor(video)
         
            # Save Video
            torchvision.io.write_video(filename=file_path.replace(".txt", "_mouth.mp4"), video_array=video, fps=info["video_fps"], video_codec="libx264")
            
            return file_path

    def prepare(self):

        # Prepare
        print("Prepare Dataset")
        dataloader = torch.utils.data.DataLoader(
            self.PrepareDataset(
                paths=glob.glob(os.path.join(self.root, "LRW", "lipread_mp4", "*", "*", "*.txt")),
                mean_face_path=self.mean_face_path
            ),
            batch_size=1,
            num_workers=self.workers_prepare,
            collate_fn=collate_fn.Collate(),
        )
        for batch in tqdm(dataloader):
            pass

class LipBengal(Dataset):

    """ LipBengal visual-only word-level dataset.

    Folder structure (example):
        datasets/
          LipBengal/
            s1/                # speaker id
              WORD_A/
                01.jpg ...
              WORD_B/
                ...
            s2/
              ...

    We implement a 60/20/20 split by speakers in sorted order by default.
    This mirrors LRW's interface and returns (video, None, label) where:
      - video: Tensor (C, T, H, W) after preprocessing
      - label: int class index for the Bengali word
    """

    def __init__(self, batch_size, collate_fn, root="datasets", shuffle=True, mode="train", img_mean=(0.5,), img_std=(0.5,), video_transform=None, speakers_split=None, num_frames=None, indices_path=None, fixed_frames=None, subset_fraction=1.0, subset_seed=0, prepared_only=False):
        super(LipBengal, self).__init__(batch_size=batch_size, collate_fn=collate_fn, root=root, shuffle=shuffle)

        assert mode in ["train", "val", "test"], "mode must be one of train/val/test"
        self.mode = mode
        self.num_frames = num_frames  # optional cap
        self.indices_path = indices_path
        self.fixed_frames = fixed_frames  # if set, enforce fixed temporal length for stacking
        self.prepared_only = prepared_only  # if True, require prepared mouth-crop tensors
        self.subset_fraction = float(subset_fraction) if subset_fraction is not None else 1.0
        self.subset_seed = int(subset_seed) if subset_seed is not None else 0

        lb_root = os.path.join(self.root, "LipBengal")
        if self.indices_path is not None and os.path.isfile(self.indices_path):
            try:
                items = torch.load(self.indices_path, map_location="cpu")
            except Exception as e:
                raise RuntimeError(f"Failed to load indices from {self.indices_path}: {e}")
            # Optionally keep only entries that have a prepared crop path (and file exists)
            if self.prepared_only:
                filtered = []
                miss = 0
                for it in items:
                    p = it.get("prepared")
                    if not p:
                        # Try to infer prepared path from frames hash and either Banglish word label or original FS word
                        try:
                            h = hashlib.sha1()
                            # Canonicalize frame paths to match prepare script hashing (relative 'datasets/LipBengal/...')
                            for fp in it.get("frames", []) or []:
                                try:
                                    idx = fp.find("datasets/LipBengal/")
                                    if idx != -1:
                                        canon = fp[idx:]
                                    else:
                                        canon = fp
                                except Exception:
                                    canon = fp
                                h.update(canon.encode("utf-8"))
                            digest = h.hexdigest()[:16]
                            base = os.path.basename(self.indices_path)
                            split = os.path.splitext(base)[0]
                            speaker = it.get("speaker")
                            word_idx = it.get("word")
                            # Original word from the filesystem (parent dir of first frame)
                            fs_word = None
                            frs = it.get("frames") or []
                            if len(frs) > 0:
                                try:
                                    fs_word = os.path.basename(os.path.dirname(frs[0]))
                                except Exception:
                                    fs_word = None
                            cand1 = os.path.join(self.root, "LipBengal", "prepared", split, speaker, word_idx or "", f"{digest}.pt") if speaker and word_idx else None
                            cand2 = os.path.join(self.root, "LipBengal", "prepared", split, speaker, fs_word or "", f"{digest}.pt") if speaker and fs_word else None
                            for cand in (cand1, cand2):
                                if cand and os.path.isfile(cand):
                                    p = cand
                                    it["prepared"] = p
                                    break
                        except Exception:
                            pass
                    if p and os.path.isfile(p):
                        filtered.append(it)
                    else:
                        miss += 1
                self.items = filtered
                if miss:
                    try:
                        print(f"LipBengal[{self.mode}] prepared_only=True: kept {len(filtered)}/{len(items)}; missing {miss}")
                    except Exception:
                        pass
            else:
                self.items = items

            # Build classes from indices 'word' field to remain consistent with label space used during preprocessing
            word_set = set()
            for it in self.items:
                w = it.get("word")
                if isinstance(w, str) and len(w) > 0:
                    word_set.add(w)
            if not word_set:
                # Fallback to filesystem scan if indices lack words
                speakers = [d for d in sorted(os.listdir(lb_root)) if d.startswith("s") and os.path.isdir(os.path.join(lb_root, d))]
                for sp in speakers:
                    sp_dir = os.path.join(lb_root, sp)
                    for word in os.listdir(sp_dir):
                        wdir = os.path.join(sp_dir, word)
                        if os.path.isdir(wdir):
                            word_set.add(word)
            self.classes = sorted(list(word_set))
            self.class_dict = {c: i for i, c in enumerate(self.classes)}
            self.num_classes = len(self.classes)
        else:
            # Discover speakers
            speakers = [d for d in sorted(os.listdir(lb_root)) if d.startswith("s") and os.path.isdir(os.path.join(lb_root, d))]
            if len(speakers) == 0:
                raise FileNotFoundError(f"No speakers found under {lb_root}")

            # Default 60/20/20 split by speakers in sorted order
            if speakers_split is None:
                n = len(speakers)
                n_train = int(round(0.6 * n))
                n_val = int(round(0.2 * n))
                train_speakers = speakers[:n_train]
                val_speakers = speakers[n_train:n_train + n_val]
                test_speakers = speakers[n_train + n_val:]
                speakers_split = {"train": train_speakers, "val": val_speakers, "test": test_speakers}
            self.speakers_split = speakers_split

            # Build class dict from all speakers to have a stable global mapping
            word_set = set()
            for sp in speakers:
                sp_dir = os.path.join(lb_root, sp)
                for word in os.listdir(sp_dir):
                    wdir = os.path.join(sp_dir, word)
                    if os.path.isdir(wdir):
                        word_set.add(word)
            self.classes = sorted(list(word_set))
            self.class_dict = {c: i for i, c in enumerate(self.classes)}
            self.num_classes = len(self.classes)

            # Gather sample paths for current split
            self.paths = []  # entries are directories for a single clip (speaker/word)
            for sp in self.speakers_split[self.mode]:
                sp_dir = os.path.join(lb_root, sp)
                for word in os.listdir(sp_dir):
                    wdir = os.path.join(sp_dir, word)
                    if os.path.isdir(wdir):
                        # ensure there is at least one jpg frame
                        if len(glob.glob(os.path.join(wdir, "*.jpg"))) > 0:
                            self.paths.append(wdir)

        # Optional subsetting for quicker experiments
        if hasattr(self, "items"):
            total = len(self.items)
            if self.subset_fraction is not None and self.subset_fraction < 1.0 and total > 0:
                keep = max(1, int(round(total * self.subset_fraction)))
                if keep < total:
                    rng = np.random.RandomState(self.subset_seed)
                    idx = rng.permutation(total)[:keep]
                    idx.sort()
                    self.items = [self.items[i] for i in idx]
                    try:
                        print(f"LipBengal[{self.mode}] subset: {keep}/{total} (~{self.subset_fraction*100:.1f}%)")
                    except Exception:
                        pass
        else:
            total = len(self.paths)
            if self.subset_fraction is not None and self.subset_fraction < 1.0 and total > 0:
                keep = max(1, int(round(total * self.subset_fraction)))
                if keep < total:
                    rng = np.random.RandomState(self.subset_seed)
                    idx = rng.permutation(total)[:keep]
                    idx.sort()
                    self.paths = [self.paths[i] for i in idx]
                    try:
                        print(f"LipBengal[{self.mode}] subset: {keep}/{total} (~{self.subset_fraction*100:.1f}%)")
                    except Exception:
                        pass

        # Video preprocessing (grayscale + normalize), optional augment at end
        self.video_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ConvertImageDtype(dtype=torch.float32),
            layers.Permute(dims=(1, 0, 2, 3)),   # (C, T, H, W) -> (T, C, H, W)
            torchvision.transforms.Grayscale(),
            layers.Permute(dims=(1, 0, 2, 3)),   # (T, C, H, W) -> (C, T, H, W)
            transforms.NormalizeVideo(mean=img_mean, std=img_std),
            video_transform if video_transform is not None else nn.Identity(),
        ])

    def __len__(self):
        return len(self.items) if hasattr(self, "items") else len(self.paths)

    def _stack_frames(self, frame_files):
        # Optionally limit number of frames first
        if self.num_frames is not None and len(frame_files) > self.num_frames:
            frame_files = frame_files[: self.num_frames]
        frames = []
        for fp in frame_files:
            img = torchvision.io.read_image(fp)  # (C, H, W), uint8
            frames.append(img)
        if len(frames) == 0:
            frames = [torch.zeros(3, 88, 88, dtype=torch.uint8)]

        # Enforce fixed length by truncating/padding (repeat last frame) if requested
        if isinstance(self.fixed_frames, int) and self.fixed_frames > 0:
            T = len(frames)
            if T >= self.fixed_frames:
                frames = frames[: self.fixed_frames]
            else:
                last = frames[-1]
                frames = frames + [last] * (self.fixed_frames - T)

        video = torch.stack(frames, dim=0)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)   # (C, T, H, W)
        return video

    def _load_frames(self, clip_dir):
        frame_files = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
        return self._stack_frames(frame_files)

    def __getitem__(self, n):
        if hasattr(self, "items"):
            entry = self.items[n]
            word = entry["word"]
            label = torch.tensor(self.class_dict[word], dtype=torch.long)
            # Prefer prepared aligned crops if available
            prepared_path = entry.get("prepared")
            # If not provided in indices, try to infer it from frames via hashing and indices_path split
            if not prepared_path and self.indices_path and entry.get("frames") and entry.get("speaker"):
                try:
                    # Hash function identical to prepare script (_hash_item)
                    h = hashlib.sha1()
                    # Canonicalize paths to relative 'datasets/LipBengal/...' to match prepared-file hashing
                    for pth in entry["frames"]:
                        try:
                            idx = pth.find("datasets/LipBengal/")
                            if idx != -1:
                                canon = pth[idx:]
                            else:
                                canon = pth
                        except Exception:
                            canon = pth
                        h.update(canon.encode("utf-8"))
                    digest = h.hexdigest()[:16]
                    # Infer split name from indices_path (train/val/test)
                    base = os.path.basename(self.indices_path)
                    split = os.path.splitext(base)[0]
                    speaker = entry.get("speaker")
                    fs_word = os.path.basename(os.path.dirname(entry["frames"][0])) if entry.get("frames") else None
                    cand1 = os.path.join(self.root, "LipBengal", "prepared", split, speaker, word, f"{digest}.pt") if speaker and word else None
                    cand2 = os.path.join(self.root, "LipBengal", "prepared", split, speaker, fs_word, f"{digest}.pt") if speaker and fs_word else None
                    # Prefer Banglish folder if present, else fallback to Bengali FS name
                    for cand in (cand1, cand2):
                        if cand and os.path.isfile(cand):
                            prepared_path = cand
                            break
                except Exception:
                    pass
            if prepared_path and os.path.isfile(prepared_path):
                obj = torch.load(prepared_path, map_location="cpu")
                # Supported formats:
                #  - {"frames": uint8 tensor (T, H, W)} from our prepare script
                #  - {"x": float tensor (T, H, W)} from earlier draft
                if isinstance(obj, dict) and "frames" in obj:
                    x = obj["frames"]  # (T, H, W) uint8
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    # to (C=1, T, H, W) float in [0,1]
                    video = x.unsqueeze(0).permute(0, 1, 2, 3).to(torch.float32) / 255.0
                elif isinstance(obj, dict) and "x" in obj:
                    x = obj["x"]  # (T, H, W) float maybe normalized
                    if isinstance(x, np.ndarray):
                        x = torch.from_numpy(x)
                    # If looks already normalized to [-1,1], shift back to [0,1] before NormalizeVideo
                    # Otherwise assume [0,1]. We won't rescale; NormalizeVideo can handle both.
                    video = x.unsqueeze(0).permute(0, 1, 2, 3).to(torch.float32)
                else:
                    # Fallback to frames
                    frame_files = entry.get("frames", [])
                    video = self._stack_frames(frame_files)
                # Enforce fixed length if requested
                if isinstance(self.fixed_frames, int) and self.fixed_frames > 0:
                    C, T, H, W = video.shape
                    if T > self.fixed_frames:
                        idx = torch.linspace(0, T - 1, steps=self.fixed_frames).round().long()
                        video = video.index_select(1, idx)
                    elif T < self.fixed_frames:
                        pad = video[:, -1:, :, :].expand(C, self.fixed_frames - T, H, W).clone()
                        video = torch.cat([video, pad], dim=1)
            else:
                frame_files = entry.get("frames", [])
                # Attempt to infer prepared path on the fly if not set
                if self.indices_path and entry.get("speaker") and frame_files:
                    try:
                        h = hashlib.sha1()
                        for pth in frame_files:
                            try:
                                idx = pth.find("datasets/LipBengal/")
                                if idx != -1:
                                    canon = pth[idx:]
                                else:
                                    canon = pth
                            except Exception:
                                canon = pth
                            h.update(canon.encode("utf-8"))
                        digest = h.hexdigest()[:16]
                        base = os.path.basename(self.indices_path)
                        split = os.path.splitext(base)[0]
                        speaker = entry.get("speaker")
                        word_idx = entry.get("word")
                        fs_word = os.path.basename(os.path.dirname(frame_files[0]))
                        cand1 = os.path.join(self.root, "LipBengal", "prepared", split, speaker, word_idx or "", f"{digest}.pt") if speaker and word_idx else None
                        cand2 = os.path.join(self.root, "LipBengal", "prepared", split, speaker, fs_word or "", f"{digest}.pt") if speaker and fs_word else None
                        use = None
                        for cand in (cand1, cand2):
                            if cand and os.path.isfile(cand):
                                use = cand
                                break
                        if use:
                            prepared_path = use
                            obj = torch.load(prepared_path, map_location="cpu")
                            if isinstance(obj, dict) and "frames" in obj:
                                x = obj["frames"]
                                if isinstance(x, np.ndarray):
                                    x = torch.from_numpy(x)
                                video = x.unsqueeze(0).permute(0, 1, 2, 3).to(torch.float32) / 255.0
                            elif isinstance(obj, dict) and "x" in obj:
                                x = obj["x"]
                                if isinstance(x, np.ndarray):
                                    x = torch.from_numpy(x)
                                video = x.unsqueeze(0).permute(0, 1, 2, 3).to(torch.float32)
                            else:
                                video = self._stack_frames(frame_files)
                        else:
                            video = self._stack_frames(frame_files)
                    except Exception:
                        video = self._stack_frames(frame_files)
                else:
                    video = self._stack_frames(frame_files)
        else:
            clip_dir = self.paths[n]
            # Label is word directory name
            word = os.path.basename(clip_dir)
            label = torch.tensor(self.class_dict[word], dtype=torch.long)
            # Load frames and preprocess
            video = self._load_frames(clip_dir)
        video = self.video_preprocessing(video)

        # Return as (video, None, label) to mirror LRW triple layout
        return video, None, label
