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

# Other
import os
import glob
import contextlib
import socket
import sentencepiece as spm

def find_last_checkpoint(callback_path, return_full_path=False):
    """Return the most recent checkpoint name.

    Original implementation picked the file with the largest global step only,
    which can select an earlier epoch if later epochs reset/alter step logic.
    We now parse both epoch and step from the naming pattern
    'checkpoints_epoch_<E>_step_<S>.ckpt' and choose by:
        1. Highest epoch number
        2. If tie, highest step number
    """

    pattern = os.path.join(callback_path, "checkpoints_epoch_*_step_*.ckpt")
    checkpoints = glob.glob(pattern)
    last_checkpoint = None
    best_epoch = -1
    best_step = -1
    for full in checkpoints:
        name = os.path.basename(full)
        parts = name.replace('.ckpt','').split('_')
        # Expected: ['checkpoints','epoch','<E>','step','<S>']
        try:
            epoch_idx = parts.index('epoch') + 1 if 'epoch' in parts else 2
            step_idx = parts.index('step') + 1 if 'step' in parts else len(parts)-1
            epoch = int(parts[epoch_idx])
            step = int(parts[step_idx])
        except Exception:
            # Fallback to previous behavior if pattern unexpected
            try:
                step = int(parts[-1])
                epoch = -1
            except Exception:
                continue
        if (epoch > best_epoch) or (epoch == best_epoch and step > best_step):
            best_epoch = epoch
            best_step = step
            last_checkpoint = name

    if last_checkpoint and return_full_path:
        last_checkpoint = os.path.join(callback_path, last_checkpoint)
    return last_checkpoint

def load_model(args):

    # Model Device
    device = torch.device("cuda:" + str(args.rank) if torch.cuda.is_available() and not args.cpu else "cpu")
    if "cuda" in str(device):
        print("Rank {} device: {}, {}, {}MB".format(args.rank, device, torch.cuda.get_device_properties(device).name, int(torch.cuda.get_device_properties(device).total_memory // 1e6)))
        args.num_gpus = torch.cuda.device_count()
    else:
        print("Rank {} device: {}".format(args.rank, device))
        args.num_gpus = 1

    # Barrier
    if args.distributed:
        torch.distributed.barrier()

    # Set Model Device
    model = args.config.model.to(device)

    for module in model.modules():
        module.to(device)

    # Set EMA Model
    if hasattr(args.config, "ema_tau") and args.rank == 0:
        model.set_ema(args.config.ema_tau)

    # Set Callbacks Path
    args.config.callback_path = getattr(args.config, "callback_path", os.path.join("callbacks", "/".join(args.config_file.replace(".py", "").split("/")[1:])))

    # Last Checkpoint
    if args.load_last:
        last_checkpoint = find_last_checkpoint(args.config.callback_path)
        if last_checkpoint != None:
            args.checkpoint = last_checkpoint

    # Load Checkpoint
    if args.checkpoint is not None:
        ckpt_path = os.path.join(args.config.callback_path, args.checkpoint)
        try:
            # Don't load optimizer when not training to avoid param group mismatches
            model.load(ckpt_path, load_optimizer=(args.mode == 'training'))
        except RuntimeError as e:
            # Fallback: filter out incompatible keys (e.g., classifier head size changed)
            if args.rank == 0:
                print(f"Warning: strict checkpoint load failed ({e}). Attempting filtered load of compatible weights only.")
            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                ckpt_state = checkpoint.get("model_state_dict", {})
                # Normalize keys if saved with DataParallel/Distributed
                normalized = {}
                for k, v in ckpt_state.items():
                    k2 = k
                    if checkpoint.get("is_distributed", False) and not getattr(model, "is_distributed", False):
                        k2 = k2.replace(".module.", ".")
                    normalized[k2] = v

                current_state = model.state_dict()
                filtered = {}
                skipped = []
                for k, v in normalized.items():
                    if k in current_state and hasattr(v, 'shape') and hasattr(current_state[k], 'shape') and tuple(v.shape) == tuple(current_state[k].shape):
                        filtered[k] = v
                    else:
                        skipped.append(k)

                missing = set(current_state.keys()) - set(filtered.keys())
                if args.rank == 0:
                    print(f"Loading {len(filtered)}/{len(current_state)} tensors from checkpoint; skipping {len(skipped)} mismatched keys; {len(missing)} missing in checkpoint.")
                model.load_state_dict(filtered, strict=False)
                # Restore model step if present
                if "model_step" in checkpoint:
                    try:
                        model.model_step.fill_(checkpoint["model_step"])  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if args.rank == 0:
                    print("Filtered checkpoint load succeeded.")
            except Exception as ee:
                raise ee

    # Barrier
    if args.distributed:
        torch.distributed.barrier()

    # Model Summary
    if args.rank == 0:
        model.summary(show_dict=args.show_dict, show_modules=args.show_modules)

    # Distribute Strategy
    if args.distributed:
        if args.rank == 0:
            print("Parallelize model on", args.world_size, "GPUs")
        model.distribute_strategy(args.rank)

    # Parallel Strategy
    if args.parallel and not args.distributed:
        print("Parallelize model on", torch.cuda.device_count(), "GPUs")
        model.parallel_strategy()

    return model

def load_datasets(args):

    # Training Dataset
    if hasattr(args.config, "training_dataset"):

        # DataLoader
        dataset_train = torch.utils.data.DataLoader(
            dataset=args.config.training_dataset,
            batch_size=args.config.training_dataset.batch_size,
            shuffle=False if args.distributed else args.config.training_dataset.shuffle,
            sampler=torch.utils.data.distributed.DistributedSampler(args.config.training_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.config.training_dataset.shuffle) if args.distributed else None,
            num_workers=args.num_workers,
            collate_fn=args.config.training_dataset.collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
            prefetch_factor=4 if args.num_workers and args.num_workers > 0 else None
        )
        
        # Loaded Print
        if args.rank == 0:
            if args.distributed:
                print("Training Dataset: {}, {:,} samples - {:,} batches - batch size {} x {}".format(dataset_train.dataset.__class__.__name__, len(dataset_train.dataset), len(dataset_train), dataset_train.dataset.batch_size, args.num_gpus))
            else:
                print("Training Dataset: {}, {:,} samples - {:,} batches - batch size {}".format(dataset_train.dataset.__class__.__name__, len(dataset_train.dataset), len(dataset_train), dataset_train.dataset.batch_size))

    else:

        dataset_train = None

    # Evaluation Dataset
    if hasattr(args.config, "evaluation_dataset"):

        # Multiple Evaluation datasets
        if isinstance(args.config.evaluation_dataset, list):

            dataset_eval = []
            for dataset in args.config.evaluation_dataset:

                # DataLoader
                dataset_eval.append(torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=args.batch_size_eval if getattr(args, "batch_size_eval", None) else dataset.batch_size,
                    shuffle=False if args.distributed else dataset.shuffle,
                    sampler=torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=dataset.shuffle) if args.distributed else None,
                    num_workers=args.num_workers,
                    collate_fn=dataset.collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
                    prefetch_factor=4 if args.num_workers and args.num_workers > 0 else None
                ))
            
                # Loaded Print
                if args.rank == 0:
                    if args.distributed:
                        print("Evaluation Dataset: {}, {:,} samples - {:,} batches - batch size {} x {}".format(dataset_eval[-1].dataset.__class__.__name__, len(dataset_eval[-1].dataset), len(dataset_eval[-1]), dataset_eval[-1].dataset.batch_size, args.num_gpus))
                    else:
                        print("Evaluation Dataset: {}, {:,} samples - {:,} batches - batch size {}".format(dataset_eval[-1].dataset.__class__.__name__, len(dataset_eval[-1].dataset), len(dataset_eval[-1]), dataset_eval[-1].dataset.batch_size))

        # One Evaluation dataset
        else:

            # DataLoader
            dataset_eval = torch.utils.data.DataLoader(
                dataset=args.config.evaluation_dataset,
                batch_size=args.batch_size_eval if getattr(args, "batch_size_eval", None) else args.config.evaluation_dataset.batch_size,
                shuffle=False if args.distributed else args.config.evaluation_dataset.shuffle,
                sampler=torch.utils.data.distributed.DistributedSampler(args.config.evaluation_dataset, num_replicas=args.world_size,rank=args.rank, shuffle=args.config.evaluation_dataset.shuffle) if args.distributed else None,
                num_workers=args.num_workers,
                collate_fn=args.config.evaluation_dataset.collate_fn,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True if args.num_workers and args.num_workers > 0 else False,
                prefetch_factor=4 if args.num_workers and args.num_workers > 0 else None
            )
            
            # Loaded Print
            if args.rank == 0:
                if args.distributed:
                    print("Evaluation Dataset: {}, {:,} samples - {:,} batches - batch size {} x {}".format(dataset_eval.dataset.__class__.__name__, len(dataset_eval.dataset), len(dataset_eval), dataset_eval.dataset.batch_size, args.num_gpus))
                else:
                    print("Evaluation Dataset: {}, {:,} samples - {:,} batches - batch size {}".format(dataset_eval.dataset.__class__.__name__, len(dataset_eval.dataset), len(dataset_eval), dataset_eval.dataset.batch_size))
    else:
        dataset_eval = None
    
    return dataset_train, dataset_eval

def get_open_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def train_tokenizer(corpus_path, tokenizer_path, vocab_size=256, vocab_type="bpe"):
    spm.SentencePieceTrainer.train(input=corpus_path, model_prefix=tokenizer_path, vocab_size=vocab_size, character_coverage=1.0, model_type=vocab_type, bos_id=-1, eos_id=-1, unk_surface="")
        