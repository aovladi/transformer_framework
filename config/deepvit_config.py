import time
from dataclasses import dataclass
from typing import Tuple
import os

import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from torch import distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from vit_pytorch.deepvit import DeepViT, Residual
import torchvision.models as models

from .base_config import base_config, fsdp_checkpointing_base, get_policy_base

if os.getenv('IMAGENET'):
    TRAIN_DATA_PATH = "/datasets01/imagenet_full_size/061417/train/"
    TEST_DATA_PATH = "/datasets01/imagenet_full_size/061417/val/"
    NUM_CLASSES=1000
else:
    TRAIN_DATA_PATH = "/data/home/andolga/data/imagenette2-320/train/"
    TEST_DATA_PATH = "/data/home/andolga/data/imagenette2-320/val/"
    NUM_CLASSES=10

@dataclass
class train_config(base_config):

    # model
    model_name = "60M"

    # available models - name is ~ num params
    # 60M
    # 500M
    # 750M
    # 1B
    # 1.5B
    # 2B
    # 2.5B
    # 3B
    # 8B

    # checkpoint models
    save_model_checkpoint: bool = True
    load_model_checkpoint: bool = False
    checkpoint_type = StateDictType.FULL_STATE_DICT
    dist_checkpoint_root_folder="distributed_checkpoints"
    dist_checkpoint_folder = "DeepVit_local_checkpoint"
    model_save_name = "deepvit-"
    checkpoint_folder = "training_checkpoints_slowmo"
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # optimizers load and save
    save_optimizer: bool = False
    load_optimizer: bool = False
    optimizer_name: str = "Adam"
    optimizer_checkpoint_file: str = "Adam-deepvit-1.pt"

    checkpoint_model_filename: str = "deepvit-1.pt"


def build_model(model_size: str):
    model_args = dict()
    if model_size == "60M":
        model_args = {
            "image_size": 224,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
            "dim": 128,#32,#1024,
            "depth": 10,#1,
            "heads": 6,#1,
            "mlp_dim": 256,#64,#2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "500M":
        model_args = {
            "image_size": 224,
            "patch_size": 32,
            "num_classes": NUM_CLASSES,
            "dim": 384,
            "depth": 59,
            "heads": 8,
            "mlp_dim": 1152,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "750M":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 89,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }

    if model_size == "1B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 118,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "1.5B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 177,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "2B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 236,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }

    if model_size == "2.5B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 296,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }

    if model_size == "3B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 357,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }
    if model_size == "8B":
        model_args = {
            "image_size": 256,
            "patch_size": 32,
            "num_classes": 1000,
            "dim": 1024,
            "depth": 952,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1,
            "emb_dropout": 0.1,
        }

    if os.getenv('RESNET') is not None:
        arch="resnet101"
        model = models.__dict__[arch](num_classes=NUM_CLASSES)
    elif os.getenv('REGNET') is not None:
        arch="regnet_y_16gf"    # "regnet_y_32gf" OOM
        model = models.__dict__[arch](num_classes=NUM_CLASSES)
    else:
        model = DeepViT(**model_args)

    return model


class GeneratedDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super()
        self._input_shape = kwargs.get("input_shape", [3, 256, 256])
        self._input_type = kwargs.get("input_type", torch.float32)
        self._len = kwargs.get("len", 1000000)
        self._num_classes = kwargs.get("num_classes", 1000)

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rand_image = torch.randn(self._input_shape, dtype=self._input_type)
        label = torch.tensor(data=[index % self._num_classes], dtype=torch.int64)
        return rand_image, label

def one_hot_encoder(target):
    _num_classes = 10
    one_hot = torch.eye(_num_classes)[target]
    return one_hot

def get_dataset(path=TRAIN_DATA_PATH, train=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        input_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        input_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return torchvision.datasets.ImageFolder(path, transform=input_transform)#, target_transform=one_hot_encoder)
    #return GeneratedDataset()


def get_policy():
    return get_policy_base({Residual})


def fsdp_checkpointing(model):
    return fsdp_checkpointing_base(model, Residual)


def train(model, data_loader, torch_profiler, optimizer, memmax, local_rank, tracking_duration, total_steps_to_run):
    cfg = train_config()
    loss_function = torch.nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(data_loader)), colour="blue", desc="Train Epoch"
        )
    for batch_index, (inputs, target) in enumerate(data_loader, start=1):
        inputs, targets = inputs.to(torch.cuda.current_device()), torch.squeeze(
            target.to(torch.cuda.current_device()), -1
        )
        if optimizer:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()

        if optimizer:
            optimizer.step()


        # update durations and memory tracking
        if local_rank == 0:
            mini_batch_time = time.perf_counter() - t0
            if tracking_duration:
                tracking_duration.append(mini_batch_time)
            if memmax:
                memmax.update()
        """
        if (
            batch_index % cfg.log_every == 0
            and torch.distributed.get_rank() == 0
            and batch_index > 1
        ):
            print(
                f"step: {batch_index-1}: time taken for the last {cfg.log_every} steps is {mini_batch_time}, loss is {loss}"
            )
        """

        # reset timer
        t0 = time.perf_counter()
        if torch_profiler is not None:
            torch_profiler.step()
        if local_rank == 0:
                inner_pbar.update(1)
        #if batch_index > total_steps_to_run:
        #    break

def validation(model, local_rank, rank, val_loader, world_size):

    cfg = train_config()

    epoch_val_accuracy = 0
    epoch_val_loss = 0
    model.eval()
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )

    loss_function = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(val_loader):
            inputs, target = inputs.to(torch.cuda.current_device()), target.to(torch.cuda.current_device())
            output = model(inputs)
            loss = loss_function(output, target)

            # measure accuracy and record loss
            acc = (output.argmax(dim=1) == target).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += loss / len(val_loader)

            if rank == 0:
                inner_pbar.update(1)

    metrics = torch.tensor([epoch_val_loss, epoch_val_accuracy]).to(torch.cuda.current_device())
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    metrics /=world_size
    epoch_val_loss, epoch_val_accuracy = metrics[0], metrics[1]
    if rank == 0:
        print(
            f"val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )
    return
