from discriminator.dataloader import get_data_loader
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from math import nan
import numpy as np
from collections.abc import Iterable
from typing import Tuple, List


def train_discriminator(model: nn.Module, optimizer: optim.Optimizer, root_path: str, batch_size: int, sample_per_epoch: int, device: torch.device) -> Tuple[float, float, float]:
    # BATCH SIZE IS 1
    epoch_loss = .0
    all_positives = .0
    all_negatives = .0
    true_positives = .0
    true_negatives = .0

    last_optimized = -1

    model.train()

    progress = get_data_loader(root_path, "Train", sample_per_epoch=sample_per_epoch, shuffle=True)
    for iter, (b_X, b_y) in progress:
        # b_X   B   2   H   64  64
        # b_y   B
        images = torch.tensor(b_X).to(device).float()
        target = torch.tensor(b_y).to(device).float()

        prediction = model(images)

        loss = F.binary_cross_entropy(prediction, target)
        decision = (prediction.detach().cpu().numpy() > 0.5)

        epoch_loss += float(loss)
        all_positives += b_y.sum()
        all_negatives += (~b_y).sum()
        true_positives += (decision[b_y]).sum()
        true_negatives += (~decision[~b_y]).sum()

        loss.backward()

        if iter - last_optimized == batch_size:
            last_optimized = iter
            optimizer.step()
            optimizer.zero_grad()
        
        
        print(f'[Train] Iteration {iter + 1:3d} - loss: {epoch_loss / (iter + 1):.2e} - TP: {true_positives * 100. / all_positives:.2f}% - TN: {true_negatives * 100. / all_negatives:.2f}%', flush=True)
    
    if iter - last_optimized != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss /= iter + 1
    true_positives *= 100. / all_positives
    true_negatives *= 100. / all_negatives
    return epoch_loss, true_positives, true_negatives

def evaluate_discriminator(model: nn.Module, root_path: str, device: torch.device) -> Tuple[float, float, float]:
    # BATCH SIZE IS 1
    epoch_loss = .0
    all_positives = .0
    all_negatives = .0
    true_positives = .0
    true_negatives = .0

    progress = get_data_loader(root_path, "Valid")
    with torch.no_grad():
        model.eval()
        for iter, (b_X, b_y) in progress:
            images = torch.tensor(b_X).to(device).float()
            target = torch.tensor(b_y).to(device).float()

            prediction = model(images)

            loss = F.binary_cross_entropy(prediction, target)
            decision = (prediction.detach().cpu().numpy() > 0.5)

            epoch_loss += float(loss)
            all_positives += b_y.sum()
            all_negatives += (~b_y).sum()
            true_positives += (decision[b_y]).sum()
            true_negatives += (~decision[~b_y]).sum()

            print(f'[Valid] Iteration {iter + 1:3d} - loss: {epoch_loss / (iter + 1):.2e} - TP: {true_positives * 100. / all_positives:.2f}% - TN: {true_negatives * 100. / all_negatives:.2f}%', flush=True)
        
        epoch_loss /= iter + 1
        true_positives *= 100. / all_positives
        true_negatives *= 100. / all_negatives
        
        return epoch_loss, true_positives, true_negatives


def extract_patches(b_X: Iterable[np.ndarray], patch_size: int) -> np.ndarray:
    samples: List[np.ndarray] = []
    for sample in b_X:
        # sample    3   H   256 256
        max_choice: int = max(1, sample.shape[1] - 63)
        patch_starts: np.ndarray = np.random.choice(max_choice, size=patch_size, replace=max_choice<patch_size)
        array = np.array([sample[start:start+64] for start in patch_starts])    # P 3   64  256 256
        samples.append(array)
    
    return np.array(samples)    # B P   3   64  256 256


def calculate_loss(prediction: torch.Tensor, target: torch.Tensor, B: int, P: int) -> torch.Tensor:
    # prediction:   B*P
    # target:       B

    patchwise_prediction: torch.Tensor = prediction.reshape(B, P)             # B P
    batchwise_prediction: torch.Tensor = patchwise_prediction.amax(dim=1)     # B

    loss = F.binary_cross_entropy(batchwise_prediction, target)

    return loss