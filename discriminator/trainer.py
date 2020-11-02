from discriminator.dataloader import get_data_loader
import torch
import torch.nn.functional as F
from math import nan


def train_discriminator(model, optimizer, root_path, batch_size, sample_per_epoch, device):
    # BATCH SIZE IS 1
    epoch_loss = 0
    all_positives = 0
    all_negatives = 0
    true_positives = 0
    true_negatives = 0

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
        
        
        print(f'[Train] Iteration {iter + 1:3d} - loss: {epoch_loss / (iter + 1):.2e} - TP: {true_positives * 100. / all_positives:.2f}% - TN: {true_negatives * 100. / all_negatives:.2f}', flush=True)
    
    if iter - last_optimized != 0:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss /= iter + 1
    true_positives *= 100. / all_positives
    true_negatives *= 100. / all_negatives
    return epoch_loss, true_positives, true_negatives

def evaluate_discriminator(model, root_path, device):
    # BATCH SIZE IS 1
    epoch_loss = 0
    all_positives = 0
    all_negatives = 0
    true_positives = 0
    true_negatives = 0

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

            print(f'[Valid] Iteration {iter + 1:3d} - loss: {epoch_loss / (iter + 1):.2e} - TP: {true_positives * 100. / all_positives:.2f}% - TN: {true_negatives * 100. / all_negatives:.2f}', flush=True)
        
        epoch_loss /= iter + 1
        true_positives *= 100. / all_positives
        true_negatives *= 100. / all_negatives
        
        return epoch_loss, true_positives, true_negatives