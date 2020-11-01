from discriminator.dataloader import get_data_loader
import torch
import torch.nn.functional as F
from math import nan


def train_discriminator(model, optimizer, root_path, batch_size):
    # BATCH SIZE IS 1
    epoch_loss = 0
    all_positives = 0
    all_negatives = 0
    true_positives = 0
    true_negatives = 0

    last_optimized = -1

    model.train()

    progress = get_data_loader(root_path, "Train", shuffle=True)
    for iter, (b_X, b_y) in progress:
        # b_X   B   2   H   64  64
        # b_y   B
        images = torch.tensor(b_X).cuda().float()
        target = torch.tensor(b_y).cuda().float()

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

        progress.set_postfix(
            loss=f"{epoch_loss / (iter + 1):.4e}",
            TP=f"{true_positives * 100. / all_positives if all_positives != 0 else nan :.2f}",
            TN=f"{true_negatives * 100. / all_negatives if all_negatives != 0 else nan :.2f}",
        )
    
    if iter - last_optimized != batch_size:
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss /= iter + 1
    true_positives /= all_positives * 100.
    true_negatives /= all_negatives * 100.

    progress.close()

    return epoch_loss, true_positives, true_negatives

def evaluate_discriminator(model, root_path):
    # BATCH SIZE IS 1
    epoch_loss = 0
    all_positives = 0
    all_negatives = 0
    true_positives = 0
    true_negatives = 0

    progress = get_data_loader(root_path, "Valid", shuffle=False)
    with torch.no_grad():
        model.eval()
        for iter, (b_X, b_y) in progress:
            images = torch.tensor(b_X).cuda().float()
            target = torch.tensor(b_y).cuda().float()

            prediction = model(images)

            loss = F.binary_cross_entropy(prediction, target)
            decision = (prediction.detach().cpu().numpy() > 0.5)

            epoch_loss += float(loss)
            all_positives += b_y.sum()
            all_negatives += (~b_y).sum()
            true_positives += (decision[b_y]).sum()
            true_negatives += (~decision[~b_y]).sum()

            progress.set_postfix(
                loss=f"{epoch_loss / (iter + 1):.4e}",
                TP=f"{true_positives * 100. / all_positives if all_positives != 0 else nan :.2f}",
                TN=f"{true_negatives * 100. / all_negatives if all_negatives != 0 else nan :.2f}",
            )
        
        epoch_loss /= iter + 1
        true_positives /= all_positives * 100.
        true_negatives /= all_negatives * 100.

        progress.close()
        
        return epoch_loss, true_positives, true_negatives