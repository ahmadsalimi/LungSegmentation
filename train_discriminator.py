from discriminator.models import MaskDiscriminator
from discriminator.trainer import train_discriminator, evaluate_discriminator
from sys import argv
from contextlib import redirect_stdout
import logging
import torch
from os import path


if __name__ == "__main__":
    if len(argv) != 6:
        print("Usage: python train_discriminator.py data_root_path model_store_directory epochs log_file_path batch_size")
    else:
        data_root_path = argv[1]
        model_store_directory = argv[2]
        epochs = int(argv[3])
        log_file_path = argv[4]
        batch_size = int(argv[5])

        with open(log_file_path, "w", buffering=1) as handle:
            with redirect_stdout(handle):
                try:
                    device = "cuda" if torch.cuda.device_count > 0 else "cpu"
                    print(f"device is {device}")
                    discriminator = MaskDiscriminator().to(device)
                    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

                    train_loss = []
                    train_TP = []
                    train_TN = []
                    val_loss = []
                    val_TP = []
                    val_TN = []
                    
                    for e in range(epochs):
                        print(f'\nEpoch {e+1}/{epochs}:', flush=True)

                        epoch_train_loss, epoch_train_TP, epoch_train_TN = train_discriminator(discriminator, optimizer, data_root_path, batch_size)
                        epoch_val_loss, epoch_val_TP, epoch_val_TN = evaluate_discriminator(discriminator, data_root_path)
                        
                        train_loss.append(epoch_train_loss)
                        train_TP.append(epoch_train_TP)
                        train_TN.append(epoch_train_TN)
                        val_loss.append(epoch_val_loss)
                        val_TP.append(epoch_val_TP)
                        val_TN.append(epoch_val_TN)

                        torch.save(discriminator.state_dict(), path.join(model_store_directory, f"model-e{epoch:03d}.pt"))
                        torch.save(optimizer.state_dict(), path.join(model_store_directory, f"optim-e{epoch:03d}.pt"))
                        torch.save({
                            "train_loss": train_loss,
                            "train_TP": train_TP,
                            "train_TN": train_TN,
                            "val_loss": val_loss,
                            "val_TP": val_TP,
                            "val_TN": val_TN,
                        }, path.join(model_store_directory, f"history"))

                except:
                    pass