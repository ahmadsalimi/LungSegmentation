from discriminator.models import MaskDiscriminator
from discriminator.trainer import train_discriminator, evaluate_discriminator
from sys import argv
from contextlib import redirect_stdout
import logging
import torch
from os import path
from time import time
import traceback 



if __name__ == "__main__":
    if len(argv) != 2:
        print("Usage: python train_discriminator.py config_file")
    else:
        config_file = argv[1]
        with open(config_file) as config:
            exec(config.read())

        data_root_path = config.data_root_path
        model_store_directory = config.model_store_directory
        log_directory = config.log_directory
        epochs = config.epochs
        batch_size = config.batch_size
        batch_norms = config.batch_norms
        learning_rate = config.learning_rate
        sample_per_epoch = config.sample_per_epoch

        with open(path.join(log_directory, "epoch_log.log"), "w", buffering=1) as handle:
            with redirect_stdout(handle):
                try:
                    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
                    print(f"Device is {device}", flush=True)
                    discriminator = MaskDiscriminator(batch_norms).to(device)
                    optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

                    train_loss = []
                    train_TP = []
                    train_TN = []
                    val_loss = []
                    val_TP = []
                    val_TN = []
                    
                    for e in range(epochs):
                        print(f'\nEpoch {e+1}/{epochs}:', flush=True)

                        t = time()

                        with open(path.join(log_directory, f"inner_epoch_{e+1:03d}.log"), "w", buffering=1) as inner_handle:
                            with redirect_stdout(inner_handle):
                                epoch_train_loss, epoch_train_TP, epoch_train_TN = train_discriminator(discriminator, optimizer, data_root_path, batch_size, sample_per_epoch, device)
                                epoch_val_loss, epoch_val_TP, epoch_val_TN = evaluate_discriminator(discriminator, data_root_path, device)
                        
                        train_loss.append(epoch_train_loss)
                        train_TP.append(epoch_train_TP)
                        train_TN.append(epoch_train_TN)
                        val_loss.append(epoch_val_loss)
                        val_TP.append(epoch_val_TP)
                        val_TN.append(epoch_val_TN)

                        torch.save(discriminator.state_dict(), path.join(model_store_directory, f"model-e{e+1:03d}.pt"))
                        torch.save(optimizer.state_dict(), path.join(model_store_directory, f"optim-e{e+1:03d}.pt"))
                        torch.save({
                            "train_loss": train_loss,
                            "train_TP": train_TP,
                            "train_TN": train_TN,
                            "val_loss": val_loss,
                            "val_TP": val_TP,
                            "val_TN": val_TN,
                        }, path.join(model_store_directory, f"train-history.pt"))

                        print(f'Epoch {e+1:03} finished in {time() - t:.2f}s', flush=True)
                        print(f'Train loss: {epoch_train_loss:.2e}', flush=True)
                        print(f'Train TP: {epoch_train_TP:.2f}%', flush=True)
                        print(f'Train TN: {epoch_train_TN:.2f}%', flush=True)
                        print(f'Val loss: {epoch_val_loss:.2e}', flush=True)
                        print(f'Val TP: {epoch_val_TP:.2f}%', flush=True)
                        print(f'Val TN: {epoch_val_TN:.2f}%', flush=True)
                        print('---------------------------------------', flush=True)

                except Exception as e:
                    traceback.print_exc(file=handle)