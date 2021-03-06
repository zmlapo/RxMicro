import datetime

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import seaborn as sns

from tst import Transformer

from src import MicroservicesDataset
from src.utils import compute_loss
from src.visualization.plot_functions import plot_error_distribution, plot_values_distribution, plot_visual_sample


# Training parameters
DATASET_PATH = 'data/data/'
LABELS_PATH = 'data/labels/'
BATCH_SIZE = 8
NUM_WORKERS = 4
LR = 2e-4
EPOCHS = 1000


def LoadDataset():
    microservicesDataset = MicroservicesDataset.MicroservicesDataset(DATASET_PATH, LABELS_PATH)

    dataset_train, dataset_val, dataset_test = random_split(microservicesDataset, (720, 44, 10))

    dataloader_train = DataLoader(dataset_train,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=False
                                )

    dataloader_val = DataLoader(dataset_val,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS
                            )

    dataloader_test = DataLoader(dataset_test,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS
                                )

    return dataloader_train, dataloader_val, dataloader_test, microservicesDataset.get_service_count()


def LoadNetwork(service_count):
    # Model parameters
    d_model = 48 # Latent dim
    q = 8 # Query size
    v = 8 # Value size
    h = 4 # Number of heads
    N = 4 # Number of encoder and decoder to stack
    attention_size = 24 # Attention window size
    dropout = 0.2 # Dropout rate
    pe = 'original' # Positional encoding
    chunk_mode = None

    d_input = service_count # From dataset
    d_output = service_count # From dataset

    # Config
    sns.set()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")


    net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_function = nn.MSELoss()

    return net, optimizer, loss_function, device


def Train():
    model_save_path = f'models/model_{datetime.datetime.now().strftime("%Y_%m_%d__%H%M%S")}.pth'
    val_loss_best = np.inf

    dataloader_train, dataloader_val, _, service_count = LoadDataset()
    net, optimizer, loss_function, device = LoadNetwork(service_count)

    # Prepare loss history
    hist_loss = np.zeros(EPOCHS)
    hist_loss_val = np.zeros(EPOCHS)
    for idx_epoch in range(EPOCHS):
        running_loss = 0
        for idx_batch, (x, y) in enumerate(dataloader_train):
            optimizer.zero_grad()

            # Propagate input
            netout = net(x.to(device))

            # Comupte loss
            loss = loss_function(y.to(device), netout)

            # Backpropage loss
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()
        
        train_loss = running_loss/len(dataloader_train)
        val_loss = compute_loss(net, dataloader_val, loss_function, device).item()
        print(train_loss)
        hist_loss[idx_epoch] = train_loss
        hist_loss_val[idx_epoch] = val_loss
        
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(net.state_dict(), model_save_path)
            
    plt.plot(hist_loss, 'o-', label='train')
    plt.plot(hist_loss_val, 'o-', label='val')
    plt.legend()
    plt.show()
    print(f"model exported to {model_save_path} with loss {val_loss_best:5f}")

def main():
    Train()

if __name__ == "__main__":
    main()

