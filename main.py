#%%
import os, sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import trange

from models import MatrixFactorization
from dataset import MovieRatings

import warnings


def plot_save(pd_files, date):

    if not os.path.exists('./results'):
        os.mkdir('./results')

    plt.figure()
    plt.title("Loss Time Series")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    pd_files.plot(marker=".")
    plt.show()

    plt.savefig('./results/' + date + '.png')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_ratio', type=float, default=0.1, help='')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--num_layers_for_mf', type=int, default=0)
    parser.add_argument('--l2_lambda', type=float, default=0.001)
    parser.add_argument('--plot', type=str, default='')
    
    args = parser.parse_args() 
    
    warnings.simplefilter(action='ignore', category=FutureWarning)

    data = MovieRatings('../datas/ml-latest-small/ratings.csv')

    path_name = 'results/movielens/'

    val_size = int(len(data) * args.val_ratio)
    train_size = len(data) - val_size
    train, val = random_split(data, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }
        
    dataset_sizes = {'train': train_size, 'val': val_size}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MatrixFactorization(int(data[:, 0].max().item())+1, int(data[:, 1].max().item())+1, args.num_layers_for_mf, num_latent=30).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    bar = trange(args.num_epochs)
    epoch_loss = {'train': 0, 'val': 0}
    dft = pd.DataFrame(columns=['train_rmse', 'val_rmse'])

    for epoch in bar:
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            bar.set_description(f'Epoch {epoch} {phase}'.ljust(20))

            # Iterate over data.
            for batch in dataloaders[phase]:
                user_ids = batch[:, 0].to(device)
                movie_ids = batch[:, 1].to(device)
                ratings = batch[:, 2].float().to(device)
                #print(user_ids)
                #print(movie_ids)
                #print(ratings)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, output_user, output_item = model(user_ids.type('torch.LongTensor'), movie_ids.type('torch.LongTensor'))
                    preds = torch.round(outputs)
                    loss = criterion(outputs, ratings)
                    l2_norm = sum(u.pow(2.0).sum() for u in output_user) + sum(v.pow(2.0).sum() for v in output_item)
                    loss = loss + args.l2_lambda * l2_norm

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * user_ids.size(0)
                
            epoch_loss[phase] = running_loss / dataset_sizes[phase]
            bar.set_postfix(train_loss=f'{epoch_loss["train"]:0.5f}', val_loss=f'{epoch_loss["val"]:0.5f}')
            dft.loc[epoch, f'{phase}_rmse'] = epoch_loss[phase]
    
    plot_save(dft, args.plot)
