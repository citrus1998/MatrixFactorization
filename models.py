import torch
import torch.nn as nn
import torch.nn.functional as F


def net_maker(io_layer, dropout):
    net = []

    for i_layer, o_layer in zip(io_layer, io_layer[1:]):
        net.extend([nn.Linear(i_layer, o_layer), nn.ReLU(), nn.Dropout(p=dropout)])

    for _ in range(2):
        net.pop()

    return nn.Sequential(*net)

class SingularValueDecomposition(nn.Module):
    def __init__(self, num_users, num_items, num_layers):
        super(SingularValueDecomposition, self).__init__()
        self.svd = torch.svd(torch.Tensor(num_users, num_items))
        self.num_layers = num_layers

    def forward(self):
        self.user_features, self.singlar, self.item_features = self.svd()



class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_layers, num_latent=30):
        super(MatrixFactorization, self).__init__()
        self.user_features = nn.Embedding(num_users, num_latent)
        self.item_features = nn.Embedding(num_items, num_latent)
        nn.init.normal_(self.user_features.weight, 0, 0.1)
        nn.init.normal_(self.item_features.weight, 0, 0.1)

        self.u_net = net_maker([num_latent, num_latent], 0.0)
        self.i_net = net_maker([num_latent, num_latent], 0.0)

        self.num_layers = num_layers


    def forward(self, user_ids, item_ids):
        # Layer for users
        users_latent = self.user_features(user_ids)
        for _ in range(self.num_layers):
            users_latent = F.relu(self.u_net(users_latent))
        # Layer for items
        items_latent = self.item_features(item_ids)
        for _ in range(self.num_layers):
            items_latent = F.relu(self.i_net(items_latent))
        # Ratings
        ratings = (users_latent * items_latent).sum(dim=1)

        return F.relu(ratings), users_latent, items_latent