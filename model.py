import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import  sparse
n_items = 20108
n_users = 116677
def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data
class Encoder(nn.Module):
    def __init__(self,  dropout=0.5, encoder_dims=[20108, 600, 200]):
        super(Encoder, self).__init__()
        self.encoder_dims = encoder_dims
        self.dropout = nn.Dropout(p= dropout, inplace=False)
        self.linear_1 = nn.Linear(self.encoder_dims[0], self.encoder_dims[1], bias=True)
        self.linear_2 = nn.Linear(self.encoder_dims[1], self.encoder_dims[2] * 2, bias=True)
        self.tanh = nn.Tanh()

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        mu_q, logvar_q = torch.chunk(x, chunks=2, dim=1)
        return mu_q, logvar_q


class Decoder(nn.Module):
    def __init__(self, decoder_dims=[200, 600, 20108]):
        super(Decoder, self).__init__()
        self.decoder_dims = decoder_dims
        self.linear_1 = nn.Linear(self.decoder_dims[0], self.decoder_dims[1], bias=True)
        self.linear_2 = nn.Linear(self.decoder_dims[1], self.decoder_dims[2], bias=True)
        self.tanh = nn.Tanh()

        for module_name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        return x

class MultiVAE(nn.Module):
    def __init__(self, dropout=0.5, encoder_dims=[20108, 600, 200], decoder_dims=[200, 600, 20108]):
        super(MultiVAE, self).__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        self.encoder = Encoder(dropout=dropout, encoder_dims=self.encoder_dims)
        self.decoder = Decoder(decoder_dims=self.decoder_dims)

    def forward(self, x):
        x = nn.functional.normalize(x, p=2, dim=1)
        mu_q, logvar_q = self.encoder.forward(x)
        std_q = torch.exp(0.5 * logvar_q)
        KL = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q ** 2 - 1), dim=1))
        epsilon = torch.randn_like(std_q, requires_grad=False)
        if self.training:
          sampled_z = mu_q + epsilon * std_q
        else:
          sampled_z = mu_q

        logits = self.decoder.forward(sampled_z)

        return logits, KL
if __name__ == '__main__':
    load_train_data("train.csv")