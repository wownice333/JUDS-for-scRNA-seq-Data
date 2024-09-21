import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam


class ae(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.fs_encoder = nn.Sequential(
            nn.Linear(input_dim, 500, bias=True),
            # nn.LeakyReLU(),
            nn.Tanh(),
        )
        self.z_layer = nn.Linear(500, hidden_dim)
        self.fs_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 500, bias=True),
            # nn.LeakyReLU(),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(500, input_dim, bias=True)

    def forward(self, x):
        encoder = self.fs_encoder(x)
        middle = self.z_layer(encoder)
        decoder = self.fs_decoder(middle)
        output = self.output_layer(decoder)
        return middle, output

class FS_CL(nn.Module):

    def __init__(self,
                 n_input,
                 n_z,
                 n_clusters,
                 pretrain_path='data_v1/ae_mnist.pkl'):
        super(FS_CL, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path
        self.n_input=n_input

        self.ae = ae(
            input_dim=n_input,
            hidden_dim=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    
    def forward(self, x):
        middle, output = self.ae(x)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(middle.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return middle, output, q



def pretrain_ae(model, pretrain_path, train_loader, lr, device):
    '''
    pretrain autoencoder
    '''
    optimizer = Adam(model.parameters(), lr=lr)
    loss_mean = nn.MSELoss(reduction='mean')
    for epoch in range(100):
        total_loss = 0.
        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            z, x_bar = model(x)

            loss = loss_mean(x_bar, x)
            total_loss += loss
            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), pretrain_path)
    print("model saved to {}.".format(pretrain_path))

class Clustering_Layer(nn.Module):
    def __init__(self, n_clusters, hidden_dim):
        super().__init__()
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, hidden_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.alpha = 1.0

    def forward(self, z):
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class model_param_init(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert isinstance(model, nn.Module), 'model not a class nn.Module'
        self.net = model
        self.initParam()
    def initParam(self):
        for param in self.net.parameters():
            nn.init.uniform_(param, a=0, b=1)






