import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils.clusteringPerformance import clusteringMetrics
from utils.clusteringPerformance import cluster_acc
from scipy.stats import mode
from model.model import *


def jacobian(inputs, outputs):
    return torch.stack(
        [torch.autograd.grad([outputs[:, i].sum()], [inputs], retain_graph=True, create_graph=True)[0] for i in
         range(outputs.size(1))], dim=-1)


def train(data_loader, model, pretrain_model, numclass, epochs, alpha, beta, k, lr, lr_milestones, device, datasetname, args):

    if not os.path.exists('./pretrain_weight/' + datasetname + '.pkl'):
        pretrain_ae(pretrain_model,path, data_loader, 0.01, device)
    else:
        path = './pretrain_weight/' + datasetname + '.pkl'

    # Load Pretrained Weights
    model.ae.load_state_dict(torch.load(path))
    print('load pretrained ae from', path)

    # Initialize the Clustering Centroids
    hidden=[]
    for idx, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
        batch_hidden, _ = model.ae(batch_x)
        hidden.append(batch_hidden)
    hidden = torch.cat(hidden, dim=0)

    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=lr_milestones, gamma=0.1)

    loss_mean = nn.MSELoss(reduction='mean')
    loss_kl_mean = nn.KLDivLoss()
    best_ACC = -1.
    acc = -1.
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:

        if epoch % 5 == 0 or epoch == epochs or epoch==1:
            U_w=[]
            tmp_q = []
            y=[]
            for idx, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
                model.eval()

                middle, _, batch_q = model(batch_x)
                tmp_q.append(batch_q.data)
                y.append(batch_y)

            tmp_q = torch.cat(tmp_q, dim=0)
            y = torch.cat(y, dim=0).numpy()

            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().detach().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)


            model.train()
            if best_ACC < acc:
                best_ACC = acc
                best_NMI = nmi
                best_ARI = ari

        fs_latent_total = torch.tensor([]).to(device)
        total_loss = 0.
        total_re_loss = 0.
        total_kl_loss = 0.
        total_l21_loss = 0.

        for idx, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
            batch_x = torch.tensor(batch_x, dtype=torch.float, requires_grad=True)
            middle, output, q = model(batch_x)
            
            batch_loss = loss_mean(batch_x, output)
            kl_loss = loss_kl_mean(q.log(), p[batch_idx])

            batch_U_w = torch.sum(torch.abs(jacobian(inputs=batch_x, outputs=middle)), dim=0)
            temp_ = torch.norm(batch_U_w, p='fro', dim=1)
            l21_loss = torch.norm(temp_, p=1)

            loss = batch_loss + alpha * l21_loss + beta * kl_loss

            fs_latent_total = torch.cat((fs_latent_total, middle), dim=0)

            total_loss += loss
            total_l21_loss += alpha * l21_loss
            total_kl_loss += beta * kl_loss
            total_re_loss += batch_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            scheduler.step()
        pbar.set_description(
            "Epoch{}| # Latent Dimension {}, Reconstruction Loss: {:.4}, L2,1 Loss: {:.4}, KLDiv Loss: {:.4}, Total Loss: {:.4}".format(
                epoch,
                k,
                total_re_loss,
                total_l21_loss,
                total_kl_loss,
                total_loss)
        )
    return best_ACC, best_NMI, best_ARI


def train_kmeans(data_loader, model, pretrain_model, x, numclass, epochs, alpha, beta, lr, lr_milestones, device, datasetname, args):
    x=torch.tensor(x)
    x = x.to(device)
    n, d= x.shape
    if not os.path.exists('./pretrain_weight/' + datasetname + '.pkl'):
        pretrain_ae(pretrain_model,path,data_loader,0.01, device)
    else:
        path = './pretrain_weight/' + datasetname + '.pkl'

    # load pretrain weights
    model.ae.load_state_dict(torch.load(path))
    print('load pretrained ae from', path)

    hidden, output = model.ae(x)

    # Initialize the centroids
    kmeans = KMeans(n_clusters=numclass, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=lr_milestones, gamma=0.1)

    loss_mean = nn.MSELoss(reduction='mean')
    loss_kl_mean = nn.KLDivLoss()
    best_ACC = -1.
    acc = -1.
    pbar = tqdm(range(1, epochs + 1))
    select_features=300

    for epoch in pbar:

        if epoch % 1 == 0 or epoch == epochs or epoch==1:
            U_w=torch.zeros(d, numclass).to(device)
            tmp_q = []
            y=[]
            for idx, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
                model.eval()
                batch_x.requires_grad = True
                middle, _, batch_q = model(batch_x)
                batch_U_w = torch.sum(torch.abs(jacobian(inputs=batch_x, outputs=middle)), dim=0)
                U_w+=batch_U_w
                tmp_q.append(batch_q.data)
                y.append(batch_y)
            tmp_q = torch.cat(tmp_q, dim=0)
            y = torch.cat(y, dim=0).numpy()

            # update target distribution p
            p = target_distribution(tmp_q)

            index = np.argsort(np.sum(np.array(np.square(U_w.cpu().detach().numpy())), axis=1))[::-1]
            w_ = index[0:select_features]

            kmeans = KMeans(n_clusters=numclass).fit(x.detach().cpu().numpy()[:, w_])  # Kmeans
            clusters = kmeans.labels_

            cluster_label = np.zeros_like(clusters)
            for i in range(numclass):
                mask = (clusters == i)
                cluster_label[mask] = mode(y[mask])[0]

            acc, nmi, _, ari, _, _, _ = clusteringMetrics(y, cluster_label)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().detach().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred


            model.train()
            if best_ACC < acc:
                best_ACC = acc
                best_NMI = nmi
                best_ARI = ari
 
        fs_latent_total = torch.tensor([]).to(device)
        total_loss = 0.
        total_re_loss = 0.
        total_kl_loss = 0.
        total_l21_loss = 0.

        for idx, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
            batch_x = torch.tensor(batch_x, dtype=torch.float, requires_grad=True)
            middle, output, q = model(batch_x)
            
            batch_loss = loss_mean(batch_x, output)
            
            kl_loss = loss_kl_mean(q.log(), p[batch_idx])
            
            batch_U_w = torch.sum(torch.abs(jacobian(inputs=batch_x, outputs=middle)), dim=0)
            temp_ = torch.norm(batch_U_w, p='fro', dim=1)
            l21_loss = torch.norm(temp_, p=1)

            loss = batch_loss + alpha * l21_loss + beta * kl_loss

            fs_latent_total = torch.cat((fs_latent_total, middle), dim=0)
            total_loss += loss
            total_l21_loss += alpha * l21_loss
            total_kl_loss += beta * kl_loss
            total_re_loss += batch_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        pbar.set_description(
            "Epoch{}| # Selected Features {}, Reconstruction Loss: {:.4}, L2,1 Loss: {:.4}, KLDiv Loss: {:.4}, Total Loss: {:.4}".format(
                epoch,
                select_features,
                total_re_loss,
                total_l21_loss,
                total_kl_loss,
                total_loss)
        )
    return best_ACC, best_NMI, best_ARI