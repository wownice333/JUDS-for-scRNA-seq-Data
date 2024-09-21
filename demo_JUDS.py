# -*- coding: utf-8 -*-
import numpy as np
import torch
from model.train import train
from utils.arguments import arg_parse
from utils.loadData import load_CSVData,load_h5ad
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from utils.clusteringPerformance import cluster_acc
from model.model import *
from utils.clusteringPerformance import clusteringMetrics
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = arg_parse()
    dataset_list = ['Xin_pancreas', 'Seg_Pancreas']
    h5_data_list = ['Wang_Lung'] 
    datasetname=args.DS = 'Seg_Pancreas'
    print(datasetname)
    args.DS = datasetname
    if args.DS in dataset_list:
        x, y = load_CSVData(args)
    else:
        x,y = load_h5ad(args)
    n, d = x.shape
    numclass = len(np.unique(y))

    epochs = args.epochs
    repNum = args.repNum

    batch_size = args.batch_size
    lr = args.lr
    lr_milestones = args.lr_milestones
    tol = args.tol
    k = numclass
    x=torch.tensor(x)
    x = x.to(device)
    idx = torch.tensor(list(range(n)))
    dataset = TensorDataset(x, torch.tensor(y), idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    
    if args.eval == True:
        model = FS_CL(d,
                k,
                numclass).to(device)
        saved_path = './weights/JUDS_'+datasetname+'.pkl'
        model.load_state_dict(torch.load(saved_path, map_location=device))
        model.eval()
        middle, _, tmp_q = model(x)
        tmp_q = tmp_q.data
        p = target_distribution(tmp_q)
        p = p

        data = x.cpu().detach().numpy()
        label = y


        # Evaluate clustering performance
        y_pred = tmp_q.cpu().detach().numpy().argmax(1)

        ACC = cluster_acc(y, y_pred)
        NMI = nmi_score(y, y_pred)
        ARI = ari_score(y, y_pred)
        print('ACC: {:.4f}'.format(ACC))
        print('NMI: {:.4f}'.format(NMI))
        print('ARI: {:.4f}'.format(ARI))
        with open('./result/' + datasetname + '_result.txt', 'a') as f:
            f.write('Load saved weights file:\n')
            f.write('ACC: {:.4f}\n'.format(ACC))
            f.write('NMI: {:.4f}\n'.format(NMI))
            f.write('ARI: {:.4f}\n'.format(ARI))

    else:
        para_set = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        for para in para_set: 
            beta = args.beta = para # Coefficient of Clustering Loss
            for para_2 in para_set:
                alpha = args.alpha =  para_2  # L2,1-norm
                ACCList = np.zeros((repNum, 1))
                NMIList = np.zeros((repNum, 1))
                ARIList = np.zeros((repNum, 1))

                ACC_MEAN = np.zeros((1, 2))
                NMI_MEAN = np.zeros((1, 2))
                ARI_MEAN = np.zeros((1, 2))

                for i in range(1, repNum + 1):
                    pretrain_model = ae(d,k).to(device)
                    model = FS_CL(d,
                            k,
                            numclass).to(device)
                    ACC, NMI, ARI = train(data_loader, model, pretrain_model, numclass, epochs, alpha,
                                                            beta, k, lr, lr_milestones,
                                                            device, datasetname, args)
                    ACCList[i - 1, :] = ACC
                    NMIList[i - 1, :] = NMI
                    ARIList[i - 1, :] = ARI

                ACC_MEAN[0, :] = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
                NMI_MEAN[0, :] = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
                ARI_MEAN[0, :] = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)

                print('ACC: {:.4f}'.format(ACC_MEAN))
                print('NMI: {:.4f}'.format(NMI_MEAN))
                print('ARI: {:.4f}'.format(ARI_MEAN))
                with open('./result/' + datasetname + '_result.txt', 'a') as f:
                    f.write('Statistic Results:\n')
                    f.write('Coefficient of L2,1-norm:' + str(alpha) + '\n')
                    f.write('Coefficient of Clustering Loss:' + str(beta) + '\n')
                    f.write('ACC_List:' + str(ACCList) + '\n')
                    f.write('NMI_List:' + str(NMIList) + '\n')
                    f.write('ARI_List:' + str(ARIList) + '\n')
                    f.write('ACC: {:.4f}\n'.format(ACC_MEAN))
                    f.write('NMI: {:.4f}\n'.format(NMI_MEAN))
                    f.write('ARI: {:.4f}\n'.format(ARI_MEAN))
