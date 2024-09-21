import math
import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import torch
from scanpy import read_h5ad
from scipy.io import loadmat
from sklearn import preprocessing

import anndata as ad
import scipy


def loadData(args):
    dataset = loadmat(args.path + '/' + args.DS + '.mat')
    if ('X' in dataset):
        x = torch.tensor(dataset['X'][:])
        y = torch.tensor(dataset['Y']).squeeze().numpy()
        if 'csc_matrix' in str(type(x)):
            x = torch.tensor(x.todense())
    elif ('feature' in dataset):
        x = torch.tensor(dataset['feature'][:])
        y = torch.tensor(dataset['label']).squeeze().numpy()
    else:
        x = torch.tensor(dataset['fea'][:])
        y = torch.tensor(dataset['gnd']).squeeze().numpy()
        if 'csc_matrix' in str(type(x)):
            x = torch.tensor(x.todense())
    minmax = preprocessing.MinMaxScaler()
    x = torch.tensor(minmax.fit_transform(x), dtype=torch.float, requires_grad=True)
    # print(type(x))
    # print(type(y))
    return x, y


def load_CSVData(args):
    dataset = pd.read_csv(args.path + '/' + args.DS + '/' + args.DS + '_x.csv', index_col=0)
    chr_name=dataset.index
    
    x = np.transpose(dataset.values)
    # x=preprocessing.scale(x)
    
    # x=np.log2(x+1)
    x=sc.pp.log1p(x)

    x = torch.tensor(x, dtype=torch.float, requires_grad=True)
    label_dataset = pd.read_csv(args.path + '/' + args.DS + '/' + args.DS + '_label.csv', index_col=0)
    label_attr = np.unique(label_dataset.values)
    # print(label_attr)
    y = label_dataset.values


    for i in range(len(label_attr)):
        idx = np.where(y == label_attr[i])
        y[idx] = i
    y = np.array(y).squeeze().astype('int')

    return x, y


def load_CSVData_filter(args):
    dataset = pd.read_csv(args.path + '/' + args.DS + '/' + args.DS + '_x.csv', index_col=0)
    # print(dataset)
    # dataset = pd.read_csv(args.path+'/' + args.DS +'/pbmcCtrl_x.csv', index_col=0)
    label_dataset = pd.read_csv(args.path + '/' + args.DS + '/' + args.DS + '_label.csv', index_col=0)
    print(dataset.values.shape)
    # dataset.values = np.transpose(dataset.values)

    andata = ad.AnnData(X=np.zeros((dataset.values.shape[1],dataset.shape[0])))
    andata.X = np.transpose(dataset.values)
    
    aData=sc.pp.highly_variable_genes(
							andata, flavor='cell_ranger',
							n_top_genes=300, inplace=False, n_bins=30)

    # print(np.where(aData.var.variable_genes==True))
    # print(aData.var.variable_genes)
    # print(aData)
    print(andata.X.shape)
    X = np.array(andata.X[:,np.where(aData.highly_variable==True)[0]])
    
    # dataset=ad.AnnData(dataset)
    # dataset.X = np.transpose(dataset.X)
    # dataset.Y = label_dataset

    # sc.pp.filter_genes(andata,min_cells=30,inplace=True)

    # dataset= sc.pp.filter_genes(dataset,max_cells=1,inplace=True)
    # print(X.shape)
    x=X
    # x = andata.X
    print(x.shape)
    # x = dataset.values
    # x=preprocessing.scale(x)
    # x=np.log2(x+1)
    x=sc.pp.log1p(x)
    # x=np.log1p(x)
    # print(x.shape)
    # sc.tl.pca(x)
    # print(x.shape)
    x = torch.tensor(x, dtype=torch.float, requires_grad=True)
    # print(x.shape)
    # print(x.shape)
    # sc.tl.pca(x)
    # print(x.shape)
    # print(x)
    # label_dataset = pd.read_csv(args.path + '/' + args.DS + '/pbmcCtrl_y.csv', index_col=0)
    label_dataset = pd.read_csv(args.path + '/' + args.DS + '/' + args.DS + '_label.csv', index_col=0)
    label_attr = np.unique(label_dataset.values)
    # label_attr = np.unique(andata.obs['Y'])
    # print(andata.obs['Y'])
    y = label_dataset.values
    for i in range(len(label_attr)):
        idx = np.where(y == label_attr[i])
        y[idx] = i
    y = np.array(y).squeeze().astype('int')
    # print(x.shape)
    # print(y.shape)
    return x, y

import h5py
import numpy as np


def load_h5(args):
    # f = h5py.File('path/filename.h5','r') #打开h5文件
    f = h5py.File(args.path + '/' + args.DS + '.h5', 'r')
    x = torch.tensor(f['X'][:], dtype=torch.float, requires_grad=True)
    y = f['Y'][:]

    # print(x.shape)
    # print(y.shape)
    import pandas as pd
    input = pd.DataFrame(f['X'][:])
    label = pd.DataFrame(y)
    input.to_csv('./dataset/SingleCellData/SingleCellData/' + args.DS + '/' + args.DS + '_x.csv')
    label.to_csv('./dataset/SingleCellData/SingleCellData/' + args.DS + '/' + args.DS + '_label.csv')
    return x, y


def load_h5ad(args):
    filePath = args.path + '/' + args.DS + '.h5ad'
    annData = read_h5ad(filePath)
    # print(annData.to_df())

    # sc.pp.filter_cells(annData, min_genes=200)
    # sc.pp.filter_genes(annData, min_cells=3)

    X = pd.DataFrame(annData.X.todense())
    # print(X)
    # Y= pd.DataFrame(annData.obs)
    cell_name = annData.obs.index
    chr_name = annData.var.index
    # print(chr_name)
    X.index = cell_name
    X.columns = chr_name
    # chr_name = list(chr_name)
    # print(np.where(chr_name=='Hopx')[0])
    # print(np.where(chr_name == 'Pdpn')[0])
    # print(np.where(chr_name == 'Sftpc')[0])
    # print(np.where(chr_name == 'Pecam1')[0])
    # print(np.where(chr_name == 'Foxj1')[0])
    # X=X.T

    # with open('./FEAST_selection/'+args.DS+'.txt', 'r') as file:
    #     data = file.read()
        
    #     splitted_data = data.split()
    #     # print(np.array(splitted_data).shape)
    # position=[]
    # # for cell in chr_name:
    # #     print(type(cell))
    # # print(chr_name.tolist())
    # for col_name in splitted_data:
    #     col_name=col_name.strip('"')
    #     # print(col_name)
        
    #     # # col_name.replace("\"", "")
    #     # print(len(col_name))
    #     # print(col_name)
    #     # print(np.where(chr_name==col_name))
    #     # print(cell_name)
    #     # print(col_name in chr_name)
    #     # print(np.where(chr_name==col_name)[0])
    #     position.append(np.where(chr_name==col_name)[0])
    # position=np.concatenate(position, axis = 0)
    
    x = X.values
    # print(position.shape)

    # scipy.io.savemat('./middle/FEAST_'+args.DS+'_indicator.mat',{'x':x[:,position]})

    print(x.shape)
    scores = scipy.io.loadmat('./CellBRF_selection/score_'+args.DS+'_k_300.mat')['score'].squeeze()
    print(scores.shape)
    position = np.argsort(scores)[::-1]
    position = position[0:300]
    
    # print(position.shape)
    scipy.io.savemat('./middle/CellBRF_'+args.DS+'_indicator.mat',{'x':x[:,position]})

    x = sc.pp.log1p(x)
    x = torch.tensor(x, dtype=torch.float, requires_grad=True)
    # print(X)
    # print(annData.obs_keys())
    y = np.array(annData.obs.cell_type1.values)
    label_attr = np.unique(y)
    print(label_attr)
    for i in range(len(label_attr)):
        idx = np.where(y == label_attr[i])
        y[idx] = i
    y = np.array(y).squeeze().astype('int')
    # print('data shape:',x.shape)
    # print('label shape', y.shape)
    # X.to_csv('/home/R/R_data/Seurat/PBMC10/output/SCALE_ATAC.tsv',sep='\t')
    return x, y

def load_h5ad_filter(args):
    filePath = args.path + '/' + args.DS + '.h5ad'
    annData = read_h5ad(filePath)
    # sc.pp.highly_variable_genes(
	# 						annData, flavor='seurat_v3',
	# 						n_top_genes=300, inplace=True)
    # print(annData.to_df())
    print(annData.X.shape)
    # sc.pp.filter_genes(annData,min_cells=30,inplace=True)
    # print(set(np.unique(np.isnan(np.array(annData.X.todense())))))
    
    
    # sc.pp.filter_cells(annData, min_genes=200)
    # sc.pp.filter_genes(annData, min_cells=3)
    # print(annData)

    # sc.pp.log1p(annData)
    aData=sc.pp.highly_variable_genes(
							annData, flavor='cell_ranger',
							n_top_genes=300, inplace=True)
    # print(annData.var.highly_variable)
    # print(np.where(aData.highly_variable==True))
    X = pd.DataFrame(annData.X.todense()[:,np.where(annData.var.highly_variable==True)[0]])
    
    # print(X.shape)
    # Y= pd.DataFrame(annData.obs)
    # cell_name = annData.obs.index
    # chr_name = annData.var.index
    # X.index = cell_name
    # X.columns = chr_name
    # print(np.where(chr_name=='Hopx')[0])
    # print(np.where(chr_name == 'Pdpn')[0])
    # print(np.where(chr_name == 'Sftpc')[0])
    # print(np.where(chr_name == 'Pecam1')[0])
    # print(np.where(chr_name == 'Foxj1')[0])
    # X=X.T
    # x = annData.X.todense()
    x=X.values
    print(x.shape)
    # print(annData.X.shape)
    x = torch.tensor(x, dtype=torch.float, requires_grad=True)
    # print(X)
    # print(annData.obs_keys())
    y = np.array(annData.obs.cell_type1.values)
    label_attr = np.unique(y)
    print(label_attr)
    for i in range(len(label_attr)):
        idx = np.where(y == label_attr[i])
        y[idx] = i
    y = np.array(y).squeeze().astype('int')
    # print('data shape:',x.shape)
    # print('label shape', y.shape)
    # X.to_csv('/home/R/R_data/Seurat/PBMC10/output/SCALE_ATAC.tsv',sep='\t')
    return x, y


def process_ct_image(path):
    # path = "../ct_image/MontgomerySet/CXR_png/"
    files = os.listdir(path)
    label = []
    data = []
    for file in files:
        if file.endswith('_0.png'):
            label.append(0)
        else:
            label.append(1)
        tmp_data = cv2.resize(cv2.imread(path + file, 0), (512, 512))
        # cv2.imshow('image',tmp_data)
        # cv2.imshow('image', tmp_data)
        # cv2.waitKey(100)
        img_cv = tmp_data.flatten()
        # print(img_cv.shape)
        data.append(img_cv)

    # print(files, cmap='gray')

    data = np.array(data)
    label = np.array(label).astype('int')
    # print(Counter(label))
    return data, label


def process_chest_image(path):
    # path = "../ct_image/MontgomerySet/CXR_png/"
    size = 256
    train_normal_files = os.listdir(path + '/train/NORMAL/')
    train_abnormal_files = os.listdir(path + '/train/PNEUMONIA/')
    test_normal_files = os.listdir(path + '/test/NORMAL/')
    test_abnormal_files = os.listdir(path + '/test/PNEUMONIA/')
    label = []
    data = []
    for file in train_normal_files:
        # print(file)
        # if file.endswith('_0.png'):
        label.append(0)
        # else:
        #     label.append(1)
        tmp_data = cv2.resize(cv2.imread(path + '/train/NORMAL/' + file, 0), (size, size))
        # cv2.imshow('image',tmp_data)
        # # cv2.imshow('image', tmp_data)
        # cv2.waitKey(100)
        img_cv = tmp_data.flatten()
        # print(img_cv.shape)
        data.append(img_cv)
    for file in train_abnormal_files:
        # if file.endswith('_0.png'):
        label.append(1)
        # else:
        #     label.append(1)
        tmp_data = cv2.resize(cv2.imread(path + '/train/PNEUMONIA/' + file, 0), (size, size))
        # cv2.imshow('image',tmp_data)
        # cv2.imshow('image', tmp_data)
        # cv2.waitKey(100)
        img_cv = tmp_data.flatten()
        # print(img_cv.shape)
        data.append(img_cv)
    for file in test_normal_files:
        # print(file)
        # if file.endswith('_0.png'):
        label.append(0)
        # else:
        #     label.append(1)
        tmp_data = cv2.resize(cv2.imread(path + '/test/NORMAL/' + file, 0), (size, size))
        # cv2.imshow('image',tmp_data)
        # # cv2.imshow('image', tmp_data)
        # cv2.waitKey(100)
        img_cv = tmp_data.flatten()
        # print(img_cv.shape)
        data.append(img_cv)
    for file in test_abnormal_files:
        # if file.endswith('_0.png'):
        label.append(1)
        # else:
        #     label.append(1)
        tmp_data = cv2.resize(cv2.imread(path + '/test/PNEUMONIA/' + file, 0), (size, size))
        # cv2.imshow('image',tmp_data)
        # cv2.imshow('image', tmp_data)
        # cv2.waitKey(100)
        img_cv = tmp_data.flatten()
        # print(img_cv.shape)
        data.append(img_cv)
    # cv2.imshow('image', tmp_data)
    # cv2.waitKey(100)
    data = np.array(data)
    print(data.shape)
    label = np.array(label).astype('int')
    data = torch.tensor(data, dtype=torch.float, requires_grad=True)
    # print(Counter(label))
    return data, label


def show_img_chest(x, label, w, num):
    size = 256
    dots_x = list()
    dots_y = list()
    a = np.array(x[num])
    print(set(label))
    # print(a.shape)
    ### reshape a(1X1024) to 32X32
    a.shape = size, size
    ###transpose a
    # a = a.T
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.imshow(a, cmap='gray')
    # plt.axis('off')
    # plt.show()
    for i in range(len(w)):
        # print("in")
        n = math.floor((w[i] - 1) / size)
        y = w[i] % size
        dots_x.append(n)
        dots_y.append(y)
    print(dots_x)
    print(dots_y)
    plt.plot(dots_x, dots_y, 'y*', alpha=0.65)
    # plt.savefig('./fs_result/' + 'AEFS_' + str(num) + '.pdf',dpi=600)
    plt.savefig('./fs_result_chest/' + str(num) + '_class_' + str(label[num]) + '.png', bbox_inches='tight',
                pad_inches=0)
    plt.show()


def load_medmnist(path, DS):
    data = np.load(path + '/' + DS + '.npz')
    # data = np.load('./medmnist/octmnist.npz')
    # data = np.load('./medmnist/organamnist.npz')
    # data = np.load('./medmnist/organcmnist.npz')
    # for key in data.files:
    #     array = data[key]
    #     print("Array name: ", key)
    #     print("Content of the array:\n", array.shape)
    train_data = data['train_images']
    train_data = train_data.reshape(train_data.shape[0], 28 * 28)
    train_label = data['train_labels'].squeeze()
    val_data = data['val_images']
    val_data = val_data.reshape(val_data.shape[0], 28 * 28)
    val_label = data['val_labels'].squeeze()
    test_data = data['test_images']
    test_data = test_data.reshape(test_data.shape[0], 28 * 28)
    test_label = data['test_labels'].squeeze()

    total_data = np.concatenate((train_data, val_data, test_data), axis=0)
    total_labels = np.concatenate((train_label, val_label, test_label), axis=0)
    print(total_data.shape)
    print(total_labels.shape)
    # visualization(total_data, total_labels)
    # print(train_data.shape)
    # print(train_label.shape)
    total_data = torch.tensor(total_data, dtype=torch.float, requires_grad=True)
    # visualization
    # print(train_label)
    # val_data = data['val_images']
    # val_label = data['val_labels']
    # test_data = data['test_images']
    # test_label = data['test_labels']
    data.close()
    return total_data, total_labels


def visualization(data, label):
    class_0_num = 0
    class_1_num = 0
    for num in range(data.shape[0]):
        a = data[num][:]
        ### reshape a(1X1024) to 32X32
        a.shape = 28, 28
        ###transpose a
        # a = a.T

        plt.imshow(a, cmap='gray')
        # plt.axis('off')
        # plt.show()
        if label[num] == 0:
            plt.savefig('./origin_result/' + str(num) + '_class_' + str(label[num]) + '.png', bbox_inches='tight',
                        pad_inches=0)
            class_0_num += 1
        if label[num] == 1:
            plt.savefig('./origin_result/' + str(num) + '_class_' + str(label[num]) + '.png', bbox_inches='tight',
                        pad_inches=0)
            class_1_num += 1
        plt.show()
        # return 0


def show_img(x, label, w, num):
    dots_x = list()
    dots_y = list()
    a = np.array(x[num])
    print(set(label))
    # print(a.shape)
    ### reshape a(1X1024) to 32X32
    a.shape = 28, 28
    ###transpose a
    # a = a.T
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.imshow(a, cmap='gray')
    # plt.axis('off')
    # plt.show()
    for i in range(len(w)):
        # print("in")
        n = math.floor((w[i] - 1) / 28)
        y = w[i] % 28
        dots_x.append(n)
        dots_y.append(y)
    print(dots_x)
    print(dots_y)
    plt.plot(dots_x, dots_y, 'y*')
    # plt.savefig('./fs_result/' + 'AEFS_' + str(num) + '.pdf',dpi=600)
    plt.savefig('./fs_result/' + str(num) + '_class_' + str(label[num]) + '.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def show_img_mont(x, label, w, num):
    dots_x = list()
    dots_y = list()
    a = np.array(x[num])
    print(set(label))
    # print(a.shape)
    ### reshape a(1X1024) to 32X32
    a.shape = 512, 512
    ###transpose a
    # a = a.T
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.imshow(a, cmap='gray')
    # plt.axis('off')
    # plt.show()
    for i in range(len(w)):
        # print("in")
        n = math.floor((w[i] - 1) / 512)
        y = w[i] % 512
        dots_x.append(n)
        dots_y.append(y)
    print(dots_x)
    print(dots_y)
    plt.plot(dots_x, dots_y, 'y*')
    # plt.savefig('./fs_result/' + 'AEFS_' + str(num) + '.pdf',dpi=600)
    plt.savefig('./fs_result_mont/' + str(num) + '_class_' + str(label[num]) + '.svg', bbox_inches='tight',
                pad_inches=0)
    plt.show()
