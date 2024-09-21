import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='JUDS Arguments.')
    parser.add_argument('--datasetname', dest='DS', default='usoskin', help='Dataset')
    parser.add_argument('--path', dest='path', default='./dataset/SingleCellData/SingleCellData',help='Data Path')
    # parser.add_argument('--path', dest='path', default='./dataset', help='Data Path')

    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
            help='Learning rate.')
    parser.add_argument('--alpha', dest='alpha', type=float, default=1000,
            help='Coefficient of L2,1-norm')
    parser.add_argument('--nu', dest='beta', type=float, default=0.1,
                        help='Coefficient of Clustering Loss')
    parser.add_argument('--k', dest='k', type=int, default=100)
    parser.add_argument('--repNum', dest='repNum', type=int,
                        help='Repeat number.', default=5)
    parser.add_argument('--epochs', dest='epochs', type=int, help='Training Epochs',default=100)
    parser.add_argument('--lr_milestones', '-lr_mile', type=list, default=[6000], help='Learning Rate Milestones')
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--eval', dest='eval', type=bool, default=True)

    return parser.parse_args()






