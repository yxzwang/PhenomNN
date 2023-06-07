import argparse
import os
import os.path as osp



def parsering():

    parser = argparse.ArgumentParser()
    # Training parameter
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
    parser.add_argument('--print_freq', type=int, default=100, help='log frequency.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--debug', action='store_true',
                        default=False, help="Enable the detailed training output.")

    parser.add_argument('--data_path', default="./data/", help="The data path.")
    parser.add_argument("--early_stopping", type=int,
                        default=100,
                        help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
    parser.add_argument("--tensorboard", action='store_true', default=False,
                        help="Disable writing logs to tensorboard")

    parser.add_argument('--save_dir', type=str, default="model", help="The saving dir.")

    # Model parameter
    parser.add_argument('--type', default='GHCN',
                        help="model type.")
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of K of APPNP')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--withbn', action='store_true', default=False,
                        help='Enable Bath Norm GCN')
    parser.add_argument('--nhiddenlayer', type=int, default=1,
                        help='The number of hidden layers.')

    parser.add_argument("--nbaseblocklayer", type=int, default=1,
                        help="The number of layers in each baseblock")  # same as '--layer' of gcnii

    parser.add_argument('--gpu', type=int, default=0, help='device id')


    # for visual object classification
    parser.add_argument('--activate_dataset', type=str, default="none", help="The hgcn benchmark.")
    parser.add_argument('--K_neigs', type=int, default=10, help="the k of knn")
    parser.add_argument('--is_probH', action='store_true', default=False, help='Don"t using probability distance map')
    parser.add_argument('--gamma', type=float, default=0.5, help="Guass kernel flat coefficiency for constructing edge-dependent vertex weights")
    parser.add_argument('--on_dataset', type=str, default="NTU2012",
                        help="select the dataset you use, ModelNet40 or NTU2012.", choices=['ModelNet40','NTU2012'])
    parser.add_argument('--mvcnn_feature_structure', action='store_true', default=False,
                        help='use_mvcnn_feature_for_structure')
    parser.add_argument('--gvcnn_feature_structure', action='store_true', default=False,
                        help='use_gvcnn_feature_for_structure')
    parser.add_argument('--use_gvcnn_feature', action='store_true', default=False,
                        help='use_gvcnn_feature_add to features X')
    parser.add_argument('--use_mvcnn_feature', action='store_true', default=False,
                        help='use_gvcnn_feature_add  to features X')



    # graument for citation dataset
    parser.add_argument('--split', type=int, default=1, help='train-test split used for the dataset')


    parser.add_argument('--save_file', default='results.csv', help='save file name')

    parser.add_argument('--degree', type=int, default=16,
                        help='degree of the approximation.')
    # parser.add_argument('--alpha', type=float, default=0.05,
    #                     help='alpha.')
    parser.add_argument('--sigma', type=float, default=1,
                        help='sigma for edge degree matirx.')

    # # argument for gcnii
    parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
    parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.') 
    # argument for phenomnn_s

    parser.add_argument('--alp', type=float, default=1, help='alp')
    parser.add_argument('--lam', type=float, default=20, help='lam.')
    parser.add_argument('--attention', action='store_true', default=False, help='whether to use attention in phenomnn_s')
    parser.add_argument('--tau', type=float, default=0.01, help='attention parameters.')
    parser.add_argument('--attn_dropout', type=float, default=0, help='attention parameters.')
    parser.add_argument('--T', type=float, default=0, help='attention parameters.')
    parser.add_argument('--p', type=float, default=1, help='attention parameters.')
    parser.add_argument('--prop_step', type=int, default=16, help='propagating steps')
    # argument for phenomnn
    parser.add_argument('--feature_noise', type=float, default=1, help="featurenoise for synthetic datasets")
    
    
    parser.add_argument('--LP', action='store_true', default=False, help='Label propagation')
    parser.add_argument('--lam4', type=float, default=0, help='lam4.')
    parser.add_argument('--lam0', type=float, default=10, help='lam0.')
    parser.add_argument('--lam1', type=float, default=10, help='lam1.')
    parser.add_argument('--normalize_type', type=str, default="full", help='normalize type for phenomnn')
    parser.add_argument('--H', action='store_true', default=False, help='whether to use compatibility matrix in phenomnn')
    parser.add_argument('--HisI', action='store_true', default=False, help='if using H and H is I ')
    parser.add_argument('--notresidual', action='store_true', default=False, help='whether to use residual in H')
    parser.add_argument('--twoHgamma', action='store_true', default=False, help='whether to use two H for gamma matrix')

    args = parser.parse_args()

    return args

def check_dir(folder, mk_dir=True):
    if not osp.exists(folder):
        if mk_dir:
            print(f'making direction {folder}!')
            os.mkdir(folder)
        else:
            raise Exception(f'Not exist direction {folder}')


def check_dirs(args):
    check_dir(args.data_path, mk_dir=False)

    check_dir(args.save_dir)

def train_args():
    args = parsering()
    args.modelnet40_ft = os.path.join(args.data_path, 'ModelNet40_mvcnn_gvcnn.mat')
    args.ntu2012_ft = os.path.join(args.data_path, 'NTU2012_mvcnn_gvcnn.mat')
    check_dirs(args)
    return args
