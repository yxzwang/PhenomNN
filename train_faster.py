import os
import time
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pprint as pp
from models.model import GCNModel
from datasets import load_feature_construct_H, generate_H_from_dist
# from datasets import source_select
from parsing import train_args
from torch.nn.utils import clip_grad_norm_
from utils import hypergraph_utils as hgut
import scipy.sparse as sp
from datasets import data
from dgl.mock_sparse import create_from_coo, diag, identity
# CUDA_LAUNCH_BLOCKING=1
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

args = train_args()
device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')
# initialize visual object classification data
data_dir = args.modelnet40_ft if args.on_dataset == 'ModelNet40' \
    else args.ntu2012_ft
adj = None
## we only need fts for input shape and lbls for num_classes here . But it's better to generate input dataset here rather than in get_data
if args.activate_dataset.startswith('coauthorship') or args.activate_dataset.startswith('cocitation'):
    dataset, idx_train1, idx_test1 = data.load(args)  # reloading due to different splits.
    idx_val1 = idx_test1
    hypergraph, fts1, lbls1 = dataset['hypergraph'], dataset['features'], dataset['labels']
    lbls1 = np.argmax(lbls1, axis=1)

    H = np.zeros((dataset['n'], len(hypergraph)))
    for i, (a, p) in enumerate(hypergraph.items()):
        H[list(p), i] = 1

    fts=fts1 = torch.Tensor(fts1).to(device)  # features -> fts
    lbls=lbls1 = torch.Tensor(lbls1).squeeze().long().to(device)
    idx_train1 = torch.Tensor(idx_train1).long().to(device)
    idx_test1 = torch.Tensor(idx_test1).long().to(device)
    idx_val1 = torch.Tensor(idx_val1).long().to(device)

    G = hgut._generate_G_from_H_sparse(H, args=args)
    G = sparse_mx_to_torch_sparse_tensor(G)


    from dgl.mock_sparse import create_from_coo
    H = sparse_mx_to_torch_sparse_tensor(sp.lil_matrix(H))

    src,dst=H.coalesce().indices().to(device)
    
    B=create_from_coo(src,dst,torch.ones_like(src).to(torch.float32),shape=H.shape)
    G=None



elif args.activate_dataset.startswith("dhg"):##dhg/20newsW100
    from load_dhg import load_dhg
    H,fts, lbls, idx_train, idx_val, idx_test=load_dhg(args)
    fts1 = fts.to(device)
    lbls1 =lbls= lbls.long().to(device)

    print("============================generating G =================")
    
    from dgl.mock_sparse import create_from_coo

    src,dst=torch.LongTensor(H.row).to(device),torch.LongTensor(H.col).to(device)
    
    H=create_from_coo(src,dst,torch.ones_like(src).to(torch.float32),shape=H.shape)
    G=None
    pass
else:
    fts, lbls, idx_train, idx_test, mvcnn_dist, gvcnn_dist = \
        load_feature_construct_H(data_dir,
                                 gamma=args.gamma,
                                 K_neigs=args.K_neigs,
                                 is_probH=args.is_probH,
                                 use_mvcnn_feature=args.use_mvcnn_feature,
                                 use_gvcnn_feature=args.use_gvcnn_feature,
                                 use_mvcnn_feature_for_structure=args.mvcnn_feature_structure,
                                 use_gvcnn_feature_for_structure=args.gvcnn_feature_structure)
    idx_val = idx_test
    fts1 = fts=torch.Tensor(fts).to(device)
    lbls1 = lbls=torch.Tensor(lbls).long().to(device).view(-1)
    idx_train1 = idx_train=torch.Tensor(idx_train).long().to(device)
    idx_test1 = idx_test=torch.Tensor(idx_test).long().to(device)
    idx_val1 = idx_val=torch.Tensor(idx_val).long().to(device)
    print(f"data_dir==============={data_dir}")
    if os.path.exists(f"{data_dir}-Hmatrix"):
        H=torch.load(f"{data_dir}-Hmatrix")
        G=torch.load(f"{data_dir}-Gmatrix")

    else:
        H = generate_H_from_dist(mvcnn_dist=mvcnn_dist,
                                    gvcnn_dist=gvcnn_dist,
                                    split_diff_scale=False,
                                    gamma=args.gamma,
                                    K_neigs=args.K_neigs,
                                    is_probH=args.is_probH,
                                    use_mvcnn_feature_for_structure=args.mvcnn_feature_structure,
                                    use_gvcnn_feature_for_structure=args.gvcnn_feature_structure)

        G = hgut.generate_G_from_H(H, args=args)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2 :
        
        H = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(H))
        G = sparse_mx_to_torch_sparse_tensor(G)
       
        torch.save(H,f"{data_dir}-Hmatrix")
        torch.save(G,f"{data_dir}-Gmatrix")

n_class = int(lbls.max()) + 1



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True 
    torch.cuda.current_device()
    torch.cuda._initialized = True


def get_data(args, device):

    ### read data only once
    
    if args.activate_dataset.startswith("dhg"):
        idx=args.idxfordhg
        idx_train2 = idx_train[idx].long().to(device)
        idx_test2 = idx_test[idx].long().to(device)
        idx_val2 = idx_val[idx].long().to(device)
        return G, H, fts1, lbls1, idx_train2, idx_test2, idx_val2

    return G, H, fts1, lbls1, idx_train1, idx_test1, idx_val1
@torch.no_grad()
def Evaluation(output, labels):
    
    from sklearn import metrics
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    '''
    binary_pred = preds
    binary_pred[binary_pred > 0.0] = 1
    binary_pred[binary_pred <= 0.0] = 0
    '''
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    # print('total number of correct is: {}'.format(num_correct))
    #print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    #'''
    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")

def B2A(B,w=None,normalize_type="full"):
        #for alpha forward , we only have the diagonal value for adj and deg and laplacian,and they are thesame
        if w is None:
            if normalize_type=="edge":
                DE=diag((B ).sum(0)** (-1))
                L_alpha=None
                A_beta=B @ (DE) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
            
            elif normalize_type=="full":

                DE=diag(torch.pow((B ).sum(0),-1))
                L_alpha=None
                A_beta=B @ (DE) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
                ##renormalization
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="none":
                L_alpha=None
                A_beta=B  @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="node":
                L_alpha=None
                A_beta=B  @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
        else:
            if normalize_type=="edge":
                DE=diag((B ).sum(0)** (-1))
                DE=DE@w
                L_alpha=None
                A_beta=B @ (DE ) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
            
            elif normalize_type=="full":
                
                DE=diag(torch.pow((B ).sum(0),-1))
                DE=DE@w
                L_alpha=None
                A_beta=B @ (DE) @ B.T
                I = identity(A_beta.shape, device=B.device)
                A_beta+=I
                #
                D_beta=diag(A_beta.sum(1))
            ####
                ##renormalization
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="none":

                L_alpha=None
                A_beta=B @ w @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
            elif normalize_type=="node":
                L_alpha=None
                A_beta=B @w @ B.T
                I = identity(A_beta.shape, device=B.device)
                #
                A_beta+=I
                D_beta=diag(A_beta.sum(1))
                A_beta=D_beta**(-1/2) @ A_beta @ D_beta**(-1/2) 
                D_beta=diag(A_beta.sum(1))
        return L_alpha,A_beta,D_beta,I

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, args=None):
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(
            os.path.join(args.save_dir, f'{time_start}'))
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    if args.activate_dataset.startswith("ml"):
        best_acc, best_epoch, best_val_loss = 999, 0, -1.0
    else:
        best_acc, best_epoch, best_val_loss = -1, 0, -1.0
    G, H, fts1, lbls1, idx_train1, idx_test1, idx_val1 = get_data(args=args,device=device)
    from tqdm import tqdm
    try:

        B=H.to("cpu")
        src,dst=B.coalesce().indices().to(device)
    
        B=create_from_coo(src,dst,torch.ones_like(src).to(torch.float32),shape=B.shape)
        H=B
        print("Converting")
    except:
        B=H
        pass
    if args.type =="phenomnn_s":
            _,A_beta,D_beta,I=B2A(B,normalize_type="node")
            L_alpha,A_gamma,D_gamma,_=B2A(B,normalize_type="full")
            A= args.lam0 * A_beta +args.lam1 *  A_gamma
            D=args.lam0 * D_beta +args.lam1 *  D_gamma
            H=A
            G=[D,I]
    elif args.type =="phenomnn":
            _,A_beta,D_beta,I=B2A(B,normalize_type="node")
            L_alpha,A_gamma,D_gamma,_=B2A(B,normalize_type="full")
            H=[A_beta,A_gamma]
            G=[D_beta,D_gamma,I]
    for epoch in tqdm(range(num_epochs)):

        if epoch % print_freq == 0:
            # print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0

            idx = idx_train1 if phase == 'train' else idx_val1  # idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):

                outputs = model(fts1, G=G, adj=H)
                if args.activate_dataset.startswith("ml"):
                    loss=criterion(outputs[idx].squeeze(), lbls1[idx])
                    preds=outputs.squeeze()
                elif args.activate_dataset.startswith("amazon"):
                    loss=criterion(outputs[idx], lbls1[idx])
                    preds=outputs
                else:

                    loss = criterion(outputs[idx], lbls1[idx])

                    _, preds = torch.max(outputs, 1)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 40)
                    optimizer.step()
                    scheduler.step(loss)

            # statistics
            if args.activate_dataset.startswith("ml"):
                running_loss += loss.item() 
                running_corrects += torch.sqrt(criterion(outputs[idx].squeeze(), lbls1[idx]))
                avg_loss = running_loss 
                epoch_acc = running_corrects
            elif args.activate_dataset.startswith("amazon"):
                running_loss += loss.item() 

                micro_f1,macro_f1= Evaluation(preds[idx],lbls1[idx])
                running_corrects=macro_f1
                avg_loss = running_loss 
                epoch_acc = running_corrects
            else:
                running_loss += loss.item() * fts1.size(0)
                running_corrects += torch.sum(preds[idx] == lbls.data[idx])
                avg_loss = running_loss / len(idx)
                epoch_acc = running_corrects.double() / len(idx)




            if epoch % print_freq == 0:
                print(
                    f'{phase} avgLoss: {avg_loss:.4f} || per_loss: {loss:.4f} || type:{args.type}  ||'
                    f' {f"dataset:{args.activate_dataset}" if args.activate_dataset!="none" else f"dataset:{args.on_dataset}"}' \
                    f'|| lr: {optimizer.param_groups[0]["lr"]} || Acc: {epoch_acc:.4f} || dropout: {args.dropout} ')
            # deep copy the model
            if phase == 'val':
                if args.activate_dataset.startswith("ml"):
                    if epoch_acc < best_acc:
                        early_stop = 0
                        best_acc = epoch_acc
                        best_epoch = epoch
                        best_val_loss = loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                    else:
                        early_stop += 1
                else:
                    if epoch_acc > best_acc:
                        early_stop = 0
                        best_acc = epoch_acc
                        best_epoch = epoch
                        best_val_loss = loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                    else:
                        early_stop += 1

            if args.tensorboard:
                if phase == 'train':
                    writer.add_scalar('loss/train', loss, epoch)
                    writer.add_scalar('acc/train', epoch_acc, epoch)
                else:
                    writer.add_scalar('loss/val', loss, epoch)
                    writer.add_scalar('acc/val', epoch_acc, epoch)
                    writer.add_scalar('best_acc/val', best_acc, epoch)


        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f} at epoch: {best_epoch}, n_layers: {args.nbaseblocklayer} ')
            print('-' * 20)

        if early_stop > args.early_stopping:
            print(f'early stop at epoch {epoch}, n_layers: {args.nbaseblocklayer}' )
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()
    # test
    outputs = model(fts1, adj=H, G=G)
    
    _, preds = torch.max(outputs, 1)
    test_acc = torch.sum(preds[idx_test1] == lbls.data[idx_test1]).double() / len(idx_test1)
    # print(args)
    print(f"test_acc={test_acc}\n"
          f"best_val_acc={best_acc}\n"
          f"best_val_epoch={best_epoch}\n"
          f"best_val_loss={best_val_loss}")

    if args.tensorboard:
        writer.add_histogram('best_acc', test_acc)
    return best_epoch, float(test_acc),time_elapsed




def param_count(model):
    print(model)
    for n, p in model.named_parameters():
        print(n, p.shape)
    param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'Number of parameters = {param_count:,}')

def _main():
    print(args)
    C=args


    
    model = GCNModel(nfeat=fts.shape[1],
                    nhid=args.hidden,
                    nclass=n_class,
                    nhidlayer=args.nhiddenlayer,
                    dropout=args.dropout,
                    baseblock=args.type,
                    nbaselayer=args.nbaseblocklayer,
                    args=args)
    print("start trainmodel-------------------------------")
    param_count(model)
    model = model.to(device)
    

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=100,
                                                        verbose=False,
                                                        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                        eps=1e-08)

    criterion = torch.nn.CrossEntropyLoss()
    
    best_epoch, test_acc,time_elapsed = train_model(model, criterion, optimizer, scheduler, args.epochs, print_freq=args.print_freq,
                                        args=args)

    return test_acc, best_epoch,time_elapsed


if __name__ == '__main__':
    elapsed_times=[]
    if args.activate_dataset.startswith('coauthor') or args.activate_dataset.startswith('cocitation'):
        setup_seed(args.seed)
        if args.debug:
            splits = [args.split]
        else:
            splits = [args.split + i for i in range(10)]
        results = []
        for split in splits:
            print(f"split: {split}/{splits}")
            args.split = split
            test_acc, best_epoch,time_elapsed = _main()
            print("complete main")
            results.append(test_acc)
            
            elapsed_times.append(time_elapsed)
            print('Acc array: ', results)
    else: # visual object &&large datasets
        if args.debug:
            seed_nums = [args.seed]  # 1000
        else:
            seed_nums = [args.seed + i for i in range(10)]  # 1000
        results = []
        if args.activate_dataset.startswith("dhg"):
            setup_seed(args.seed)
        
        for idx,seed_num in enumerate(seed_nums):
            args.idxfordhg=idx
            print(f"seed:{seed_num}/{seed_nums}")
            if args.activate_dataset.startswith("dhg"):
                pass
            else:
                setup_seed(seed_num)
            test_acc, best_epoch, time_elapsed = _main()
            results.append(test_acc)
            
            elapsed_times.append(time_elapsed)
            print('Acc array: ', results)


    results = np.array(results)

    elapsed_times = np.array(elapsed_times)
    print(f"\nAvg_test_acc={results.mean():.5f} \n"
          f"std={results.std():.5f}\n"
          f"elapsed_times:{elapsed_times.mean():.4f}+-{elapsed_times.std():.4f}s.")

