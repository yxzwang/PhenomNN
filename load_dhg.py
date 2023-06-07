
import torch
from utils.convert_datasets_to_pygDataset import dataset_Hypergraph
import numpy as np
import scipy.sparse as sp
existing_dataset = ['20newsW100', 'ModelNet40', 'zoo',
                        'NTU2012', 'Mushroom',
                        'coauthor_cora', 'coauthor_dblp',
                        'yelp', 'amazon-reviews', 'walmart-trips', 'house-committees',
                        'walmart-trips-100', 'house-committees-100',
                        'cora', 'citeseer', 'pubmed']
    
synthetic_list = ['amazon-reviews', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']
def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    try:
        num_nodes = data.n_x[0]
    except:
        num_nodes=data.n_x
    try:
        num_hyperedges = data.num_hyperedges[0]
    except:
        num_hyperedges = data.num_hyperedges
    
    if not ((num_nodes+num_hyperedges-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data
def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx
def load_dhg(args):
    datasetname=args.activate_dataset.split("/")[-1]
    feature_noise=args.feature_noise
    if datasetname in existing_dataset:
        dname = datasetname
        f_noise = feature_noise
        if (f_noise is not None) and dname in synthetic_list:
            p2raw = 'data/raw_data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname, 
                    feature_noise=f_noise,
                    p2raw = p2raw)
        else:
            if dname in ['cora', 'citeseer','pubmed']:
                p2raw = 'data/raw_data/AllSet_all_raw_data/cocitation/'
            elif dname in ['coauthor_cora', 'coauthor_dblp']:
                p2raw = 'data/raw_data/AllSet_all_raw_data/coauthorship/'
            elif dname in ['yelp']:
                p2raw = 'data/raw_data/AllSet_all_raw_data/yelp/'
            else:
                p2raw = 'data/raw_data/AllSet_all_raw_data/'
            dataset = dataset_Hypergraph(name=dname,root = 'data/pyg_data/hypergraph_dataset_updated/',
                                         p2raw = p2raw)
    data=dataset.data
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    if datasetname in ['yelp', 'walmart-trips', 'house-committees', 'walmart-trips-100', 'house-committees-100']:
        #         Shift the y label to start with 0
        num_classes = len(data.y.unique())
        data.y = data.y - data.y.min()
    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max()-data.n_x[0]+1])
    
    try:
        num_nodes = data.n_x[0]
    except:
        num_nodes=data.n_x
    try:
        num_hyperedges = data.num_hyperedges[0]
    except:
        num_hyperedges = data.num_hyperedges
    data=ExtractV2E(data)
    edge_index=data.edge_index## in edge_index, hyperedge is set as node_index in [nodenum,nodenum+hyperedgenum-1]
    edge_index[1]=edge_index[1]-data.n_x ##get the hyperedgeidx to [0,hyperedgenum-1]

    Hrow,Hcol=edge_index[0],edge_index[1]
    Hval=np.ones_like(Hrow,dtype=np.int8)

    H=sp.coo_matrix((Hval,(Hrow,Hcol)),(num_nodes,int(num_hyperedges)))
    totalsplit=[]
    for i in range(10):
        split_idx = rand_train_test_idx(
                data.y, train_prop=0.5, valid_prop=0.25)
        totalsplit.append(split_idx)
    features=data.x
    labels=data.y.view(-1)
    idx_train=[split_idx["train"] for split_idx in totalsplit ]
    idx_val=[split_idx["valid"] for split_idx in totalsplit ]
    idx_test=[split_idx["test"] for split_idx in totalsplit ]

    return H,features,labels,idx_train,idx_test,idx_val
if __name__=="__main__":
    pass