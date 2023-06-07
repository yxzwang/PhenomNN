import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix



def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat




def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, args=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H_sparse(H, args=args)  # D_v^1/2 H W D_e^-1 H.T D_v^-1/2
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, args))
        return G


def _generate_G_from_H_sparse(H, add_self_loop=False, sigma=None, args=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    # add_self_loop = False
    if args is not None:
        sigma = args.sigma
    else:
        sigma = -1

    H = coo_matrix(H)
    n_edge = H.shape[1]  # 4024
    # the weight of the hyperedge
    W = np.ones(n_edge)

    # the degree of the hyperedge
    DE = np.sum(H, axis=0)  # [4024]


    DE = DE.tolist()[0]
    invDE = np.power(DE, sigma)
    invDE[np.isinf(invDE)] = 0
    invDE = coo_matrix((invDE, (range(n_edge), range(n_edge))), shape=(n_edge, n_edge))
    K = H * invDE * H.T
    # if args.add_self_loop:
    print('renormalization!!')
    K += sp.eye(H.shape[0])

    DV = np.sum(K, 0).tolist()[0]
    invDV = np.power(DV, -0.5)
    invDV[np.isinf(invDV)] = 0
    DV2 = coo_matrix((invDV, (range(H.shape[0]), range(H.shape[0]))), shape=(H.shape[0], H.shape[0]))


    G = DV2 * K * DV2

    return G




def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, gamma=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param gamma: prob
    :return: N_object X N_hyperedge
    """

    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        # dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(-dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (gamma * avg_dis ** 2))
            else:
                H[node_idx, center_idx] = 1.0


    return H


def cal_distance_map(X):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param gamma: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    dis_mat = -Eu_dis(X)

    return dis_mat


def construct_H_with_KNN(dis_mat, K_neigs, split_diff_scale=False, is_probH=True, gamma=1):
    if type(K_neigs) == int:
        K_neigs = [K_neigs]
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, gamma)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)
