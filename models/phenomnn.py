
import math
import torch 
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgl.mock_sparse import create_from_coo, diag, identity

       
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False, incidence_v=100, incidence_e=50,
                 init_dist=None, args=None):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        self.args = args
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features
        self.lam4=args.lam4
        if self.lam4!=0:
            print("lam4 is not zero!!!!!!! wrong")
            exit(0)
        self.lam0=args.lam0
        self.lam1=args.lam1
        self.alpha=args.alp if args.alp !=0 else 1/(1+args.lam4+args.lam0+args.lam1)
        self.num_steps=args.prop_step
        self.out_features = out_features
        self.residual = residual
        self.notresidual=args.notresidual
        self.twoHgamma=args.twoHgamma
        # self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.adj = None
        self.normalize_type=args.normalize_type#in ["edge","none","full","node"]
        if args.H:
            # H = torch.rand(in_features, in_features)
            # bound = 4/in_features # normal
            # nn.init.normal_(H, 0, bound)

            # H = torch.rand(in_features, in_features)
            # bound = 1/in_features # normal
            # nn.init.normal_(H, 0, bound)
            # H = H + torch.eye(in_features)
            # self.H=nn.Parameter(H)
            H = {}
            for t in ["beta","gamma1","gamma2"]:

                if args.notresidual:
                    H[t] = torch.rand(in_features, in_features)
                    bound = 4/in_features # normal
                    nn.init.normal_(H[t], 0, bound)
                    H[t] = nn.Parameter(H[t])
                else:


                    H[t] = torch.rand(in_features, in_features)
                    bound =1/in_features # normal
                    nn.init.normal_(H[t], 0, bound)
                    H[t] = H[t] + th.eye(in_features)
                
                
                    H[t] = nn.Parameter(H[t])
                
            self.H = nn.ParameterDict(H)

        else:
            self.H=None
       

        self.init_attn=None
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.out_features)
        # self.weight.data.uniform_(-stdv, stdv)
        pass
    
    def forward(self, X, A, D):
        A_beta,A_gamma=A
        D_beta,D_gamma,I=D
        ##B is the incidence matrix with N x E
        
        ##after linear and dropout
         # Compute Y = Y0 = f(X; W) using a two-layer MLP.
        H=self.H 
        # Y = Y0 = self.act_fn(self.mlp(X))
        Y = Y0 = X
       

        ####

        # Compute diagonal matrix Q_tild.
        if H is not None:
            # D_stinv=diag((B ).sum(0))
            # # import ipdb
            # # ipdb.set_trace()
            # B_=B @ (D_stinv)**(-1)
            # # Q_tild=self.lam4*L_alpha + self.lam0*D_beta+self.lam1*B_@D_stinv@B_.T + I
            # Q_tild=self.lam1*B_@D_stinv@B_.T + I
            # D_st=diag((B ).sum(1))
            ###############################diagD
            Q_tild= self.lam0*D_beta+self.lam1*D_gamma + I
            diagD=True

            L_gamma=D_gamma.as_sparse()-A_gamma
            # D_st=diag(B.sum(1))
            H_1=H["beta"]
            H_2=H["gamma1"]
            H_3=H["gamma2"]
        else:

            Q_tild= self.lam0*D_beta+self.lam1*D_gamma + I

        # Iteratively compute new Y by equation (6) in the paper.
        for k in range(self.num_steps):
            if H is not None:
                
                # Y_hat = self.lam0 * A_beta @ Y + Y0 + self.lam1 * ( B @B_.T @ Y @ H.T+ B_ @ B.T @ Y @ H- D_st @ Y @ H @ H.T )
                ##diagD
                if diagD:
                    if self.twoHgamma:
                        Y_hat = self.lam0 * (A_beta @ Y @ (H_1+H_1.T)- D_beta @ Y @ H_1 @ H_1.T ) + Y0 + self.lam1/2 * ( L_gamma @ Y + A_gamma @ Y @ (H_2+H_2.T)- D_gamma @ Y @ H_2 @ H_2.T + A_gamma @ Y @ (H_3+H_3.T)- A_gamma @ Y @ H_3 @ H_3.T)
                    else:
                        if self.args.HisI:
                            Y_hat = self.lam0 * (2*A_beta @ Y- D_beta @ Y ) + Y0 + self.lam1 *  A_gamma @ Y 
                        else:

                            Y_hat = self.lam0 * (A_beta @ Y @ (H_1+H_1.T)- D_beta @ Y @ H_1 @ H_1.T ) + Y0 + self.lam1 * ( L_gamma @ Y + A_gamma @ Y @ (H_2+H_2.T)- D_gamma @ Y @ H_2 @ H_2.T )
            else:

                Y_hat = self.lam0 * A_beta @ Y + Y0 + self.lam1 *  A_gamma @ Y 
            Y = (1 - self.alpha) * Y + self.alpha * (Q_tild ** -1) @ Y_hat


        # we have linear out of this module
        return Y
        


class phenomnn(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, incidence_v=100, incidence_e=50,
                 init_dist=None, args=None):
        super(phenomnn, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(1):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant,args=args))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.in_features = nfeat
        self.out_features = nclass
        self.hiddendim = nhidden
        self.nhiddenlayer = nlayers

        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, input, adj, D):
        _layers = []
        x = F.dropout(input, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        # layer_inner = input
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, A=adj, D=D))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        self.adj = con.adj  # 保存看看学的结果
        return layer_inner  # F.log_softmax(layer_inner, dim=1)

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s lamda=%s alpha=%s (%d - [%d:%d] > %d)" % (self.__class__.__name__,self.lamda,
                                                    self.alpha,
                                                    self.in_features,
                                                    self.hiddendim,
                                                    self.nhiddenlayer,
                                                    self.out_features)
