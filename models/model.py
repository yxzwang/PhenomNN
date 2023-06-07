from torch import nn

from models.phenomnn import phenomnn
from models.phenomnn_s import phenomnn_s
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class GCNModel(nn.Module):
    """
       The model architecture likes:
       All options are configurable.
    """

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 baseblock="phenomnn",
                 inputlayer=None,
                 outputlayer=None,
                 nbaselayer=0,
                 args=None,
                 ):
        """
        Initial function.
        :param nfeat: the input feature dimension.
        :param nhid:  the hidden feature dimension.
        :param nclass: the output feature dimension.
        :param nhidlayer: the number of hidden blocks.
        :param dropout:  the dropout ratio.
        :param baseblock: the baseblock type, can be "phenomnn", "phenomnn_s".
        :param nbaselayer: the number of layers in one hidden block.
        """
        super(GCNModel, self).__init__()
        self.dropout = dropout
        self.baseblock = baseblock.lower()
        self.nbaselayer = nbaselayer
        self.args = args


        if self.baseblock == "phenomnn":
            self.BASEBLOCK = phenomnn
        elif self.baseblock == "phenomnn_s":
            self.BASEBLOCK = phenomnn_s
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))

        self.midlayer = nn.ModuleList()

        for i in range(nhidlayer):
            
            if baseblock.lower() in ['phenomnn',"phenomnn_s"]:
                gcb = self.BASEBLOCK(nfeat=nfeat,
                                     nlayers=nbaselayer,
                                     nhidden=nhid,
                                     nclass=nclass,
                                     dropout=dropout,
                                     lamda=args.lamda,
                                     alpha=args.alpha,
                                     variant=args.variant,
                                     args=args,
                                     )

            else:  # gcn
                NotImplementedError("Current baseblock %s is not supported." % (baseblock))
            self.midlayer.append(gcb)
        if baseblock.lower() in ['phenomnn',"phenomnn_s"]:
            # self.ingc = nn.Linear(nfeat, nhid)
            # self.outgc = nn.Linear(nhid, nclass)
            # self.fcs = nn.ModuleList([self.ingc, self.outgc])
            self.params1 = self.midlayer[0].params1
            self.params2 = self.midlayer[0].params2
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, fea, adj, G=None):

        if self.baseblock=="phenomnn_s":
            out = self.midlayer[0](input=fea, adj=adj, D=G)
            return out
        elif self.baseblock=="phenomnn":
            out = self.midlayer[0](input=fea, adj=adj, D=G)
            return out
