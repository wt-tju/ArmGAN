from __future__ import division
from __future__ import print_function

import argparse
import time
import warnings
warnings.filterwarnings('ignore')
#from pyecharts.charts import Line

import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.autograd import Variable

from model import GCNModelVAE, Discriminator,Discriminator_FC,Generator_FC,Mine
import itertools
from optimizer import loss_function
from utils import load_data,load_data2, mask_test_edges, preprocess_graph, get_roc_score,preprocess_features

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=512, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=256, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--lambda1', type=float, default=1)
parser.add_argument('--lambda2', type=float, default=1, help='reconstruction loss')
parser.add_argument('--reg1', type=float, default=0.01, help='reg hyperparameter')
parser.add_argument('--reg2', type=float, default=0.01, help='reg hyperparameter')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='citeseer', help='type of dataset.')
# ['cornell', 'texas', 'washington', 'wiscosin', 'uai2010'] 'cora','citeseer','pubmed'
args = parser.parse_args()
print(args)

###moving average
def update_target(ma_net, net, update_rate=1e-1):
    # update moving average network parameters using network
    for ma_net_param, net_param in zip(ma_net.parameters(), net.parameters()):
        ma_net_param.data.copy_((1.0 - update_rate) \
                                * ma_net_param.data + update_rate * net_param.data)

def get_NMI(n_clusters,emb,true_label):
    from sklearn.cluster import KMeans
    from sklearn import metrics
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(emb)
    predict_labels = kmeans.predict(emb)
    nmi = metrics.normalized_mutual_info_score(true_label, predict_labels)
    ac=clusteringAcc(true_label, predict_labels)
    return nmi,ac
def clusteringAcc(true_label, pred_label):
    from sklearn import metrics
    from munkres import Munkres, print_matrix
    # best mapping between true_label and predict label
    l1 = list(set(true_label))
    numclass1 = len(l1)
    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0
    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)
    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(true_label, new_predict)
    '''
    f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
    precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
    recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
    f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
    precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
    recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
    return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
    '''
    return acc

###estimate the mutual information
def learn_mine(model, M, M_opt, feature,adj_norm,X_hat,ma_rate=0.001,):
    '''
    Mine is learning for MI of (input, output) of Generator.
    '''
    recovered, mu = model(feature, adj_norm)
    et = torch.mean(torch.exp(M(X_hat, mu)))
    if M.ma_et is None:
        M.ma_et = et.detach().item()
    M.ma_et += ma_rate * (et.detach().item() - M.ma_et)
    mutual_information = torch.mean(M(feature, mu)) \
                         - torch.log(et) * et.detach() / M.ma_et
    loss = - mutual_information

    M_opt.zero_grad()
    loss.backward(retain_graph=True)
    M_opt.step()

    return mutual_information.item()



def log(x):
    return torch.log(x + 1e-8)
def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))

    if args.dataset_str in ['cornell', 'texas', 'washington', 'wiscosin', 'uai2010']:
        adj, features, label = load_data2(args.dataset_str)
    else:
        adj, features, label = load_data(args.dataset_str)
    #print(adj.shape,features.shape)

    #adj, features = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape
    # features = sp.lil.lil_matrix(features)
    # features, _ = preprocess_features(features)
    # features,_ = preprocess_features(features)
    # features = features.todense()
    # features = torch.from_numpy(features)
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    #print(adj.shape)
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    #adj = adj_train
    adj=adj_orig
    adj_train=adj
    #T=torch.FloatTensor(adj.todense())

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    mini_batch=adj.shape[0]
    print(mini_batch,pos_weight,norm,adj_norm.shape,adj_label.shape)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    print(model)
    below = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    print(below)
    print(below.parameters() == model.parameters())
    D = Discriminator_FC(args.hidden2, args.hidden1, feat_dim)
    print(D)
    #Mutual information estimator
    M = Mine(feat_dim,args.hidden2)
    print(M)
    below_M = Mine(feat_dim, args.hidden2)
    print(below_M)
    G_ma = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    G_ma.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
    M_opt = optim.Adam(M.parameters(), lr=2e-4)
    below_optimizer = optim.Adam(below.parameters(), lr=args.lr)
    below_M_opt = optim.Adam(below_M.parameters(), lr=2e-4)
    below_G_ma = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    below_G_ma.load_state_dict(below.state_dict())

    hidden_emb = None
    NMI=[]
    AC=[]
    embedding=[]
    for epoch in range(args.epochs):
        d_loss=0
        g_loss=0
        cur_loss=0
        t = time.time()
        idx = np.random.permutation(n_nodes)
        shuf_fts = features[idx, :]
        ####################
        model.train()
        optimizer.zero_grad()
        recovered, mu = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()


        D.train()
        D_optimizer.zero_grad()
        

        fakeadj, fakez = below(shuf_fts, adj_norm)
        D_result = D(features, fakez)
        D_fake_loss= D_result
        recovered, mu = model(features, adj_norm)
        D_result=D(features,mu)
        D_real_loss= D_result
        D_train_loss = -torch.mean(log(D_real_loss) + log(1 - D_fake_loss))
        D_train_loss.backward(retain_graph=True)
        d_loss=D_train_loss.item()
        D_optimizer.step()
        #################
        model.train()

        D.eval()
        below.train()
        optimizer.zero_grad()
        below_optimizer.zero_grad()
        

        fakeadj, fakez = model(shuf_fts, adj_norm)
        D_result = D(features, fakez)
        D_fake_loss= D_result
        re, z = below(shuf_fts, adj_norm)
        below_loss = D(features, z)
        recovered, mu = model(features, adj_norm)
        loss=loss_function(preds=recovered, labels=adj_label, norm=norm, pos_weight=pos_weight)
        below_reloss = loss_function(preds=re, labels=adj_label, norm=norm, pos_weight=pos_weight)
        D_result = D(features, mu)
        D_real_loss= D_result
        

        G_train_loss=   args.lambda1*loss
        mi = torch.mean(M(features, mu)) - torch.log(torch.mean(torch.exp(M(features, fakez))) + 1e-8)
        G_train_loss -=  args.reg1 * mi
        G_train_loss.backward(retain_graph=True)
        optimizer.step()
        update_target(G_ma, model)
        learn_mine(model, M, M_opt, features, adj_norm, shuf_fts)
        G2_train_loss= -torch.mean(log(D_real_loss))
        G2_train_loss.backward(retain_graph=True)
        g_loss = G_train_loss.item()
        optimizer.step()
        below_mi = torch.mean(below_M(shuf_fts, z)) - torch.log(torch.mean(torch.exp(below_M(shuf_fts, mu))) + 1e-8)
        below_loss = -torch.mean(log(below_loss)) + args.lambda2*below_reloss
        below_loss -= args.reg2 * below_mi
        below_loss.backward()
        below_optimizer.step()
        update_target(below_G_ma, below)
        learn_mine(below, below_M, below_M_opt, shuf_fts, adj_norm, features)
        ##################
        
        
        hidden_emb = mu.data.numpy()

        nmi,ac=get_NMI(int(label.max()+1),hidden_emb,label)
        NMI.append(nmi)
        AC.append(ac)
        embedding.append(hidden_emb)
        
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              g_loss,-d_loss,'nmi:%.5f'%nmi,'ac:%.5f'%ac,
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")
    import pyecharts.options as opts
    from pyecharts.charts import Line

    #x = list(range(args.epochs))
    #Line().add_xaxis(x).add_yaxis("NMI",NMI, label_opts = opts.LabelOpts(is_show=False)).add_yaxis("AC",AC, label_opts = opts.LabelOpts(is_show=False)).set_global_opts(yaxis_opts= opts.AxisOpts(max_=0.8, min_= 0.3), title_opts=opts.TitleOpts(title="NMI&AC")).render("cora-nmiac.html")
    # line = Line()
    # line.add_xaxis(x)
    # line.add_yaxis('NMI',NMI,markpoint_opts=['max'], is_smooth=True,)
    # line.add_yaxis('JS', JS, is_smooth=True)
    # #line.add('NMI', x, NMI, mark_point=['max', 'min'], mark_line=['average'])
    # #line.add('JS', x, JS)
    # line.render(f'{args.dataset_str}-nmi&JS.html')
    #line1 = pyecharts.Line(f'AC on {args.dataset_str}')
    #line1.add('AC', x, AC, mark_point=['max', 'min'], mark_line=['average'])
    #line1.render(f'{args.dataset_str}-ac.html')
    #roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    #print('Test ROC score: ' + str(roc_score))
    #print('Test AP score: ' + str(ap_score))
    
    print("Optimization Finished!")
    print(max(NMI),NMI.index(max(NMI)))
    print(max(AC),AC.index(max(AC)))
    import scipy.io as sio
    emb=embedding[NMI.index(max(NMI))]
    print(emb.shape)
    #sio.savemat(args.dataset_str+'-Embedding.mat', {"H": emb})
    #sio.savemat('./result/'+args.dataset_str+'-'+str(args.hidden2)+'-'+str(max(NMI))+'-Embedding.mat', {"H": emb})
    #sio.savemat(args.dataset_str+'-lambda1-'+str(args.lambda1)+'-'+str(max(NMI))+'-Embedding.mat', {"H": emb})
    #sio.savemat(args.dataset_str+'-lambda2-'+str(args.lambda2)+'-'+str(max(NMI))+'-Embedding.mat', {"H": emb})


if __name__ == '__main__':
    #args.dataset_str='pubmed'
    gae_for(args)
    '''
    for i in ['cornell', 'texas', 'washington', 'wiscosin']:
        args.dataset_str=i
        gae_for(args)
    '''
    '''
    for i in ['cornell', 'texas', 'washington', 'wiscosin', 'uai2010', 'cora','citeseer','pubmed']:
        args.dataset_str=i
        gae_for(args)
    '''
    '''
    for i in [0.1, 0.5, 1, 5, 10]:
        args.dataset_str='citeseer'
        args.lambda1=i
        gae_for(args)
    '''
    '''
    for i in [16, 32, 64, 128]:
        args.dataset_str='citeseer'
        args.hidden1=i*2
        args.hidden2=i
        gae_for(args)
    '''