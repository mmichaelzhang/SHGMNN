import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from gen_admm_data import cal_inference
import matlab.engine
from torch_geometric.data.data import Data

def get_adj(row, col, value, N, asymm_norm=False, set_diag=True, remove_diag=False):
    
    adj = SparseTensor(row=row, col=col, value = value, sparse_sizes=(N, N))
    if set_diag:
        print('... setting diagonal entries')
        adj = adj.set_diag()
    elif remove_diag:
        print('... removing diagonal entries')
        adj = adj.remove_diag()
    else:
        print('... keeping diag elements as they are')
    if not asymm_norm:
        print('... performing symmetric normalization')
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    else:
        print('... performing asymmetric normalization')
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        adj = deg_inv.view(-1, 1) * adj

    adj = adj.to_scipy(layout='csr')
    
    return adj

def main():
    parser = argparse.ArgumentParser(description='Scalable Heterogeneous Graph Markov Neural Network (SHGMNN)')
    parser.add_argument('--file_name', type=str, default="processed")
    parser.add_argument('--undirected_num_propagations', type=int, default=3)
    parser.add_argument('--undirected_dropedge_rate', type=float, default=0.4)
    parser.add_argument('--undirected_asymm_norm', action='store_true')
    parser.add_argument('--undirected_remove_diag', action='store_true')
    parser.add_argument('--undirected_set_diag', action='store_true')
    
    args = parser.parse_args()

    name = '../Data/'

    x = []
    with open(name + 'Embeddings.txt','r') as f:
        for line in f:
            xx = []
            for e in line.strip('\n').split():
                xx.append(float(e))
            x.append(xx)
    x = torch.tensor(x)

    train_idx = []
    with open(name + 'Train_idx.txt','r') as f:
        for line in f:
            train_idx.append(int(line.strip('\n')))
    train_idx = torch.tensor(train_idx)    

    valid_idx = []
    with open(name + 'Validate_idx.txt','r') as f:
        for line in f:
            valid_idx.append(int(line.strip('\n')))
    valid_idx = torch.tensor(valid_idx)    

    test_idx = []
    with open(name + 'Test_idx.txt','r') as f:
        for line in f:
            test_idx.append(int(line.strip('\n')))
    test_idx = torch.tensor(test_idx)

    edge_index1 = [[],[]]
    edge_attr1 = []
    with open(name + 'edges1.txt','r') as f:
        for line in f:
            a = int(line.strip('\n').split('\t')[0])
            b = int(line.strip('\n').split('\t')[1])
            c = float(line.strip('\n').split('\t')[2])
            edge_index1[0].append(a)
            edge_index1[1].append(b)
            edge_attr1.append([c])
    edge_attr1 = torch.tensor(edge_attr1)
    edge_index1 = torch.tensor(edge_index1)

    edge_index2 = [[],[]]
    edge_attr2 = []
    with open(name + 'edges2.txt','r') as f:
        for line in f:
            a = int(line.strip('\n').split('\t')[0])
            b = int(line.strip('\n').split('\t')[1])
            c = float(line.strip('\n').split('\t')[2])
            edge_index2[0].append(a)
            edge_index2[1].append(b)
            edge_attr2.append([c])
    edge_attr2 = torch.tensor(edge_attr2)
    edge_index2 = torch.tensor(edge_index2)

    y = []
    with open(name + 'Labels.txt','r') as f:
        for line in f:
            y.append([int(line.strip('\n'))])
    y = torch.tensor(y)

    data1 = Data(x = x, edge_index = edge_index1, y = y, edge_attr = edge_attr1)
    data2 = Data(x = x, edge_index = edge_index2, y = y, edge_attr = edge_attr2)

    datas = [data1, data2]
    for i in range(len(datas)):
        data = datas[i]
        x = data.x.numpy()
        N = data.num_nodes

        all_idx = torch.cat([train_idx, valid_idx, test_idx])

        op_dict = {}
        op_dict['label'] = data.y.data[all_idx].to(torch.long)
        op_dict['split_idx'] = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

        op_dict['op_embedding'] = []
        op_dict['op_embedding'].append(torch.from_numpy(x[all_idx]).to(torch.float))

        print('Start processing')
        
        # subsample operator
        print('Subsampling (dropping {} %)'.format(100 * args.undirected_dropedge_rate))
        edge_index, edge_attr = dropout_adj(data.edge_index, p=args.undirected_dropedge_rate, num_nodes=data.num_nodes)
        
        # to undirected
        print('Making the graph undirected')
        edge_index = to_undirected(edge_index, data.num_nodes)
        row, col = edge_index
        
        # get adj
        print('Getting adj matrix')
        adj = get_adj(row, col, edge_attr, N, asymm_norm=args.undirected_asymm_norm, set_diag=args.undirected_set_diag, remove_diag=args.undirected_remove_diag)
        # preprocessing of features
        print('Diffusing node features')

        x = data.x.numpy()
        for _ in tqdm(range(args.undirected_num_propagations)):
            x = adj @ x
            op_dict['op_embedding'].append(torch.from_numpy(x[all_idx]).to(torch.float))
        torch.save(op_dict, '{}.pt'.format(name + args.file_name + '_' + str(i+1)))

main()
cal_inference()
eng = matlab.engine.start_matlab()
eng.ADMM(nargout = 0)