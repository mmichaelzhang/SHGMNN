import argparse
from tqdm import tqdm

import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as acc 
from sklearn.metrics import average_precision_score as aupr 
from sklearn.metrics import precision_score as ps
from sklearn.metrics import f1_score as f1

from logger import Logger


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert self.x.size(0) == self.y.size(0)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, LW, device):
        super(MLP, self).__init__()
        self.LW = LW
        self.device = device
        self.coe = torch.nn.Parameter(torch.randn((2), requires_grad = True))
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.coe = torch.nn.Parameter(torch.randn((2), requires_grad = True))

    def forward(self, x):
        if (self.LW):
            p = F.softmax(self.coe.to(self.device), dim = 0)
        else:
            p = F.softmax(torch.ones_like(self.coe).to(self.device), dim = 0)
        x = x @ p
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)

def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y.squeeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(model, device, loader):
    model.eval()

    y_raw, y_pred, y_true = [], [], []
    for x, y in tqdm(loader):
        x = x.to(device)
        out = model(x)

        y_pred.append(torch.argmax(out, dim=1, keepdim=True).cpu())
        y_raw.append(out.cpu().data)
        y_true.append(y)

    y_raw = torch.exp(torch.cat(y_raw, dim = 0))[:,1]
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    _, s_t = zip(*sorted(zip(y_raw, y_true), reverse = True))
    s_t = torch.tensor(s_t)
    
    p_10 = ps(s_t[:10], torch.ones_like(s_t[:10]))
    p_50 = ps(s_t[:50], torch.ones_like(s_t[:50]))
    p_100 = ps(s_t[:100], torch.ones_like(s_t[:100]))
    acc_score = acc(y_true, y_pred)
    auc_score = auc(y_true, y_raw)
    aupr_score = aupr(y_true, y_raw)
    f1_score = f1(y_true, y_pred)

    return acc_score, p_10, p_50, p_100, auc_score, aupr_score, f1_score

def main():
    parser = argparse.ArgumentParser(description='Scalable Heterogeneous Graph Markov Neural Network (SHGMNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--embeddings_file_name', type=str, default='processed')
    parser.add_argument('--result_file_name', type=str, default='results.txt')
    parser.add_argument('--LW', type=int, default=1)
    parser.add_argument('--INF', type=int, default=1)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)


    name = '../Data/'

    try:
        data_dict1 = torch.load(name + args.embeddings_file_name + '_1.pt')
        data_dict2 = torch.load(name + args.embeddings_file_name + '_2.pt')
    except:
        raise RuntimeError('File {} not found. Need to run python preprocessing.py first'.format(args.embeddings_file_name))

    split_idx = data_dict1['split_idx']
    x = torch.cat(data_dict1['op_embedding'], dim=1)
    y = data_dict1['label'].to(torch.long)
    x2 = torch.cat(data_dict2['op_embedding'], dim=1)
    x = torch.cat((x.unsqueeze(-1), x2.unsqueeze(-1)), dim = 2)

    num_classes = 2
    print('Input feature dimension: {}'.format(x.shape[1]))
    print('Total number of nodes: {}'.format(x.shape[0]))

    train_dataset = SimpleDataset(x[split_idx['train']], y[split_idx['train']])
    valid_dataset = SimpleDataset(x[split_idx['valid']], y[split_idx['valid']])
    test_dataset = SimpleDataset(x[split_idx['test']], y[split_idx['test']])
    y_0 = torch.clone(y)

    non_train_id = {}
    with open(name + 'non_train_idx.txt','r') as f:
        for line in f:
            a = int(line.strip('\n').split()[0])
            b = int(line.strip('\n').split()[1])
            non_train_id[b] = a
    count = 0
    with open(name + 'Inference.txt','r') as f:
        for line in f:
            y_0[non_train_id[count]] = float(line.strip('\n'))
            count += 1

    all_dataset_train = SimpleDataset(x, y_0)
    all_dataset_valid = SimpleDataset(x, y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    all_loader_train = DataLoader(all_dataset_train, batch_size=args.batch_size, shuffle=False)
    all_loader_valid = DataLoader(all_dataset_valid, batch_size=args.batch_size, shuffle=False)

    model = MLP(
        x.size(1),
        args.hidden_channels,
        num_classes,
        args.num_layers,
        args.dropout,
        args.LW,
        device).to(device)

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of parameters: {}.'.format(num_trainable_parameters))
    
    logger = Logger(args.runs, info=args, file_name=args.result_file_name)

    for run in range(args.runs):
        
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            if (args.INF == 1):
                train(model, device, all_loader_train, optimizer)
            else:
                train(model, device, train_loader, optimizer)
            p = model.state_dict()
            print(F.softmax(p['coe'], dim = 0).to('cpu').data)
            train_acc,_,_,_,_,_,_ = test(model, device, train_loader)
            valid_acc,_,_,_,_,_,_ = test(model, device, valid_loader)
            test_acc,p_10, p_50, p_100, auc_score, aupr_score, f1_score  = test(model, device, test_loader)

            logger.add_result(run, (train_acc, valid_acc, test_acc, p_10, p_50, p_100, auc_score, aupr_score, f1_score))

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%, '
                      f'   Time: {time.time()}')

        logger.print_statistics(run)
    logger.print_statistics()

if __name__=='__main__':
    main()
