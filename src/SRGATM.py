import model
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math
import numpy as np
import datetime

device = torch.device('cuda:0')
class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class SRGAT(nn.Module):
    def __init__(self, args, n_items, n_rels, D_node, adj_entity, adj_relation):
        super().__init__()
        self.n_items = n_items
        self.n_rels = n_rels

        self.emb_size = args.emb_size
        self.hidden_size = args.emb_size
        self.n_layers = args.n_layers
        self.emb_dropout = args.emb_dropout
        self.hidden_dropout = args.hidden_dropout
        self.gradient_clip = args.gradient_clip

        self.order = args.order
        self.neibor_size = args.neibor_size
        self.attention = args.attention
        self.aggregate = args.aggregate

        self.D_node = D_node
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.batch_size = args.batchSize
        self.nonhybrid = args.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=args.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

        self.model_init()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def model_init(self):
        self.rel_emb_table = nn.Embedding(self.n_rels, self.emb_size, padding_idx=0)
        self.item_emb_table = nn.Embedding(self.n_items, self.emb_size, padding_idx=0)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # local interest
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)  # global perference
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.linear_attention_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_attention = nn.Linear(self.hidden_size, 1, bias=True)

        self.linear_attr = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        if self.aggregate == 'concat':
            self.linear_attention_transform = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.linear_attention_transform = nn.Linear(self.hidden_size, self.hidden_size)

        self.gru = nn.GRU(self.emb_size, self.hidden_size, self.n_layers,
                          dropout=self.hidden_dropout, batch_first=True)
        self.emb_dropout_layer = nn.Dropout(self.emb_dropout)
        self.loss_function = nn.CrossEntropyLoss()

        self.final_activation = nn.ReLU()
        self.activation_sigmoid = nn.Sigmoid()

    def get_entitygat(self,args, batch):
        engat = model1.transform_input(h_items.to(device), h_attrs.to(device), t_item.to(device))

        return engat

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def forward(model, i, data, entitygat):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    qstar = torch.cat(hidden,entitygat)
    get = lambda i: qstar [i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, mask)

def train_test(model, train_data, test_data, batch, args):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        qt = model.get_entitygat(args, batch)
        targets, scores = forward(model, i, train_data,qt)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr




