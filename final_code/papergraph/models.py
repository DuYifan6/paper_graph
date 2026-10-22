"""Equations (14)-(20). Parameter counts are measured, never padded to old totals."""
import torch
from torch import nn
from .graphs import Graph

def edge_tensors(graph,device,valid=None):
    u=torch.as_tensor(graph.u,dtype=torch.long,device=device)
    v=torch.as_tensor(graph.v,dtype=torch.long,device=device)
    w=torch.as_tensor(graph.weight,device=device)
    if valid is not None:
        keep=valid[u]&valid[v];u,v,w=u[keep],v[keep],w[keep]
    return u,v,w

def aggregate(h,graph):
    u,v,w=edge_tensors(graph,h.device)
    out=torch.zeros_like(h);degree=h.new_zeros(len(h))
    out.index_add_(0,u,w[:,None]*h[v]);out.index_add_(0,v,w[:,None]*h[u])
    degree.index_add_(0,u,w);degree.index_add_(0,v,w)
    return out/(degree[:,None]+1e-6)

class GraphMix(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.gate=nn.Linear(2*hidden,hidden)
        self.message=nn.Linear(2*hidden,hidden)
    def forward(self,h,graph=None,identity=False):
        neighbor=h if identity else aggregate(h,graph)
        cat=torch.cat((h,neighbor),dim=-1)
        return h+torch.sigmoid(self.gate(cat))*torch.relu(self.message(cat))

class SelfOnlyEncoder(nn.Module):
    """Self-only 20x48 input, exactly the GraphMix budget; no unused filler weights.

    A fixed sparse projection exposes every input coordinate, with 512 trainable
    connections per output for the paper's H=128. Every weight affects the output.
    Bias and channel gates give H*(4H+2), the same budget as GraphMix.
    """
    def __init__(self,hidden=128,input_size=960):
        super().__init__();width=4*hidden
        if width>input_size:raise ValueError('Self encoder projection wider than input')
        step=(input_size+hidden-1)//hidden
        if width<step:raise ValueError('Parameter budget cannot cover this HF input size')
        index=(torch.arange(hidden)[:,None]*step+torch.arange(width)[None,:])%input_size
        self.register_buffer('index',index);self.input_size=input_size
        self.weight=nn.Parameter(torch.empty(hidden,width));self.bias=nn.Parameter(torch.zeros(hidden))
        self.gate=nn.Parameter(torch.zeros(hidden));nn.init.xavier_uniform_(self.weight)
    def forward(self,x):
        dense=self.weight.new_zeros(len(self.bias),self.input_size).scatter(1,self.index,self.weight)
        return torch.sigmoid(self.gate)*torch.relu(torch.nn.functional.linear(x,dense,self.bias))

class Forecaster(nn.Module):
    def __init__(self,features=9,hidden=128,variant='graph2',hf_size=960):
        super().__init__();self.variant=variant
        self.temporal=nn.LSTM(features,hidden,num_layers=1,batch_first=True)
        self.graph=SelfOnlyEncoder(hidden,hf_size) if variant=='5m_nograph' else GraphMix(hidden)
        self.head=nn.Linear(hidden,1)
    def forward(self,sequence,graph=None,hf=None):
        _,(state,_)=self.temporal(sequence);h=state[-1]
        if self.variant=='5m_nograph':h=h+self.graph(hf)
        elif self.variant!='zero':h=self.graph(h,graph,identity=self.variant=='identity')
        return self.head(h).squeeze(-1)

class NeighborGCN(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__();self.own=nn.Linear(inputs,outputs);self.neighbor=nn.Linear(inputs,outputs)
        self.norm=nn.LayerNorm(outputs);self.dropout=nn.Dropout(.1)
    def forward(self,x,graph):
        return self.dropout(torch.relu(self.norm(self.own(x)+self.neighbor(aggregate(x,graph)))))

class GCNForecaster(nn.Module):
    """External two-GCN / LSTM benchmark, sharing the exact Graph-2 store."""
    variant='gcn_lstm'
    def __init__(self,features=9,hidden=128):
        super().__init__();self.graph=nn.ModuleList([NeighborGCN(features,64),NeighborGCN(64,64)])
        self.temporal=nn.LSTM(64,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Linear(hidden//2,1))
    def forward(self,sequence,graphs,hf=None):
        states=[]
        for i,graph in enumerate(graphs):
            h=sequence[:,i]
            for layer in self.graph:h=layer(h,graph)
            states.append(h)
        _,(state,_)=self.temporal(torch.stack(states,dim=1))
        return self.head(state[-1]).squeeze(-1)

def laplacian_loss(pred,graph,valid):
    """Eq (19): 2/|V| y' L_sym y. Zero-degree diagonal is explicitly zero.

    The undirected edge sum is y' L_sym y. It is a SUM, never mean over
    padded neighbor slots. Degrees are recomputed on the same induced valid set.
    """
    valid=valid.bool();n=valid.sum()
    if int(n)==0:raise ValueError('Empty evaluation set for loss')
    u,v,w=edge_tensors(graph,pred.device,valid)
    degree=torch.zeros_like(pred)
    degree.index_add_(0,u,w);degree.index_add_(0,v,w)
    safe_degree=torch.where(degree>0,degree,torch.ones_like(degree))
    normalized=pred/torch.sqrt(safe_degree)
    energy=(w*(normalized[u]-normalized[v]).square()).sum()
    return 2*energy/n

def parameter_audit(model,sequence,graph=None,hf=None):
    model.zero_grad(set_to_none=True)
    model(sequence,graph,hf).square().sum().backward()
    result={'registered':sum(p.numel() for p in model.parameters()),
            'active':sum(p.numel() for p in model.parameters() if p.grad is not None),
            'graph_registered':sum(p.numel() for p in model.graph.parameters()),
            'head_registered':sum(p.numel() for p in model.head.parameters())}
    model.zero_grad(set_to_none=True)
    return result
