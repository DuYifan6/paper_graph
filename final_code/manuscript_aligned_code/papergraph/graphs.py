"""Positive dependence, directed Top-s then UNDIRECTED UNION with no second cap."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import SCHEMA, write_json

@dataclass
class Graph:
    # Each undirected edge is stored once, u < v. Variable degree is preserved.
    n: int
    u: np.ndarray
    v: np.ndarray
    weight: np.ndarray
    valid: np.ndarray
    def validate(self):
        self.u=np.asarray(self.u,dtype=np.int64);self.v=np.asarray(self.v,dtype=np.int64)
        self.weight=np.asarray(self.weight,dtype=np.float32);self.valid=np.asarray(self.valid,dtype=bool)
        if self.valid.shape!=(self.n,) or not(self.u.shape==self.v.shape==self.weight.shape):
            raise ValueError('Graph dimensions disagree')
        if len(self.u):
            if np.any(self.u<0) or np.any(self.v>=self.n) or np.any(self.u>=self.v):
                raise ValueError('Undirected storage requires 0 <= u < v < N')
            if len(np.unique(self.u*self.n+self.v))!=len(self.u):raise ValueError('Duplicate edges')
            if not np.isfinite(self.weight).all() or np.any(self.weight<=0):raise ValueError('Bad weights')
            if not(self.valid[self.u].all() and self.valid[self.v].all()):raise ValueError('Invalid-node edge')
        return self
    def degree(self):
        return np.bincount(np.r_[self.u,self.v],minlength=self.n)
    def fingerprint(self):
        h=hashlib.sha256(str(self.n).encode())
        for a in (self.u,self.v,self.weight,self.valid):h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    def audit(self):
        self.validate();degree=self.degree();nv=int(self.valid.sum())
        return {'nodes':self.n,'valid_nodes':nv,'undirected_edges':len(self.u),
                'symmetric':True,'self_loops':0,'duplicate_edges':0,
                'max_degree':int(degree.max(initial=0)),
                'density':2*len(self.u)/(nv*(nv-1)) if nv>1 else 0.,
                'isolated_valid_nodes':int(np.sum(self.valid & (degree==0))),
                'fingerprint':self.fingerprint()}

def empty_graph(n,valid=None):
    return Graph(n,np.array([],dtype=np.int64),np.array([],dtype=np.int64),
                 np.array([],dtype=np.float32),np.zeros(n,dtype=bool) if valid is None else valid).validate()

def positive_graph(samples,valid,neighbors=40,threshold=.10,binary=True):
    """samples[N, time]; clip -> equal-weight market removal -> Pearson -> Top-s."""
    x=np.asarray(samples,dtype=np.float64);valid=np.asarray(valid,dtype=bool).copy()
    if x.ndim!=2 or valid.shape!=(x.shape[0],):raise ValueError('Bad return matrix')
    valid &= np.isfinite(x).all(axis=1)
    if x.shape[1]<2 or valid.sum()<2:return empty_graph(len(valid),valid)
    ids=np.flatnonzero(valid)
    x=np.clip(x[ids],-.08,.08)
    x-=x.mean(axis=0,keepdims=True)  # completed/available observations only
    x-=x.mean(axis=1,keepdims=True)
    sd=x.std(axis=1,ddof=1)
    keep=sd>1e-12
    valid[ids[~keep]]=False;ids=ids[keep];x=x[keep];sd=sd[keep]
    if len(ids)<2:return empty_graph(len(valid),valid)
    z=x/sd[:,None]
    similarity=np.maximum(np.clip((z@z.T)/(x.shape[1]-1),-1,1),0)
    np.fill_diagonal(similarity,0)
    # Stable tie handling, by original node order. No absolute correlations.
    k=min(neighbors,len(ids)-1)
    pairs={}
    for i in range(len(ids)):
        order=np.argsort(-similarity[i],kind='stable')[:k]
        for j in order:
            w=float(similarity[i,j])
            if i==j or w<=0:continue
            if binary and w<threshold:continue
            a,b=sorted((int(ids[i]),int(ids[j])))
            pairs[(a,b)]=1. if binary else max(pairs.get((a,b),0.),w)
    ordered=sorted(pairs)
    return Graph(len(valid),np.array([a for a,b in ordered]),np.array([b for a,b in ordered]),
                 np.array([pairs[p] for p in ordered]),valid).validate()

def graph_for_target(returns,available,t,config,variant='graph2'):
    """Row t labels target date t. Graph uses raw return dates [t-W, t), never t."""
    n=returns.shape[1]
    if t<config.window:return empty_graph(n)
    valid=np.asarray(available[t-config.window:t],dtype=bool).all(axis=0)
    window=np.asarray(returns[t-config.window:t],dtype=np.float64)
    if variant=='graph3':window=window[:,:,-24:]
    if config.protocol=='hybrid1455':
        if variant!='graph2':raise ValueError('Hybrid14:55 currently specifies Graph-2 only')
        # Raw input requires 49 aligned closes / 48 returns; last return ends 15:00.
        samples=np.concatenate([window[:-1].transpose(1,0,2).reshape(n,-1),window[-1,:,:47]],axis=1)
    else:samples=window.transpose(1,0,2).reshape(n,-1)
    return positive_graph(samples,valid,config.neighbors,config.threshold,binary=variant!='graph1')

def rewire(graph,seed,swaps_per_edge=5,max_attempts_factor=30,strict=True):
    """Undirected double-edge swaps; preserve exact node degrees and reciprocity."""
    graph.validate()
    if not np.all(graph.weight==1):raise ValueError('RandomGraph reference must be binary')
    rng=np.random.default_rng(seed);edges=list(zip(graph.u.tolist(),graph.v.tolist()))
    original=set(edges);existing=set(edges);target=int(np.ceil(swaps_per_edge*len(edges)))
    success=attempts=0;limit=max_attempts_factor*max(target,1)
    while success<target and attempts<limit and len(edges)>=2:
        attempts+=1;i,j=rng.choice(len(edges),2,replace=False)
        a,b=edges[i];c,d=edges[j]
        if rng.integers(2):a,b=b,a
        if rng.integers(2):c,d=d,c
        if len({a,b,c,d})!=4:continue
        p=tuple(sorted((a,d)));q=tuple(sorted((c,b)))
        if p in existing or q in existing or p==q:continue
        existing.remove(edges[i]);existing.remove(edges[j]);existing.update((p,q))
        edges[i]=p;edges[j]=q;success+=1
    if strict and target and success<target:
        raise RuntimeError(f'Rewiring incomplete: {success}/{target} swaps after {attempts} attempts')
    ordered=sorted(edges)
    result=Graph(graph.n,np.array([a for a,b in ordered]),np.array([b for a,b in ordered]),
                 np.ones(len(ordered),dtype=np.float32),graph.valid.copy()).validate()
    if not np.array_equal(result.degree(),graph.degree()):raise AssertionError('Degree changed')
    audit=result.audit()|{'successful_swaps':success,'requested_swaps':target,'attempts':attempts,
                         'edge_overlap':len(original&existing)/len(original) if original else 0.,
                         'rewiring_complete':success==target,'seed':int(seed)}
    return result,audit

class GraphStore:
    def __init__(self,path,identity,config,variant):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.manifest={'schema':SCHEMA,'identity':identity,'config':config.dictionary(),'variant':variant}
        file=self.path/'manifest.json'
        if file.exists() and json.loads(file.read_text(encoding='utf-8'))!=self.manifest:
            raise ValueError('Incompatible graph cache: use a new directory; old KNN caches are rejected')
        if not file.exists():write_json(file,self.manifest)
    def get(self,t,builder):
        file=self.path/f'day_{t:04d}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as z:
                g=Graph(int(z['n']),z['u'],z['v'],z['weight'],z['valid']).validate()
                if str(z['fingerprint'].item())!=g.fingerprint():raise ValueError('Corrupt graph cache')
            return g
        graph=builder().validate()
        # Never write old padded Top-40 arrays under the new schema.
        with file.open('wb') as f:np.savez_compressed(f,n=graph.n,u=graph.u,v=graph.v,
            weight=graph.weight,valid=graph.valid,fingerprint=graph.fingerprint())
        return graph
