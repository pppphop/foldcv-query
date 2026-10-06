"""Direct saved-state, two-tail queries; author software remains unmodified."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import ctypes as C
import csv
import importlib.util
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'software/src'))
sys.path.insert(0,str(ROOT/'upstream/FavMac'))
from foldcv_query.engine import FoldIndex, Policy
from foldcv_query import __version__
from quantiletree import QuantileTree
D=C.POINTER(C.c_double); I=C.POINTER(C.c_int64)
def dp(x):return x.ctypes.data_as(D)
def ip(x):return x.ctypes.data_as(I)
native=C.CDLL(str(ROOT/('native_cfcausal_flat.dll' if sys.platform=='win32' else 'native_cfcausal_flat.so')))
native.cfcausal_flat.argtypes=[D,I,D,C.c_int64,D,D,C.c_double,C.c_double,C.c_int,D,C.POINTER(C.c_int)]
METHODS=('numpy_flat','native_flat','numpy_prefix','author_tree','public_value')

def save(path,x):path.write_text(json.dumps(x,indent=2),encoding='utf-8')

class Prefix:
    def __init__(self,r,f,b,k):
        self.r=[];self.pf=[]
        for j in range(k):
            g=np.flatnonzero(f==j);g=g[np.argsort(r[g],kind='stable')]
            self.r.append(np.array(r[g]));self.pf.append(np.r_[0.,np.cumsum(b[g])])
        self.k=k
        self.total=sum(p[-1] for p in self.pf)
    def bounds_mass(self,j,c,p):
        r=self.r[j];n=len(r)
        i=int(np.searchsorted(r,p-c,side='right'))
        # Correct inverse-translation rounding using actual endpoint comparisons.
        while i>0 and c+r[i-1]>p:i-=1
        while i<n and c+r[i]<=p:i+=1
        return (c+r[i-1] if i else -np.inf,
                c+r[i] if i<n else np.inf,float(self.pf[j][i]))
    def extreme(self,j,c):return c+self.r[j][0],c+self.r[j][-1]
    def payload(self):return sum(r.nbytes+p.nbytes for r,p in zip(self.r,self.pf))

class Trees:
    def __init__(self,r,f,b,k):
        self.trees=[];self.k=k
        for j in range(k):
            tree=QuantileTree(debug=False)
            for i in np.flatnonzero(f==j):tree.insert(float(r[i]),float(b[i]))
            self.trees.append(tree)
        self.total=sum(t.root.sum for t in self.trees)
    def bounds_mass(self,j,c,p):
        tree=self.trees[j];node=tree.root;left=right=tree.nil
        while node!=tree.nil:
            if c+node.val<=p:left=node;node=node.right
            else:right=node;node=node.left
        # Original author CDF query; links above only find neighboring shifted atoms.
        mass=tree.query_sum(left.val,inclusive=True) if left!=tree.nil else 0.
        return (c+left.val if left!=tree.nil else -np.inf,
                c+right.val if right!=tree.nil else np.inf,float(mass))
    def extreme(self,j,c):
        t=self.trees[j];return c+t.minimum(t.root).val,c+t.maximum(t.root).val
    def payload(self):
        seen=set()
        def size(x):
            if id(x) in seen:return 0
            seen.add(id(x));n=sys.getsizeof(x)
            if isinstance(x,dict):n+=sum(size(k)+size(v) for k,v in x.items())
            elif isinstance(x,(list,tuple)):n+=sum(size(v) for v in x)
            elif hasattr(x,'__dict__') and not isinstance(x,type):n+=size(x.__dict__)
            return n
        return size(self.trees)

def indexed_cutoff(state,c,target):
    extremes=[state.extreme(j,c[j]) for j in range(state.k)]
    lo=min(v[0] for v in extremes);hi=max(v[1] for v in extremes)
    for it in range(160):
        if lo==hi:return lo,it
        p=lo+(hi-lo)*.5
        bounds=[state.bounds_mass(j,c[j],p) for j in range(state.k)]
        mass=sum(v[2] for v in bounds)
        if mass>target:hi=max(v[0] for v in bounds)
        else:lo=min(v[1] for v in bounds)
        if not np.isfinite(lo+hi) or lo>hi:raise RuntimeError('invalid discrete search bracket')
    raise RuntimeError('discrete search did not converge')

def prepare(method,r,f,b,k):
    if method=='numpy_prefix':return Prefix(r,f,b,k)
    if method=='author_tree':return Trees(r,f,b,k)
    if method=='public_value':return FoldIndex(r,f,b,k)
    # All methods own calibration input state; flat methods need no index.
    return (r.copy(),f.copy(),b.copy())

def invoke(method,state,r,f,b,centers,t):
    k=centers.shape[1];values=np.empty((len(t),2));iterations=[];fallback=0
    cal=float(np.sum(b));ones=np.ones(k)
    for i,(c,tm) in enumerate(zip(centers,t)):
        qt=(1+float(tm)/cal)*.9
        if qt>=1:values[i]=[-np.inf,np.inf];continue
        threshold=(qt-1e-12)*cal
        qeff=.9-1e-12*cal/(cal+float(tm))
        for j,cs in enumerate((-c,c)):
            if method in ('author_tree','numpy_prefix'):
                v,it=indexed_cutoff(state,cs,threshold);iterations.append(it)
            elif method=='public_value':
                ep=state.endpoint(cs,ones,float(tm),qeff,1,policy=Policy.UTILS_STRICT)
                v=ep.value;fallback+=int(ep.path=='fallback')
            elif method=='numpy_flat':
                vals=cs[f]+r;order=np.argsort(vals,kind='stable')
                cw=np.cumsum(b[order]/cal)
                v=vals[order[np.searchsorted(cw,qt-1e-12,side='right')]]
            elif method=='native_flat':
                rv,fv,bv=state
                out=C.c_double();status=C.c_int()
                # Existing native fused selector has JAWS lower rule. Both calls
                # here are upper cutoffs, followed by negation for the lower tail.
                cs=np.ascontiguousarray(cs)
                native.cfcausal_flat(dp(rv),ip(fv),dp(bv),len(rv),
                    dp(cs),dp(ones),float(tm),qeff,1,C.byref(out),C.byref(status))
                v=out.value;fallback+=bool(status.value)
            values[i,j]=-v if j==0 else v
    return values,dict(fallbacks=fallback,max_iterations=max(iterations,default=0))

def difference(a,b):
    finite=np.isfinite(a)&np.isfinite(b);diff=np.abs(a[finite]-b[finite])
    return dict(max_finite_difference=float(diff.max(initial=0)),
                finite_tolerance_disagreements=int(np.count_nonzero(diff>1e-10)),
                nonfinite_disagreements=int(np.count_nonzero(~finite & (a!=b))),
                exact_disagreements=int(np.count_nonzero(a!=b)))

