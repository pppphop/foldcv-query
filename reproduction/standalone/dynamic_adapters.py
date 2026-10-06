import json
import pickle
import sys
import time
import ctypes as C
from pathlib import Path
import adapters as ext
import numpy as np

METHODS=('numpy_prefix','author_tree','native_flat','public_value')

def indexed(state,c,a,threshold):
    extremes=[state.extreme(j,c[j]) for j in range(state.k)]
    lo=min(x[0] for x in extremes);hi=max(x[1] for x in extremes)
    for it in range(160):
        if lo==hi:return lo,it
        p=lo+(hi-lo)*.5
        if p==hi:p=lo
        bounds=[state.bounds_mass(j,c[j],p) for j in range(state.k)]
        mass=sum(float(a[j])*v[2] for j,v in enumerate(bounds))
        if mass>threshold:hi=max(v[0] for v in bounds)
        else:lo=min(v[1] for v in bounds)
    raise RuntimeError('search limit')

def invoke(method,state,r,f,b,c,a,t):
    values=np.empty((len(t),2));iterations=[];fallbacks=0
    foldmass=np.array([np.sum(b[f==j]) for j in range(c.shape[1])])
    for i in range(len(t)):
        cal=float(np.dot(a[i],foldmass));qt=(1+float(t[i])/cal)*.9
        if qt>=1:values[i]=[-np.inf,np.inf];continue
        target=(qt-1e-12)*cal;qeff=.9-1e-12*cal/(cal+float(t[i]))
        ai=np.ascontiguousarray(a[i])
        for j,cs in enumerate((-c[i],c[i])):
            if method in ('author_tree','numpy_prefix'):
                value,it=indexed(state,cs,ai,target);iterations.append(it)
            elif method=='public_value':
                ep=state.endpoint(cs,ai,float(t[i]),qeff,1,policy=ext.Policy.UTILS_STRICT)
                value=ep.value;fallbacks+=ep.path=='fallback'
            else:
                rr,ff,bb=state;out=C.c_double();status=C.c_int();cs=np.ascontiguousarray(cs)
                ext.native.cfcausal_flat(ext.dp(rr),ext.ip(ff),ext.dp(bb),len(rr),
                    ext.dp(cs),ext.dp(ai),float(t[i]),qeff,1,C.byref(out),C.byref(status))
                value=out.value;fallbacks+=bool(status.value)
            values[i,j]=-value if j==0 else value
    return values,dict(fallbacks=int(fallbacks),max_iterations=max(iterations,default=0))

def reference(r,f,b,c,a,t):
    values=np.empty((len(t),2))
    for i in range(len(t)):
        w=a[i,f]*b;cal=float(np.sum(w));qt=(1+float(t[i])/cal)*.9
        if qt>=1:values[i]=[-np.inf,np.inf];continue
        for j,cs in enumerate((-c[i],c[i])):
            vals=cs[f]+r;order=np.argsort(vals,kind='stable')
            value=vals[order[np.searchsorted(np.cumsum(w[order]/cal),qt-1e-12,side='right')]]
            values[i,j]=-value if j==0 else value
    return values

