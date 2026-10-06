"""Indexed weighted CV+ queries with selectable value and rank searches."""
import ctypes
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import sys

class Policy(str,Enum):
    CALIBRATE_GE = "jawsx-calibrate-ge-append-v1"
    UTILS_STRICT = "jawsx-utils-strict-separate-v1"

class Backend(str,Enum):
    VALUE = "value"
    RANK = "rank"

import numpy as np
PD=ctypes.POINTER(ctypes.c_double);PI=ctypes.POINTER(ctypes.c_int64);PC=ctypes.POINTER(ctypes.c_int)
def immutable(x,dtype):
    a=np.asarray(x,dtype=dtype,order='C')
    return np.frombuffer(a.tobytes(),dtype=dtype).reshape(a.shape)
def ptr(a):return a.ctypes.data_as(PD)
@dataclass(frozen=True,slots=True)
class Endpoint:
    value:float
    path:str
    secondary_fallback:bool=False
@dataclass(frozen=True,slots=True)
class IntervalBatch:
    bounds:np.ndarray
    used_fallback:np.ndarray
    secondary_fallback:np.ndarray
@dataclass(frozen=True,slots=True,init=False)
class FoldIndex:
    _lib:object
    _residual:np.ndarray
    _fold:np.ndarray
    _b:np.ndarray
    _r:np.ndarray
    _pf:np.ndarray
    _pr:np.ndarray
    _offsets:np.ndarray
    _rmax:np.ndarray
    k:int
    supported:bool
    epsilon:float
    backend:Backend

    def __init__(self,residual,fold,b,k,library_path=None,*,backend=Backend.VALUE):
        try: backend=Backend(backend)
        except (ValueError,TypeError):raise ValueError("backend must be 'value' or 'rank'") from None
        if isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 1<=k<=4096:raise ValueError('k must be an integer in1..4096')
        r=np.asarray(residual,dtype=np.float64);f=np.asarray(fold);w=np.asarray(b,dtype=np.float64)
        if r.ndim!=1 or f.ndim!=1 or w.ndim!=1 or r.shape!=f.shape or r.shape!=w.shape or not 1<=len(r)<=2**30:raise ValueError('matching nonempty one-dimensional arrays required')
        if f.dtype.kind not in 'iu' or np.any(f<0) or np.any(f>=k):raise ValueError('fold IDs must be integers in0..k-1')
        if not np.isfinite(r).all() or np.any(r<0) or not np.isfinite(w).all() or np.any(w<0):raise ValueError('residual and b must be finite nonnegative')
        r=immutable(r,np.float64);f=immutable(f,np.int64);w=immutable(w,np.float64)
        library_path=Path(library_path) if library_path is not None else Path(__file__).with_name('weighted.dll' if sys.platform=='win32' else 'weighted.so')
        lib=ctypes.CDLL(str(library_path.resolve()))
        lib.weighted_environment.argtypes=[];lib.weighted_environment.restype=ctypes.c_int
        lib.weighted_flat.argtypes=[PD,PD,ctypes.c_int64,ctypes.c_double,ctypes.c_int,ctypes.c_int,PD,PC]
        lib.weighted_index.argtypes=[PD,PI,PD,PD,PD,PD,ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int,PD,PC]
        lib.weighted_flat.restype=lib.weighted_index.restype=ctypes.c_int
        lib.weighted_value_index.argtypes=lib.weighted_index.argtypes
        lib.weighted_value_index.restype=ctypes.c_int
        lib.weighted_interval_batch.argtypes=[PD,PI,PD,PD,PD,PD,PD,ctypes.c_int64,ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int,PD,PC]
        lib.weighted_interval_batch.restype=ctypes.c_int
        if not lib.weighted_environment():raise RuntimeError('binary64 RN with gradual underflow required')
        supported=bool(np.all((w>=2.**-100)&(w<=2.**100)))
        groups=[np.flatnonzero(f==j) for j in range(k)]
        maximum=np.array([np.max(r[idx]) if len(idx) else 0. for idx in groups])
        groups=[idx[np.argsort(r[idx],kind='stable')] for idx in groups]
        offsets=np.array([0]+list(np.cumsum([len(idx) for idx in groups])),dtype=np.int64)
        ordered=np.concatenate([r[idx] for idx in groups])
        if supported:
            pf=np.concatenate([np.r_[0.,np.cumsum(w[idx])] for idx in groups])
            pr=np.concatenate([np.r_[0.,np.cumsum(w[idx][::-1])] for idx in groups])
        else:pf=pr=np.empty(0)
        for name,value in {'_lib':lib,'_residual':r,'_fold':f,'_b':w,'_r':immutable(ordered,np.float64),
            '_pf':immutable(pf,np.float64),'_pr':immutable(pr,np.float64),'_offsets':immutable(offsets,np.int64),
            '_rmax':immutable(maximum,np.float64),'k':int(k),'supported':supported,'backend':backend}.items():object.__setattr__(self,name,value)
        size=len(r)+k+5;u=2.**-53;object.__setattr__(self,'epsilon',float(np.nextafter(128*size*u/(1-size*u),np.inf)))

    def endpoint(self,centers,a,t,q,sign,*,policy):
        try: policy=Policy(policy)
        except (ValueError,TypeError):raise ValueError("explicit supported Policy required") from None
        if np.ndim(q)!=0 or np.ndim(t)!=0:raise ValueError("scalar q and t required")
        if not isinstance(sign,(int,np.integer)) or sign not in (-1,1) or isinstance(sign,(bool,np.bool_)):raise ValueError('sign must be-1 or+1')
        try:q=float(q);t=float(t)
        except (TypeError,ValueError):raise ValueError('scalar q and t required') from None
        if not np.isfinite(q) or not ((sign==-1 and 0<q<.5) or (sign==1 and .5<q<1)):raise ValueError('q must be in the requested tail, excluding median')
        if not np.isfinite(t) or t<0:raise ValueError('t must be finite nonnegative')
        c=np.array(centers,dtype=np.float64,order='C',copy=True);a=np.array(a,dtype=np.float64,order='C',copy=True)
        if c.shape!=(self.k,) or a.shape!=(self.k,) or not np.isfinite(c).all() or not np.isfinite(a).all() or np.any(a<0):raise ValueError('finite K-vectors; a nonnegative')
        if not self._lib.weighted_environment():raise RuntimeError('floating environment changed')
        with np.errstate(over='ignore',invalid='ignore'):
            if not np.isfinite(c+sign*self._rmax).all():raise FloatingPointError('finite input endpoint overflow')
        out=ctypes.c_double();status=ctypes.c_int()
        if self.supported:
            search=self._lib.weighted_value_index if self.backend is Backend.VALUE else self._lib.weighted_index
            rc=search(ptr(self._r),self._offsets.ctypes.data_as(PI),ptr(self._pf),ptr(self._pr),ptr(c),ptr(a),t,self.k,self.epsilon,q,int(sign),ctypes.byref(out),ctypes.byref(status))
            if rc:raise RuntimeError('native index error')
            if not status.value:return Endpoint(out.value,'index')
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            cal=a[self._fold]*self._b;raw=np.r_[cal,t];den=float(np.sum(raw) if policy is Policy.CALIBRATE_GE else np.sum(cal)+t);values=np.r_[c[self._fold]+sign*self._residual,-np.inf if sign<0 else np.inf]
            if not np.isfinite(raw).all() or not np.isfinite(den) or den<=0:raise FloatingPointError('raw mass or normalization overflow; no silent rescaling')
            weights=raw/den
        rc=self._lib.weighted_flat(ptr(values),ptr(weights),len(values),q,1,int(policy is Policy.UTILS_STRICT),ctypes.byref(out),ctypes.byref(status))
        if rc or np.isnan(out.value):raise FloatingPointError('native error or no floating CDF crossing')
        return Endpoint(out.value,'fallback',bool(status.value))

    def interval(self,centers,a,t,alpha=.1,*,policy):
        return self.endpoint(centers,a,t,alpha,-1,policy=policy),self.endpoint(centers,a,t,1-float(alpha),1,policy=policy)

    def scs_interval(self,centers,test_weight,alpha=.1):
        """utils wCV SCS contract: b=train ratios, a=ones, test atom=ratio."""
        return self.interval(centers,np.ones(self.k),test_weight,alpha,policy=Policy.UTILS_STRICT)

    def interval_batch(self,centers,a,t,alpha=.1,*,policy):
        """Owned Q-by-K query inputs, one native call, immutable Q-by-2 results.

        Exact scalar-policy fallback is retained for the flagged endpoints.
        Empty batches are supported. Model/calibration state is not rebuilt.
        """
        try:policy=Policy(policy)
        except (ValueError,TypeError):raise ValueError('explicit supported Policy required') from None
        if np.ndim(alpha)!=0 or not np.isfinite(float(alpha)) or not 0<float(alpha)<.5:
            raise ValueError('finite scalar alpha in(0,.5) required')
        alpha=float(alpha)
        c=np.array(centers,dtype=np.float64,order='C',copy=True)
        factors=np.array(a,dtype=np.float64,order='C',copy=True)
        test=np.array(t,dtype=np.float64,order='C',copy=True)
        if c.ndim!=2 or c.shape[1]!=self.k or factors.shape!=c.shape or test.shape!=(len(c),):
            raise ValueError('centers and a must be Q-by-K; t must be a Q-vector')
        if not np.isfinite(c).all() or not np.isfinite(factors).all() or np.any(factors<0) or not np.isfinite(test).all() or np.any(test<0):
            raise ValueError('finite query arrays; a and t nonnegative')
        if not self._lib.weighted_environment():raise RuntimeError('floating environment changed')
        with np.errstate(over='ignore',invalid='ignore'):
            if not np.isfinite(c-self._rmax).all() or not np.isfinite(c+self._rmax).all():
                raise FloatingPointError('finite input endpoint overflow')
        bounds=np.empty((len(c),2),dtype=np.float64)
        status=np.zeros(bounds.shape,dtype=np.int32) if self.supported else np.ones(bounds.shape,dtype=np.int32)
        secondary=np.zeros(bounds.shape,dtype=bool)
        if len(c) and self.supported:
            rc=self._lib.weighted_interval_batch(ptr(self._r),self._offsets.ctypes.data_as(PI),
                ptr(self._pf),ptr(self._pr),ptr(c),ptr(factors),ptr(test),len(c),self.k,self.epsilon,
                alpha,int(self.backend is Backend.VALUE),ptr(bounds),status.ctypes.data_as(PC))
            if rc:raise RuntimeError('native batch index error')
        # NumPy normalization/reduction and policy-sensitive ties keep the same
        # scalar behavior. Only the ambiguous endpoints return to Python.
        for i,j in np.argwhere(status):
            ep=self.endpoint(c[i],factors[i],test[i],alpha if j==0 else 1.-alpha,
                             -1 if j==0 else 1,policy=policy)
            bounds[i,j]=ep.value;secondary[i,j]=ep.secondary_fallback
            status[i,j]=int(ep.path=='fallback')
        return IntervalBatch(immutable(bounds,np.float64),immutable(status!=0,np.bool_),
                             immutable(secondary,np.bool_))
