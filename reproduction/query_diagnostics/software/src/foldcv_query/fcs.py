"""Uniform finite-pool feedback-shift workflow. Existing statistical method.

Training targets alone define clipped score preferences. Query labels are never
an input. The finite pool is the complete declared base-distribution support,
not a Monte Carlo approximation to an unspecified continuous distribution.
"""
import json
import pickle
from dataclasses import dataclass
import numpy as np
from .engine import Backend,FoldIndex,Policy,immutable

def _integer_vector(x,length=None,maximum=None,name='indices'):
    a=np.asarray(x)
    if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iu':raise ValueError(name+' must be a nonempty integer vector')
    if length is not None and len(a)!=length:raise ValueError(name+' length mismatch')
    if np.any(a<0) or (maximum is not None and np.any(a>=maximum)):raise ValueError(name+' outside declared support')
    return immutable(a,np.int64)

def _pred(model,x):
    result=np.asarray(model.predict(x),dtype=np.float64)
    if result.shape!=(len(x),) or not np.isfinite(result).all():raise ValueError('predict must return one finite scalar per row')
    return result

def _tilt(pred,location,scale,strength):
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        result=np.exp(-strength*np.clip((pred-location)/scale,-3.,3.))
    if not np.isfinite(result).all() or np.any(result<=0):raise FloatingPointError('finite positive tilt required; no silent rescaling')
    return result

@dataclass(frozen=True,slots=True)
class FCSBatch:
    indices:np.ndarray
    point:np.ndarray
    centers:np.ndarray
    a:np.ndarray
    t:np.ndarray
    bounds:np.ndarray
    used_fallback:np.ndarray
    secondary_fallback:np.ndarray

    def upper_below(self,threshold):
        if np.ndim(threshold)!=0 or not np.isfinite(float(threshold)):raise ValueError('finite scalar threshold required')
        return immutable(self.bounds[:,1]<=float(threshold),np.bool_)

@dataclass(frozen=True,slots=True,init=False)
class UniformFinitePoolFCS:
    _pool:np.ndarray
    _models:tuple
    _model_snapshots:tuple
    _index:FoldIndex
    _location:np.ndarray
    _scale:np.ndarray
    _z:np.ndarray
    _probabilities:np.ndarray
    _provenance_json:str
    _strength:float

    @classmethod
    def prepare(cls,pool,train_indices,ytrain,fold,fitted_models,*,strength=1.,allow_unequal_folds=False,backend=Backend.VALUE):
        """Snapshot trusted caller-owned fitted estimators, then prepare masses.

        Models are in fold order, followed by the full model. They must have
        been fitted on the respective complements/full training data. The
        prepared session retains its own copies of the fitted predictors.
        No arbitrary feature-query API is exposed: predict_indices uses pool IDs.
        """
        x=np.asarray(pool,dtype=np.float64)
        if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():raise ValueError('finite nonempty two-dimensional pool required')
        ids=_integer_vector(train_indices,maximum=len(x),name='training pool indices')
        if len(ids)>2**30:raise ValueError('training size exceeds index contract')
        y=np.asarray(ytrain,dtype=np.float64)
        if y.shape!=(len(ids),) or not np.isfinite(y).all():raise ValueError('finite training labels matching IDs required')
        if not isinstance(allow_unequal_folds,(bool,np.bool_)):raise ValueError('explicit Boolean unequal-fold choice required')
        supplied=tuple(fitted_models);k=len(supplied)-1
        if not 2<=k<=4096:raise ValueError('K fold models plus one full model;2<=K<=4096')
        f=_integer_vector(fold,length=len(ids),maximum=k,name='fold IDs');counts=np.bincount(f,minlength=k)
        if np.any(counts==0):raise ValueError('each fold must contain training rows')
        equal=bool(np.all(counts==counts[0]))
        if not equal and not allow_unequal_folds:raise ValueError('unequal folds require explicit arithmetic-only opt-in')
        if np.ndim(strength)!=0 or not np.isfinite(float(strength)) or float(strength)<0:raise ValueError('finite nonnegative scalar strength required')
        strength=float(strength);x=immutable(x,np.float64);y=immutable(y,np.float64)
        # Clone through caller-owned pickle snapshots, not public/remote artifacts.
        snapshots=[pickle.dumps(m,protocol=5) for m in supplied];models=tuple(pickle.loads(blob) for blob in snapshots)
        xt=x[ids];oof=np.empty(len(y));location=[];scale=[]
        for j,model in enumerate(models[:-1]):
            mask=f==j;oof[mask]=_pred(model,xt[mask]);location.append(float(np.mean(y[~mask])));scale.append(float(np.std(y[~mask])))
        location.append(float(np.mean(y)));scale.append(float(np.std(y)));location=np.asarray(location);scale=np.asarray(scale)
        if not np.isfinite(location).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('positive finite training score scales required')
        # Preserve full-matrix reduction shape of frozen reference workflow.
        population=np.column_stack([_pred(m,x) for m in models]);u=_tilt(population,location,scale,strength);z=np.mean(u,axis=0)
        if not np.isfinite(z).all() or np.any(z<=0):raise FloatingPointError('normalizers must be finite positive')
        b=_tilt(oof,location[f],scale[f],strength)/z[f]
        residual=np.abs(y-oof)
        if not np.isfinite(b).all() or not np.isfinite(residual).all():raise FloatingPointError('calibration mass/residual overflow')
        probabilities=u[:,-1]/np.sum(u[:,-1])
        if not np.isfinite(probabilities).all() or np.any(probabilities<0):raise FloatingPointError('query probabilities overflow')
        index=FoldIndex(residual,f,b,k,backend=backend)
        provenance={
            'base_distribution':'uniform over supplied finite pool slots',
            'pool_rows':len(x),'features':x.shape[1],'training_rows':len(y),
            'k':k,'fold_counts':counts.tolist(),'equal_folds':equal,
            'unequal_folds_arithmetic_only':not equal,'strength':strength,
            'score_clip':[-3.,3.],'source_policy':Policy.CALIBRATE_GE.value,
            'backend':index.backend.value,'score_location':location.tolist(),
            'score_scale':scale.tolist(),'normalizers':z.tolist()}
        obj=object.__new__(cls)
        for name,value in {'_pool':x,'_models':models,'_model_snapshots':tuple(snapshots),'_index':index,'_location':immutable(location,np.float64),'_scale':immutable(scale,np.float64),'_z':immutable(z,np.float64),'_probabilities':immutable(probabilities,np.float64),'_provenance_json':json.dumps(provenance,sort_keys=True),'_strength':strength}.items():object.__setattr__(obj,name,value)
        return obj

    @classmethod
    def fit(cls,pool,train_indices,ytrain,estimator,*,nfolds=5,seed=20261004,strength=1.,backend=Backend.VALUE):
        """Optional sklearn lifecycle: fit K complements/full model, then prepare.

        Balanced folds enforced here; sklearn is an optional dependency. Every
        predictor used in querying belongs to the private supplied-model copy.
        """
        if isinstance(nfolds,(bool,np.bool_)) or not isinstance(nfolds,(int,np.integer)) or not 2<=nfolds<=4096:raise ValueError('integer nfolds in2..4096 required')
        x=np.asarray(pool,dtype=np.float64)
        if x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():raise ValueError('finite pool matrix required')
        ids=_integer_vector(train_indices,maximum=len(x));y=np.asarray(ytrain,dtype='f8')
        if y.shape!=(len(ids),) or not np.isfinite(y).all() or len(ids)%nfolds or len(ids)<nfolds:raise ValueError('finite labels and training count divisible by K required')
        from sklearn.base import clone
        from sklearn.model_selection import KFold
        xt=x[ids];models=[];fold=np.empty(len(y),dtype=np.int64)
        for j,(train,cal) in enumerate(KFold(nfolds,shuffle=True,random_state=seed).split(xt)):
            models.append(clone(estimator).fit(xt[train],y[train]));fold[cal]=j
        models.append(clone(estimator).fit(xt,y))
        return cls.prepare(x,ids,y,fold,models,strength=strength,backend=backend)

    @property
    def backend(self):return self._index.backend

    @property
    def model_snapshot_bytes(self):
        """Construction-time snapshots of caller-owned fitted models."""
        return self._model_snapshots

    @property
    def provenance(self):return json.loads(self._provenance_json)

    @property
    def query_probabilities(self):return self._probabilities

    def sample_query_indices(self,count,*,seed):
        if isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or not 1<=count<=2**30:raise ValueError('positive integer query count<=2**30 required')
        return immutable(np.random.default_rng(seed).choice(len(self._pool),size=int(count),replace=True,p=self._probabilities),np.int64)

    def predict_indices(self,indices,alpha=.1):
        ids=_integer_vector(indices,maximum=len(self._pool))
        if np.ndim(alpha)!=0 or not np.isfinite(float(alpha)) or not 0<float(alpha)<.5:raise ValueError('finite scalar alpha in(0,.5) required')
        x=self._pool[ids];point=_pred(self._models[-1],x);centers=np.column_stack([_pred(m,x) for m in self._models[:-1]])
        a=_tilt(centers,self._location[:-1],self._scale[:-1],self._strength)/self._z[:-1]
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):t=(_tilt(point,self._location[-1],self._scale[-1],self._strength)/self._z[-1])**2
        if not np.isfinite(a).all() or not np.isfinite(t).all():raise FloatingPointError('query factor overflow')
        batch=self._index.interval_batch(centers,a,t,float(alpha),policy=Policy.CALIBRATE_GE)
        return FCSBatch(immutable(ids,np.int64),immutable(point,np.float64),immutable(centers,np.float64),immutable(a,np.float64),immutable(t,np.float64),batch.bounds,batch.used_fallback,batch.secondary_fallback)
