"""Actual whole-pool acquisition jobs; independently fit every backend trajectory."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ctypes as C
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent

# Import the new public package before loading the frozen flat comparator.
sys.path.insert(0,str(ROOT/'software/src'))
from foldcv_query import Backend, FoldIndex, Policy, __version__


D=C.POINTER(C.c_double); I=C.POINTER(C.c_int64)
flatlib=C.CDLL(str(ROOT/('flat_batch.dll' if sys.platform=='win32' else 'flat_batch.so')))
flatlib.fused_flat_batch.argtypes=[D,I,D,C.c_int64,D,D,D,C.c_int64,C.c_int,C.c_double,D,C.POINTER(C.c_int)]
flatlib.weighted_environment.argtypes=[];flatlib.weighted_environment.restype=C.c_int
from foldcv_query.engine import immutable

def make_flat(residual,fold,b,k):
    r=np.array(residual,copy=True);f=np.array(fold,dtype='i8',copy=True);w=np.array(b,copy=True)
    rmax=np.array([np.max(r[f==j]) if np.any(f==j) else 0. for j in range(k)])
    return dict(r=r,f=f,b=w,k=k,rmax=rmax)

def public_kernels(state,inputs):
    centers,factors,test=inputs
    if state['method']=='value_batch':
        result=state['idx'].interval_batch(centers,factors,test,policy=Policy.CALIBRATE_GE)
        return result.bounds,result.used_fallback
    if state['method']=='flat_batch':
        s=state['flat'];k=s['k'];c=np.array(centers,dtype='f8',order='C',copy=True)
        a=np.array(factors,dtype='f8',order='C',copy=True);t=np.array(test,dtype='f8',order='C',copy=True)
        if c.ndim!=2 or c.shape[1]!=k or a.shape!=c.shape or t.shape!=(len(c),):raise ValueError('Q-by-K centers/factors and Q-vector test masses')
        if not np.isfinite(c).all() or not np.isfinite(a).all() or np.any(a<0) or not np.isfinite(t).all() or np.any(t<0):raise ValueError('finite inputs, nonnegative masses')
        if not flatlib.weighted_environment():raise RuntimeError('floating environment changed')
        with np.errstate(over='ignore',invalid='ignore'):
            if not np.isfinite(c-s['rmax']).all() or not np.isfinite(c+s['rmax']).all():raise FloatingPointError('endpoint overflow')
        out=np.empty((len(c),2),dtype='f8');status=np.zeros(out.shape,dtype=np.int32)
        rc=flatlib.fused_flat_batch(s['r'].ctypes.data_as(D),s['f'].ctypes.data_as(I),s['b'].ctypes.data_as(D),len(s['r']),c.ctypes.data_as(D),a.ctypes.data_as(D),t.ctypes.data_as(D),len(c),k,.1,out.ctypes.data_as(D),status.ctypes.data_as(C.POINTER(C.c_int)))
        if rc or np.isnan(out).any():raise FloatingPointError('native flat batch failed')
        return immutable(out,np.float64),immutable(status!=0,np.bool_)
    bounds=np.empty((len(test),2));fallback=np.zeros(bounds.shape,dtype=bool)
    for i in range(len(test)):
        lower,upper=state['idx'].interval(centers[i],factors[i],float(test[i]),policy=Policy.CALIBRATE_GE)
        bounds[i]=[lower.value,upper.value];fallback[i]=[lower.path=='fallback',upper.path=='fallback']
    return bounds,fallback
METHODS=('flat_batch','value_scalar','value_batch')
SEEDS=(20261011,20261012,20261013)

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def job(args):
    folder=args.out/f'seed{args.seed}_{args.method}'
    folder.mkdir(exist_ok=False)
    with np.load(ROOT/'data/housing.npz') as data:
        raw_x=data['X']; y=np.log1p(data['y'])
    order=np.random.default_rng(args.seed).permutation(len(y))
    holdout=order[:10000]; train=np.sort(order[10000:10200]); pool=np.sort(order[10200:])
    start=time.perf_counter_ns()
    scaler=StandardScaler().fit(raw_x[train]); x=scaler.transform(raw_x)
    preprocessing_ns=time.perf_counter_ns()-start
    measurements=[]; outputs=dict(holdout_ids=holdout,initial_train_ids=train.copy())
    for iteration in range(100):
        begin=time.perf_counter_ns()
        permutation=np.random.default_rng(args.seed+iteration+1).permutation(len(train))
        fold=np.empty(len(train),dtype='i8'); fold[permutation]=np.repeat(np.arange(5),len(train)//5)
        models=tuple(Ridge().fit(x[train[fold!=k]],y[train[fold!=k]]) for k in range(5))
        full=Ridge().fit(x[train],y[train])
        fit_end=time.perf_counter_ns()
        domain_x=np.concatenate((x[train],x[pool]))
        domain_y=np.r_[np.zeros(len(train),dtype='i8'),np.ones(len(pool),dtype='i8')]
        density=CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=50,max_depth=4,n_jobs=1,
                                   random_state=args.seed+iteration),
            method='isotonic',cv=3,n_jobs=1)
        density.fit(domain_x,domain_y)
        probabilities=np.clip(density.predict_proba(domain_x)[:,1],1e-8,1-1e-8)
        weights=np.clip(probabilities/(1-probabilities)*len(train)/len(pool),.05,20.)
        b=np.ascontiguousarray(weights[:len(train)])
        test=np.ascontiguousarray(weights[len(train):])
        density_end=time.perf_counter_ns()
        oof=np.empty(len(train))
        for k,model in enumerate(models):
            mask=fold==k; oof[mask]=model.predict(x[train[mask]])
        residual=np.ascontiguousarray(np.abs(y[train]-oof))
        index=FoldIndex(residual,fold,b,5) if args.method!='flat_batch' else None
        flat=make_flat(residual,fold,b,5) if args.method=='flat_batch' else None
        state=dict(r=residual,f=fold,b=b,k=5,idx=index,flat=flat,method=args.method)
        prepare_end=time.perf_counter_ns()
        centers=np.column_stack([model.predict(x[pool]) for model in models])
        factors=np.ones_like(centers)
        predict_end=time.perf_counter_ns()
        bounds,fallback=public_kernels(state,(centers,factors,test))
        query_end=time.perf_counter_ns()
        widths=bounds[:,1]-bounds[:,0]
        if np.isnan(widths).any():
            raise FloatingPointError('NaN acquisition width; keep partial record, stop job')
        selected=pool[np.lexsort((pool,-widths))[:25]]
        holdout_mse=float(np.mean((full.predict(x[holdout])-y[holdout])**2))
        next_pool=pool[~np.isin(pool,selected,assume_unique=True)]
        next_train=np.sort(np.r_[train,selected])
        end=time.perf_counter_ns()
        record=dict(iteration=iteration,train_rows=len(train),queries=len(pool),
            fit_ns=fit_end-begin,density_ns=density_end-fit_end,
            preparation_ns=prepare_end-density_end,prediction_ns=predict_end-prepare_end,
            interval_ns=query_end-predict_end,selection_evaluation_ns=end-query_end,
            total_ns=end-begin,fallback_endpoints=int(fallback.sum()),
            infinite_widths=int(np.isinf(widths).sum()),holdout_point_mse=holdout_mse)
        measurements.append(record)
        prefix=f'round{iteration:03d}'
        outputs[prefix+'_pool_ids']=pool.copy()
        outputs[prefix+'_selected_ids']=selected.copy()
        outputs[prefix+'_bounds']=bounds
        outputs[prefix+'_fallback']=fallback
        outputs[prefix+'_calibration_weights']=b
        outputs[prefix+'_test_weights']=test
        train,pool=next_train,next_pool
        if iteration%10==9:
            write(folder/'progress.json',measurements)
            print(json.dumps(dict(seed=args.seed,method=args.method,round=iteration+1,
                operation_seconds=sum(r['total_ns'] for r in measurements)/1e9,
                last_round_seconds=record['total_ns']/1e9,fallbacks=record['fallback_endpoints'])),flush=True)
    outputs['final_train_ids']=train; outputs['final_pool_ids']=pool
    report=dict(seed=args.seed,method=args.method,preprocessing_ns=preprocessing_ns,
        whole_task_ns=preprocessing_ns+sum(r['total_ns'] for r in measurements),rounds=measurements)
    np.savez_compressed(folder/'outputs.npz',**outputs)
    write(folder/'result.json',report)
    print(json.dumps(dict(job_complete=True,seed=args.seed,method=args.method,
        whole_task_seconds=report['whole_task_ns']/1e9)),flush=True)

def compare(out,seed):
    comparisons=[]
    with np.load(out/f'seed{seed}_flat_batch/outputs.npz') as ref:
        for method in ('value_scalar','value_batch'):
            with np.load(out/f'seed{seed}_{method}/outputs.npz') as other:
                finite_diff=0.; nonfinite_diff=0; selected_diff=0; pool_diff=0
                for name in ref.files:
                    if name.endswith('_bounds'):
                        a,b=ref[name],other[name]
                        if a.shape!=b.shape:
                            raise RuntimeError('Divergent pool size in actual acquisition jobs')
                        finite=np.isfinite(a)&np.isfinite(b)
                        if finite.any(): finite_diff=max(finite_diff,float(np.max(np.abs(a[finite]-b[finite]))))
                        nonfinite_diff+=int(np.count_nonzero(~finite&(a!=b)))
                    elif name.endswith('_selected_ids'):
                        selected_diff+=int(np.count_nonzero(ref[name]!=other[name]))
                    elif name.endswith('_pool_ids'):
                        pool_diff+=int(np.count_nonzero(ref[name]!=other[name]))
                comparisons.append(dict(method=method,max_finite_endpoint_difference=finite_diff,
                    nonfinite_disagreements=nonfinite_diff,selected_id_disagreements=selected_diff,
                    candidate_id_disagreements=pool_diff))
    write(out/f'comparison_seed{seed}.json',comparisons)
    print(json.dumps(dict(seed=seed,comparisons=comparisons)),flush=True)

def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    write(args.out/'configuration.json',dict(seeds=SEEDS,methods=METHODS,rounds=100,
        initial_labels=200,acquisition_batch=25,holdout=10000,k=5,alpha_levels=[.1,.9],
        ratio_clip=[.05,20],python=platform.python_version(),numpy=np.__version__,
        sklearn=sklearn.__version__,platform=platform.platform(),numerical_threads=1,
        software_version=__version__,public_indexed_api=True,flat_backend='validated one-call fused native flat batch',
        application='three-seed complete public scalar/batch integration comparison'))
    results=[]
    for number,seed in enumerate(SEEDS):
        for method in METHODS[number:]+METHODS[:number]:
            command=[sys.executable,str(Path(__file__).resolve()),'--child','--out',str(args.out),
                     '--seed',str(seed),'--method',method]
            log=args.out/f'seed{seed}_{method}.log'
            with log.open('w',encoding='utf-8') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                print(json.dumps(dict(job_started=True,seed=seed,method=method,pid=process.pid)),flush=True)
                status=process.wait()
            if status:
                raise RuntimeError(f'Actual job failed: {seed}/{method}; see {log}')
            result=json.loads((args.out/f'seed{seed}_{method}/result.json').read_text(encoding='utf-8'))
            results.append(result); write(args.out/'progress.json',results)
            print(json.dumps(dict(job_complete=True,seed=seed,method=method,
                whole_task_seconds=result['whole_task_ns']/1e9)),flush=True)
        compare(args.out,seed)
    write(args.out/'results.json',results)
    print('ALL_NINE_PUBLIC_BATCH_APPLICATION_JOBS_COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=lambda value:Path(value).resolve(),required=True)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--method',choices=METHODS)
    args=parser.parse_args()
    job(args) if args.child else main(args)
