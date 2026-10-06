"""Reproduce six existing-software and six changing-factor saved-state cases."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import adapters as ext
import dynamic_adapters as dyn

ROOT = Path(__file__).resolve().parent
CASES = ('rf', 'ridge', 'housing_rf', 'housing_ridge', 'bike_rf', 'bike_ridge')

def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument('--rscript', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import foldcv_query
    save(args.out/'configuration.json', dict(cases=CASES, queries=[32,512], repetitions=3,
        python=sys.executable, python_version=platform.python_version(), numpy=np.__version__,
        software_version=foldcv_query.__version__, package_path=foldcv_query.__file__,
        working_directory=str(Path.cwd()), entrypoint=str(Path(__file__).resolve()),
        fitting_included=False, cfcausal_commit='e4ca6706599873505e1cce2d8cf0078ceff5d2e6',
        FavMac_commit='0130e63b86ac0092b4359bb9993113b4956c87ea',
        scope='saved-state endpoint comparison; dynamic factors extend the tree adapter, not original cfcausal'))
    records=[]; dynamic=[]
    for case in CASES:
        folder=args.out/case; folder.mkdir()
        with np.load(ROOT/'cases'/f'{case}.npz') as data:
            r=data['residual']; f=data['fold']; b=data['b']; c=data['centers']; t=data['t']; a=data['a']
        k=c.shape[1]
        np.savetxt(folder/'calibration.csv',np.column_stack((r,f,b)),delimiter=',',header='residual,fold,b',comments='',fmt='%.17g')
        np.savetxt(folder/'centers.csv',c,delimiter=',',header=','.join(f'c{j}' for j in range(k)),comments='',fmt='%.17g')
        np.savetxt(folder/'testmass.txt',t,fmt='%.17g')
        renv=os.environ.copy()
        for key in ('LANG','LC_ALL','LC_CTYPE'): renv.pop(key,None)
        subprocess.run([args.rscript,'--vanilla',str(ROOT/'run_cfcausal.R'),str(ROOT/'upstream/cfcausal'),str(folder)],check=True,env=renv)
        with (folder/'cfcausal_times.csv').open(encoding='utf-8') as stream: rt=list(csv.DictReader(stream))
        ref={q:np.genfromtxt(folder/f'cfcausal_q{q}_rep1.csv',delimiter=',',skip_header=1) for q in (32,512)}
        prep={m:[] for m in ext.METHODS}; states={}; storage={}
        for rep in range(3):
            for method in ext.METHODS[rep:]+ext.METHODS[:rep]:
                start=time.perf_counter_ns(); state=ext.prepare(method,r,f,b,k)
                prep[method].append((time.perf_counter_ns()-start)/1e9); states[method]=state
                if method in ('numpy_prefix','author_tree'): storage[method]=state.payload()
                elif method=='public_value': storage[method]=sum(getattr(state,n).nbytes for n in ('_residual','_fold','_b','_r','_pf','_pr','_offsets','_rmax'))
                else: storage[method]=sum(v.nbytes for v in state)
        row=dict(case=case,n=len(r),k=k,prepare_seconds=prep,retained_bytes=storage,queries=[])
        arrays={}
        for q in (32,512):
            times={m:[] for m in ext.METHODS}; diffs={m:[] for m in ext.METHODS}
            for rep in range(3):
                for method in ext.METHODS[rep:]+ext.METHODS[:rep]:
                    start=time.perf_counter_ns(); values,info=ext.invoke(method,states[method],r,f,b,c[:q],t[:q])
                    times[method].append((time.perf_counter_ns()-start)/1e9)
                    delta=ext.difference(ref[q],values); diffs[method].append(delta)
                    if delta['finite_tolerance_disagreements'] or delta['nonfinite_disagreements']: raise RuntimeError(f'cfcausal mismatch {case}/{method}')
                    arrays[f'{method}_q{q}_rep{rep}']=values
            rr=[v for v in rt if int(v['q'])==q]; times['cfcausal']=[float(v['query_seconds']) for v in rr]
            totals={m:float(np.median(prep[m])+np.median(times[m])) for m in ext.METHODS}
            totals['cfcausal']=float(np.median([float(v['prepare_seconds']) for v in rr])+np.median(times['cfcausal']))
            row['queries'].append(dict(q=q,query_seconds=times,differences=diffs,median_prepare_plus_query_seconds=totals))
        reference=dyn.reference(r,f,b,c,a,t)
        dtimes={m:[] for m in dyn.METHODS}; ddiffs={m:[] for m in dyn.METHODS}
        for rep in range(3):
            for method in dyn.METHODS[rep:]+dyn.METHODS[:rep]:
                start=time.perf_counter_ns(); values,info=dyn.invoke(method,states[method],r,f,b,c,a,t)
                dtimes[method].append((time.perf_counter_ns()-start)/1e9)
                delta=ext.difference(reference,values); ddiffs[method].append(delta)
                if delta['finite_tolerance_disagreements'] or delta['nonfinite_disagreements']: raise RuntimeError(f'dynamic mismatch {case}/{method}')
                arrays[f'dynamic_{method}_rep{rep}']=values
        dynamic.append(dict(case=case,n=len(r),q=len(t),query_seconds=dtimes,differences=ddiffs))
        np.savez_compressed(folder/'outputs.npz',**arrays)
        save(folder/'results.json',row); records.append(row)
        save(args.out/'progress.json',dict(existing_software=records,dynamic_factors=dynamic))
        print(json.dumps(dict(case_complete=case,n=len(r))),flush=True)
    save(args.out/'results.json',dict(existing_software=records,dynamic_factors=dynamic))
    print('ALL_TWELVE_PORTABLE_CASES_COMPLETE',flush=True)

if __name__ == '__main__': main()
