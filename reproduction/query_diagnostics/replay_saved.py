"""Self-contained saved-state search diagnostics: NumPy + local C++ build."""
from pathlib import Path
import sys
import ctypes as C
import argparse
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'software/src'))
from foldcv_query.engine import FoldIndex,Backend,Policy,ptr,PI

class Trace(C.Structure):
    _fields_=[('iterations',C.c_int),('reason',C.c_int),('pivot',C.c_double),('ties',C.c_int64)]

def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    library=ROOT/'trace.dll'
    lib=C.CDLL(str(library))
    PD=C.POINTER(C.c_double);PC=C.POINTER(C.c_int)
    lib.diagnostic_index.argtypes=[PD,PI,PD,PD,PD,PD,C.c_double,C.c_int,C.c_double,C.c_double,C.c_int,C.c_int,PD,PC,C.POINTER(Trace)]
    reports=[]
    for folder in sorted((ROOT/'states').iterdir()):
        if args.case and folder.name!=args.case:continue
        with np.load(folder/'state.npz') as data:
            r,f,b,c,a,t=(data[name] for name in ('residual','fold','b','centers','a','t'))
        with np.load(folder/'trace.npz') as data:original={key:data[key] for key in data.files}
        arrays={};record=dict(case=folder.name,queries=len(t),backends=[])
        for backend in (Backend.VALUE,Backend.RANK):
            idx=FoldIndex(r,f,b,c.shape[1],library_path=library,backend=backend)
            bounds=np.empty((len(t),2));iters=np.empty(bounds.shape,dtype='i4');reasons=np.empty(bounds.shape,dtype='i4')
            for i in range(len(t)):
                ci=np.ascontiguousarray(c[i]);ai=np.ascontiguousarray(a[i])
                for j,(sign,level) in enumerate(((-1,.1),(1,.9))):
                    value=C.c_double();status=C.c_int();trace=Trace()
                    lib.diagnostic_index(ptr(idx._r),idx._offsets.ctypes.data_as(PI),ptr(idx._pf),ptr(idx._pr),ptr(ci),ptr(ai),float(t[i]),idx.k,idx.epsilon,level,sign,int(backend is Backend.VALUE),C.byref(value),C.byref(status),C.byref(trace))
                    iters[i,j]=trace.iterations;reasons[i,j]=trace.reason
                    bounds[i,j]=idx.endpoint(ci,ai,float(t[i]),level,sign,policy=Policy.CALIBRATE_GE).value if status.value else value.value
            arrays.update({backend.value+'_bounds':bounds,backend.value+'_iterations':iters,backend.value+'_reason':reasons})
            for suffix,x in (('bounds',bounds),('iterations',iters),('reason',reasons)):
                if not np.array_equal(x,original[backend.value+'_'+suffix]):
                    raise RuntimeError(f'Reproduced diagnostic differs: {folder.name}/{backend}/{suffix}')
            record['backends'].append(dict(backend=backend.value,fallbacks=int(np.count_nonzero(reasons)),maximum_iterations=int(iters.max()),reproduced_endpoints=True,reproduced_iterations=True,reproduced_reasons=True))
        np.savez_compressed(args.out/(folder.name+'.npz'),**arrays)
        reports.append(record);print(json.dumps(record),flush=True)
    if not reports:raise ValueError('No requested saved state')
    (args.out/'results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=lambda x:Path(x).resolve(),required=True)
    parser.add_argument('--case');main(parser.parse_args())
