import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from foldcv_query import Backend,FoldIndex,Policy,UniformFinitePoolFCS
from test_weighted_contract import reference

class BatchContract(unittest.TestCase):
    def test_independent_materialized_tails_and_scalar_flags(self):
        rng=np.random.default_rng(20261017)
        for case in range(40):
            n=int(rng.integers(10,301));k=int(rng.integers(2,9));q=32
            r=rng.exponential(size=n);f=rng.integers(k,size=n,dtype='i8');b=np.exp(rng.uniform(-8,8,n))
            c=rng.normal(size=(q,k));a=np.exp(rng.uniform(-6,6,(q,k)));t=np.exp(rng.uniform(-12,12,q))
            if case%3==0:r=np.round(r);c=np.round(c)
            if case%5==0:b[0]=0.
            t[::11]=0.;a[::13,0]=0.;t[::17]=2.**110
            for backend in Backend:
                idx=FoldIndex(r,f,b,k,backend=backend)
                for policy in Policy:
                    batch=idx.interval_batch(c,a,t,policy=policy)
                    for i in range(q):
                        pair=idx.interval(c[i],a[i],t[i],policy=policy)
                        for j,(level,sign) in enumerate(((.1,-1),(.9,1))):
                            self.assertEqual(batch.bounds[i,j],reference(r,f,b,c[i],a[i],t[i],level,sign,policy))
                            self.assertEqual(batch.bounds[i,j],pair[j].value)
                            self.assertEqual(batch.used_fallback[i,j],pair[j].path=='fallback')
                            self.assertEqual(batch.secondary_fallback[i,j],pair[j].secondary_fallback)
                    self.assertFalse(batch.bounds.flags.writeable)

    def test_inputs_and_fitted_application(self):
        from sklearn.linear_model import Ridge
        rng=np.random.default_rng(20261018);pool=rng.normal(size=(300,4));ids=np.arange(60)
        y=pool[ids]@np.array([1.,.2,-.4,.7])+rng.normal(scale=.3,size=60)
        session=UniformFinitePoolFCS.fit(pool,ids,y,Ridge(),nfolds=3,seed=17)
        queries=session.sample_query_indices(240,seed=42);got=session.predict_indices(queries)
        idx=session._index
        scalar=np.array([[e.value for e in idx.interval(c,a,t,policy=Policy.CALIBRATE_GE)] for c,a,t in zip(got.centers,got.a,got.t)])
        np.testing.assert_array_equal(got.bounds,scalar)
        empty=idx.interval_batch(np.empty((0,3)),np.empty((0,3)),np.empty(0),policy=Policy.CALIBRATE_GE)
        self.assertEqual(empty.bounds.shape,(0,2))
        with self.assertRaises(ValueError):idx.interval_batch([[1,2]],[[1,2]],[1],policy=Policy.CALIBRATE_GE)
        with self.assertRaises(ValueError):idx.interval_batch([[1,2,3]],[[1,-1,1]],[1],policy=Policy.CALIBRATE_GE)
        with self.assertRaises(ValueError):idx.interval_batch([[1,2,3]],[[1,1,1]],[1],alpha=.5,policy=Policy.CALIBRATE_GE)

if __name__=='__main__':unittest.main(verbosity=2)
