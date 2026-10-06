"""Public backends against independently materialized discrete quantiles."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from foldcv_query import Backend,FoldIndex,Policy,UniformFinitePoolFCS

def reference(residual,fold,b,centers,a,t,q,sign,policy):
    calibration=a[fold]*b
    mass=np.r_[calibration,t]
    denominator=np.sum(mass) if policy is Policy.CALIBRATE_GE else np.sum(calibration)+t
    mass=mass/denominator
    values=np.r_[centers[fold]+sign*residual,-np.inf if sign<0 else np.inf]
    order=np.lexsort((np.arange(len(values)),mass,values))
    cumulative=np.cumsum(mass[order])
    crossing=np.flatnonzero(cumulative>=q if policy is Policy.CALIBRATE_GE else cumulative>q)[0]
    if sign>0 or crossing==0 or (policy is Policy.CALIBRATE_GE and cumulative[crossing]==q):
        return values[order[crossing]]
    return values[order[crossing-1]]

class PublicBackendContract(unittest.TestCase):
    def test_materialized_reference_tails(self):
        rng=np.random.default_rng(20261015)
        for case in range(600):
            n=int(rng.integers(1,301)); k=int(rng.integers(1,13))
            residual=rng.exponential(size=n)
            fold=rng.integers(0,k,size=n,dtype='i8')
            b=np.exp(rng.uniform(-8,8,size=n))
            centers=rng.normal(size=k); factors=np.exp(rng.uniform(-6,6,size=k))
            t=float(np.exp(rng.uniform(-12,12)))
            if case%5==0: residual=np.round(residual,0); centers=np.round(centers,0)
            if case%7==0: b[rng.integers(n)]=0
            if case%11==0: t=0.
            if case%13==0: factors[rng.integers(k)]=0.
            if case%17==0: t=2.**110
            alpha=(.01,.1,.3)[case%3]
            for backend in Backend:
                index=FoldIndex(residual,fold,b,k,backend=backend)
                for policy in Policy:
                    for q,sign in ((alpha,-1),(1-alpha,1)):
                        with self.subTest(case=case,backend=backend,policy=policy,sign=sign):
                            expected=reference(residual,fold,b,centers,factors,t,q,sign,policy)
                            result=index.endpoint(centers,factors,t,q,sign,policy=policy)
                            self.assertEqual(result.value,expected)

    def test_default_choice_and_input_ownership(self):
        residual=np.array([.2,.7,.4]); fold=np.array([0,1,0]); mass=np.ones(3)
        index=FoldIndex(residual,fold,mass,2)
        before=index.interval([0.,1.],[1.,2.],.2,policy=Policy.CALIBRATE_GE)
        residual[:]=20.;mass[:]=100.;fold[:]=1
        after=index.interval([0.,1.],[1.,2.],.2,policy=Policy.CALIBRATE_GE)
        self.assertEqual(index.backend,Backend.VALUE)
        self.assertEqual(before,after)
        with self.assertRaises(ValueError):FoldIndex([1],np.array([0]),[1],1,backend='unknown')

    def test_finite_pool_fit_selects_backend(self):
        from sklearn.linear_model import Ridge
        rng=np.random.default_rng(20261016)
        pool=rng.normal(size=(300,4)); ids=np.arange(60)
        labels=pool[ids]@np.array([1.,-.2,.4,.7])+rng.normal(scale=.2,size=60)
        value=UniformFinitePoolFCS.fit(pool,ids,labels,Ridge(),nfolds=3,seed=17)
        rank=UniformFinitePoolFCS.fit(pool,ids,labels,Ridge(),nfolds=3,seed=17,backend='rank')
        self.assertEqual(value.backend,Backend.VALUE);self.assertEqual(rank.backend,Backend.RANK)
        queries=value.sample_query_indices(240,seed=42)
        a=value.predict_indices(queries);b=rank.predict_indices(queries)
        np.testing.assert_array_equal(a.bounds,b.bounds)
        np.testing.assert_array_equal(a.upper_below(0),b.upper_below(0))
        self.assertEqual(value.provenance['backend'],'value')

if __name__=='__main__':unittest.main(verbosity=2)
