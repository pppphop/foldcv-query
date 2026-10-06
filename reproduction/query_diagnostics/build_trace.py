"""Counter-only source instrumentation; never modify production code."""
from pathlib import Path
import argparse
import subprocess
ROOT = Path(__file__).resolve().parent

def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f'Unexpected instrumentation site: {old!r}')
    return text.replace(old, new)

def main(cxx):
    path=ROOT/'software/src/foldcv_query/weighted.cpp'
    if not path.exists():path=ROOT.parent/'software_versions/foldcv_query_0_2_3/src/foldcv_query/weighted.cpp'
    source = path.read_text(encoding='utf-8')
    source = source.replace('struct Node {', 'struct Trace {int iterations,reason;double pivot;int64_t ties;};\nstatic thread_local Trace trace={0,0,0.,0};\nstruct Node {',1)
    start=source.index('static int weighted_index_impl(')
    end=source.index('// Keep the original rank-search ABI',start)
    pre, body, post=source[:start],source[start:end],source[end:]
    body=replace(body,'auto fail=[&](){','auto fail=[&](int reason){trace.reason=reason;')
    body=replace(body,'if(!(t>=minw&&t<=maxw))return fail();','if(!(t>=minw&&t<=maxw))return fail(1);')
    body=replace(body,'if(!(a[j]>=minw&&a[j]<=maxw)||!std::isfinite(c[j]))return fail();','if(!(a[j]>=minw&&a[j]<=maxw)||!std::isfinite(c[j]))return fail(1);')
    body=replace(body,'if(len&&(!std::isfinite(endpoint(r,off,c,j,0,sign))||!std::isfinite(endpoint(r,off,c,j,len-1,sign))))return fail();','if(len&&(!std::isfinite(endpoint(r,off,c,j,0,sign))||!std::isfinite(endpoint(r,off,c,j,len-1,sign))))return fail(2);')
    body=replace(body,'if(std::abs(q-inf_cdf)<=eps)return fail();','if(std::abs(q-inf_cdf)<=eps)return fail(3);')
    body=replace(body,'for(int iter=0;iter<80;++iter){','for(int iter=0;iter<80;++iter){trace.iterations=iter+1;')
    body=replace(body,'if(!std::isfinite(lower)||!std::isfinite(upper))return fail();','if(!std::isfinite(lower)||!std::isfinite(upper))return fail(4);')
    body=replace(body,'if(mids.empty())return fail();','if(mids.empty())return fail(4);')
    body=replace(body,'double before=mass0/total,atmost=mass1/total;','trace.pivot=p;trace.ties=multiplicity;\n        double before=mass0/total,atmost=mass1/total;')
    body=replace(body,'if(p==0.)return fail();','if(p==0.)return fail(5);')
    body=replace(body,'if(multiplicity!=1)return fail();','if(multiplicity!=1)return fail(6);')
    body=replace(body,'if(previous==0.)return fail();','if(previous==0.)return fail(7);')
    body=replace(body,'}else return fail();','}else return fail(8);')
    body=replace(body,'    return fail();','    return fail(9);')
    assert 'return fail();' not in body
    source=pre+body+post+r'''
API int diagnostic_index(const double*r,const int64_t*off,const double*pf,const double*pr,
 const double*c,const double*a,double t,int k,double eps,double q,int sign,int value_search,
 double*out,int*status,Trace*record){
 trace={0,0,0.,0};
 int rc=weighted_index_impl(r,off,pf,pr,c,a,t,k,eps,q,sign,out,status,value_search!=0);
 *record=trace;return rc;
}
'''
    target=ROOT/'trace.cpp';target.write_text(source,encoding='utf-8')
    command=[cxx,'-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-shared','-static','-static-libgcc','-static-libstdc++',str(target),'-o',str(ROOT/'trace.dll')]
    subprocess.run(command,check=True)
    print('Instrumented diagnostic library built; production source unchanged.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cxx',required=True)
    main(parser.parse_args().cxx)
