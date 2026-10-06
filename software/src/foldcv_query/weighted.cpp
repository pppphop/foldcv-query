#include <cfenv>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
struct Node {double v,w;int64_t row;};
static double exact_sort(std::vector<Node> items,double q,int policy){
    std::sort(items.begin(),items.end(),[](const Node&a,const Node&b){
        if(a.v!=b.v)return a.v<b.v;
        if(a.w!=b.w)return a.w<b.w;
        return a.row<b.row;
    });
    double cdf=0.;
    for(size_t i=0;i<items.size();++i){
        cdf+=items[i].w;
        if(policy ? cdf>q : cdf>=q) return q>.5||i==0||(!policy&&cdf==q)?items[i].v:items[i-1].v;
    }
    return std::nan("");
}
static double selected(std::vector<Node> items,double q,int policy,int*status){
    const double u=std::ldexp(1.,-53),m=4.*items.size()+5.;
    const double eps=std::nextafter(128.*m*u/(1.-m*u),INFINITY);
    size_t lo=0,hi=items.size();double base=0.;
    auto fail=[&](){*status=1;return exact_sort(std::move(items),q,policy);};
    for(int iter=0;iter<80&&lo<hi;++iter){
        size_t mid=lo+(hi-lo)/2;
        std::nth_element(items.begin()+lo,items.begin()+mid,items.begin()+hi,[](const Node&a,const Node&b){return a.v<b.v;});
        double pivot=items[mid].v,below=0.,equal=0.;int64_t ties=0;
        for(size_t i=lo;i<hi;++i){if(items[i].v<pivot)below+=items[i].w;else if(items[i].v==pivot){equal+=items[i].w;++ties;}}
        double before=base+below,atmost=before+equal;
        if(q<before-eps){
            hi=std::partition(items.begin()+lo,items.begin()+hi,[&](const Node&n){return n.v<pivot;})-items.begin();
        }else if(q>atmost+eps){
            lo=std::partition(items.begin()+lo,items.begin()+hi,[&](const Node&n){return n.v<=pivot;})-items.begin();base=atmost;
        }else if(q>before+eps&&q<atmost-eps){
            if(pivot==0.)return fail();
            if(q>.5){*status=0;return pivot;}
            if(ties!=1)return fail();
            double previous=-INFINITY;
            for(const auto& n:items)if(n.v<pivot&&n.v>previous)previous=n.v;
            if(previous==0.)return fail();
            *status=0;return previous;
        }else return fail();
    }
    return fail();
}
API int weighted_flat(const double*v,const double*w,int64_t n,double q,int mode,int policy,double*out,int*status){
    std::vector<Node> items;items.reserve(n);
    for(int64_t i=0;i<n;++i)items.push_back({v[i],w[i],i});
    *status=0;*out=mode?selected(std::move(items),q,policy,status):exact_sort(std::move(items),q,policy);
    return 0;
}
static double endpoint(const double*r,const int64_t*off,const double*c,int j,int64_t i,int sign){
    return sign>0?c[j]+r[off[j]+i]:c[j]-r[off[j+1]-1-i];
}
static int weighted_index_impl(const double*r,const int64_t*off,const double*pf,const double*pr,
    const double*c,const double*a,double t,int k,double eps,double q,int sign,double*out,int*status,bool value_search){
    const double*prefix=sign>0?pf:pr;
    const double minw=std::ldexp(1.,-100),maxw=std::ldexp(1.,100);
    auto fail=[&](){*status=1;*out=std::nan("");return 0;};
    if(!(t>=minw&&t<=maxw))return fail();
    double cal=0.;
    for(int j=0;j<k;++j){
        int64_t len=off[j+1]-off[j],poff=off[j]+j;
        if(!(a[j]>=minw&&a[j]<=maxw)||!std::isfinite(c[j]))return fail();
        if(len&&(!std::isfinite(endpoint(r,off,c,j,0,sign))||!std::isfinite(endpoint(r,off,c,j,len-1,sign))))return fail();
        cal+=a[j]*prefix[poff+len];
    }
    double total=cal+t,inf_cdf=sign<0?t/total:cal/total;
    if(sign<0&&q<inf_cdf-eps){*out=-INFINITY;*status=0;return 0;}
    if(sign>0&&q>inf_cdf+eps){*out=INFINITY;*status=0;return 0;}
    if(std::abs(q-inf_cdf)<=eps)return fail();
    std::vector<int64_t> lo(k,0),hi(k),lb(k),ub(k);
    for(int j=0;j<k;++j)hi[j]=off[j+1]-off[j];
    auto bound=[&](int j,double p,bool upper){
        int64_t left=0,right=off[j+1]-off[j];
        while(left<right){int64_t mid=left+(right-left)/2;double v=endpoint(r,off,c,j,mid,sign);
            if(v<p||(upper&&v==p))left=mid+1;else right=mid;}
        return left;
    };
    for(int iter=0;iter<80;++iter){
        double p;
        if(value_search){
            double lower=INFINITY,upper=-INFINITY;
            for(int j=0;j<k;++j)if(hi[j]>lo[j]){
                lower=std::min(lower,endpoint(r,off,c,j,lo[j],sign));
                upper=std::max(upper,endpoint(r,off,c,j,hi[j]-1,sign));
            }
            if(!std::isfinite(lower)||!std::isfinite(upper))return fail();
            p=lower+(upper-lower)*.5;
            if(!std::isfinite(p))p=lower*.5+upper*.5;
        }else{
            std::vector<std::pair<double,int64_t>> mids;int64_t active=0;
            for(int j=0;j<k;++j)if(hi[j]>lo[j]){int64_t len=hi[j]-lo[j];active+=len;mids.push_back({endpoint(r,off,c,j,lo[j]+len/2,sign),len});}
            if(mids.empty())return fail();
            std::sort(mids.begin(),mids.end());int64_t running=0;p=mids.back().first;
            for(const auto& mid:mids){running+=mid.second;if(running>=(active+1)/2){p=mid.first;break;}}
        }
        double mass0=0.,mass1=0.;int64_t multiplicity=0;
        for(int j=0;j<k;++j){lb[j]=bound(j,p,false);ub[j]=bound(j,p,true);multiplicity+=ub[j]-lb[j];
            int64_t poff=off[j]+j;mass0+=a[j]*prefix[poff+lb[j]];mass1+=a[j]*prefix[poff+ub[j]];}
        if(sign<0){mass0+=t;mass1+=t;}
        double before=mass0/total,atmost=mass1/total;
        if(q<before-eps){for(int j=0;j<k;++j)hi[j]=std::min(hi[j],lb[j]);}
        else if(q>atmost+eps){for(int j=0;j<k;++j)lo[j]=std::max(lo[j],ub[j]);}
        else if(q>before+eps&&q<atmost-eps){
            if(p==0.)return fail();
            if(q>.5){*out=p;*status=0;return 0;}
            if(multiplicity!=1)return fail();
            double previous=-INFINITY;
            for(int j=0;j<k;++j)if(lb[j]>0)previous=std::max(previous,endpoint(r,off,c,j,lb[j]-1,sign));
            if(previous==0.)return fail();
            *out=previous;*status=0;return 0;
        }else return fail();
    }
    return fail();
}

// Keep the original rank-search ABI for existing native callers.
API int weighted_index(const double*r,const int64_t*off,const double*pf,const double*pr,
    const double*c,const double*a,double t,int k,double eps,double q,int sign,double*out,int*status){
    return weighted_index_impl(r,off,pf,pr,c,a,t,k,eps,q,sign,out,status,false);
}
API int weighted_value_index(const double*r,const int64_t*off,const double*pf,const double*pr,
    const double*c,const double*a,double t,int k,double eps,double q,int sign,double*out,int*status){
    return weighted_index_impl(r,off,pf,pr,c,a,t,k,eps,q,sign,out,status,true);
}

// Query-major Q x K inputs and Q x 2 outputs; scalar ABI remains available.
API int weighted_interval_batch(const double*r,const int64_t*off,const double*pf,const double*pr,
    const double*c,const double*a,const double*t,int64_t rows,int k,double eps,double alpha,
    int value_search,double*out,int*status){
    for(int64_t i=0;i<rows;++i){
        int rc=weighted_index_impl(r,off,pf,pr,c+i*k,a+i*k,t[i],k,eps,alpha,-1,
            out+2*i,status+2*i,value_search!=0);
        if(rc)return rc;
        rc=weighted_index_impl(r,off,pf,pr,c+i*k,a+i*k,t[i],k,eps,1.-alpha,1,
            out+2*i+1,status+2*i+1,value_search!=0);
        if(rc)return rc;
    }
    return 0;
}

API int weighted_environment(){
    volatile double x=std::numeric_limits<double>::denorm_min(),one=1.;
    return std::numeric_limits<double>::is_iec559 && sizeof(double)==8 && std::fegetround()==FE_TONEAREST && x*one!=0.;
}
