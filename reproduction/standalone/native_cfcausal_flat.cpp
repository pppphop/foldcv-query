#include "software/src/foldcv_query/weighted.cpp"
API int cfcausal_flat(const double*r,const int64_t*f,const double*b,int64_t n,
 const double*c,const double*a,double t,double q,int sign,double*out,int*status){
    std::vector<Node> items;items.reserve(n+1);double total=0.;
    for(int64_t i=0;i<n;++i){double w=a[f[i]]*b[i];total+=w;
        items.push_back({c[f[i]]+sign*r[i],w,i});}
    total+=t;items.push_back({INFINITY,t,n});
    for(auto& v:items)v.w/=total;
    *status=0;*out=selected(std::move(items),q,1,status);return 0;
}
