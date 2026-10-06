#include "software/src/foldcv_query/weighted.cpp"
API int fused_flat(const double*r,const int64_t*f,const double*b,int64_t n,
 const double*c,const double*a,double t,double q,int sign,double*out,int*status){
    std::vector<Node> items;items.reserve(n+1);double total=0.;
    for(int64_t i=0;i<n;++i){double w=a[f[i]]*b[i];total+=w;
        items.push_back({c[f[i]]+sign*r[i],w,i});}
    total+=t;items.push_back({sign<0?-INFINITY:INFINITY,t,n});
    for(auto& v:items)v.w/=total;
    *status=0;*out=selected(std::move(items),q,0,status);return 0;
}

API int fused_flat_batch(const double*r,const int64_t*f,const double*b,int64_t n,
    const double*c,const double*a,const double*t,int64_t rows,int k,double alpha,
    double*out,int*status){
    for(int64_t i=0;i<rows;++i){
        int rc=fused_flat(r,f,b,n,c+i*k,a+i*k,t[i],alpha,-1,out+2*i,status+2*i);
        if(rc)return rc;
        rc=fused_flat(r,f,b,n,c+i*k,a+i*k,t[i],1.-alpha,1,out+2*i+1,status+2*i+1);
        if(rc)return rc;
    }
    return 0;
}
