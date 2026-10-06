args <- commandArgs(trailingOnly=TRUE)
repo <- args[1]; folder <- args[2]
source(file.path(repo,'R/utils.R'))
source(file.path(repo,'R/conformal_utils.R'))
source(file.path(repo,'R/conformal_CV.R'))
cal <- as.matrix(read.csv(file.path(folder,'calibration.csv')))
centers <- as.matrix(read.csv(file.path(folder,'centers.csv')))
testmass <- scan(file.path(folder,'testmass.txt'),quiet=TRUE)
k <- ncol(centers)
make_object <- function() {
    info <- lapply(seq_len(k),function(j) {
        g <- which(cal[,2]==j-1)
        list(Yscore=cal[g,1],wt=cal[g,3],
             Ymodel=local({jj<-j; function(X) centers[X[,1],jj]}))
    })
    object <- list(info=info,wtfun=function(X) testmass[X[,1]],
                   type='mean',side='two',nfolds=k)
    class(object)<-'conformalCV'
    object
}
rows<-list(); ix<-0
for (q in c(32,512)) {
    for (rep in 1:3) {
        start<-proc.time()[['elapsed']]; obj<-make_object()
        prep<-proc.time()[['elapsed']]-start
        Xtest<-matrix(seq_len(q),ncol=1)
        start<-proc.time()[['elapsed']]
        result<-predict(obj,Xtest,alpha=.1,wthigh=Inf,wtlow=0,useInf=TRUE)
        elapsed<-proc.time()[['elapsed']]-start
        write.csv(result,file.path(folder,paste0('cfcausal_q',q,'_rep',rep,'.csv')),row.names=FALSE)
        ix<-ix+1; rows[[ix]]<-data.frame(q=q,rep=rep,prepare_seconds=prep,query_seconds=elapsed,
                                        saved_object_bytes=as.numeric(object.size(obj)))
    }
}
write.csv(do.call(rbind,rows),file.path(folder,'cfcausal_times.csv'),row.names=FALSE)
writeLines(c(R.version.string,'Original author prediction source, supplied saved state; no fitting'),
           file.path(folder,'R_version.txt'))
