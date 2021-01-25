set.seed(1234120101)
N<-1000
n<-50
th<-rnorm(N)
diff<-rnorm(n)

th2<-matrix(th,nrow=N,ncol=n,byrow=FALSE)
diff2<-matrix(diff,nrow=N,ncol=n,byrow=TRUE)

del<-th2-diff2
p<-1/(1+exp(-del))
test<-matrix(runif(N*n),nrow=N,ncol=n)
resp<-ifelse(p>test,1,0)

## id - a unique id associatd with the respondent
## itemkey - a unique id associated with each item
## sequence_number - the position (1-indexed) that the respondent encounters the item in
## resp - a dichotomously coded response indicator, coded 0 if incorrect and 1 if correct

L<-list()
for (j in 1:ncol(resp)) {
    L[[j]]<-data.frame(id=1:N,itemkey=j,sequence_number=sample(1:n,N,replace=TRUE),resp=resp[,j])
}
x<-data.frame(do.call("rbind",L))

write.csv(x, file='data.csv', row.names=FALSE)

