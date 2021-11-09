set.seed(1234120101)
N<-1000
n<-50
th<-rnorm(N)
diff<-rnorm(n)

resp<-expand.grid(id=1:N,itemkey=1:n)
resp$th<-th[resp$id]
resp$diff<-diff[resp$itemkey]
resp$sequence_number<-sample(1:n,N*n,replace=TRUE)
resp$diff<-ifelse(resp$sequence_number>40,resp$diff+.1,resp$diff)

p<-1/(1+exp(-1*(resp$th-resp$diff)))
resp$resp<-rbinom(N*n,1,p)


## id - a unique id associatd with the respondent
## itemkey - a unique id associated with each item
## sequence_number - the position (1-indexed) that the respondent encounters the item in
## resp - a dichotomously coded response indicator, coded 0 if incorrect and 1 if correct
x<-resp[,c("id","itemkey","sequence_number","resp")]
write.csv(x, file='data2.csv', row.names=FALSE)

