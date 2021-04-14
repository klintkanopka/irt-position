##note the division by 10 to set pi
##uniform change
set.seed(1234120101)
N<-1000
n<-50
th<-rnorm(N)
diff<-rnorm(n)

resp<-expand.grid(id=1:N,itemkey=1:n)
resp$th<-th[resp$id]
resp$diff<-diff[resp$itemkey]
resp$sequence_number<-sample(1:n,N*n,replace=TRUE)
#resp$diff.early<-resp$diff.late<-resp$diff
invlogit<-function(x) 1/(1+exp(-x))
invlogit<-Vectorize(invlogit)

resp$pi<-invlogit(((n/2)-resp$sequence_number)/10)
resp$diff.early<-resp$diff-.2
resp$diff.late<-resp$diff+.2

resp$p.early<-invlogit(resp$th-resp$diff.early)
resp$p.late<-invlogit(resp$th-resp$diff.late)
resp$p<-resp$pi*resp$p.early+(1-resp$pi)*resp$p.late

resp$resp<-rbinom(N*n,1,resp$p)
## id - a unique id associatd with the respondent
## itemkey - a unique id associated with each item
## sequence_number - the position (1-indexed) that the respondent encounters the item in
## resp - a dichotomously coded response indicator, coded 0 if incorrect and 1 if correct
x<-resp[,c("id","itemkey","sequence_number","resp")]
write.csv(x, file='data5.csv', row.names=FALSE)
