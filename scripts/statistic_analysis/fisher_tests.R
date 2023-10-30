setwd(".") #set local dir here
library(exact2x2)
library(effsize)
library(xtable)

res=list(JavaVersion=c(),Masking=c(),Fisher.p=c(),Fisher.OR=c())

javaversions=c(2,5,6,7,11,14,16,17)

maskings=c("token", "construct", "block")

for(j in javaversions)
{
    print(paste(j))
    namefile=paste("<folder_containing_the correct and wrong predictions>/","java","_",j,".csv",sep="")
    t1<-read.csv(namefile,na.strings = "None",sep=',',header=TRUE)
    for (i in 1:3)
    {
      m=array(c(t1$JavaX.OK[i],
                t1$JavaX.WRONG[i],
                t1$Java8.OK[i],
                t1$Java8.WRONG[i]),dim=c(2,2))
      
      res$JavaVersion=c(res$JavaVersion,as.character(j))
      res$Masking=c(res$Masking,as.character(maskings[i]))
      
      f=fisher.test(m)
      res$Fisher.p=c(res$Fisher.p,f$p.value)
      res$Fisher.OR=c(res$Fisher.OR,1/f$estimate)
      
      
  }
}

res=data.frame(res)
res2=res
#p-value adjustment
res2$Fisher.p=p.adjust(res2$Fisher.p,method="BH")

print(xtable(res2),include.rownames=FALSE)
