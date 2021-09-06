bs_10 = read.csv("batchsize_10.csv", header = FALSE,check.names=FALSE)
bs_50 = read.csv("batchsize_50.csv", header = FALSE,check.names=FALSE)
bs_100 = read.csv("batchsize_100.csv", header = FALSE,check.names=FALSE)
bs_150 = read.csv("batchsize_150.csv", header = FALSE,check.names=FALSE)
bs_200 = read.csv("batchsize_200.csv", header = FALSE,check.names=FALSE)
bs_500 = read.csv("batchsize_500.csv", header = FALSE,check.names=FALSE)
bs_1000 = read.csv("batchsize_1000.csv", header = FALSE,check.names=FALSE)
xs <- seq(1,21)
print(data_bn)

plot(x = xs,y = bs_100,type="l",col="orange",ylim=c(83,88.2),xlab = "epoch",ylab = "accuracy (%)")
lines(x = xs,y = bs_10,col="grey")
lines(x = xs,y = bs_50,col="darkviolet")
lines(x = xs,y = bs_150,col="red")
lines(x = xs,y = bs_200,col="blue")
lines(x = xs,y = bs_500,col="black")
lines(x = xs,y = bs_1000,col="chartreuse4")
#text(locator(), labels = c("BNN mit BN", "BNN ohne BN"),col="black")
legend(13, 85.5, legend=c("10","50","100", "150","200","500","1000"),title = "Batchsize",
       col=c("grey","darkviolet","orange", "red","blue","black","chartreuse4"), lty=1)