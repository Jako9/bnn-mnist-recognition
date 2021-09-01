data_bn = read.csv("accuracy_with_bn.csv", header = FALSE,check.names=FALSE)
data_no_bn = read.csv("accuracy_without_bn.csv", header = FALSE,check.names=FALSE)
xs <- seq(1,51)
print(data)

plot(x = xs,y = data_bn,type="l",col="red", ylim=c(79,90),xlab = "Epoche",ylab = "Genauigkeit")
lines(x = xs,y = data_no_bn,col="blue")
text(locator(), labels = c("BNN mit BN", "BNN ohne BN"),col="black")
