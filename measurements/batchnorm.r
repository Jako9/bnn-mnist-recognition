data_bn = read.csv("accuracy_with_bn.csv", header = FALSE,check.names=FALSE)
data_no_bn = read.csv("accuracy_without_bn.csv", header = FALSE,check.names=FALSE)
xs <- seq(1,51)
print(max(data_bn)-max(data_no_bn))

plot(x = xs,y = data_bn,type="l",col="red", ylim=c(79,90),xlab = "Epoch",ylab = "Accuracy (%)")
lines(x = xs,y = data_no_bn,col="blue")
text(locator(), labels = c("with BN layer", "without BN layer"),col="black")
