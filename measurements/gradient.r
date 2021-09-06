xs <- seq(1,9)
eq = function(x){x^2-10*x+25}
vanish<- c(0,0.4,0.6,0.7,0.75)
exploding<- c(2,8,2.5,8.5)


par(bg=NA)
curve(expr = eq,from = 0,to = 10,xlab = "x",ylab = "Error",xaxt='n',yaxt='n')
lines(x = vanish,eq(vanish),col="orange" )
points(x = vanish,eq(vanish),col="red" )

lines(x = exploding,eq(exploding),col="blue" )
points(x = exploding,eq(exploding),col="blue" )

legend(7.5, 5, legend=c("vanish","explode"),title = "gradient slope",
       col=c("red","blue"), lty=1)