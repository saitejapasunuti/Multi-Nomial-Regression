############# MULTINOMIAL REGRESSION ###########################

install.packages('mlogit')
?mlogit
#mlogit is a package which enables the estimations of the multinomial logit models with individual or alternative specific variables
require(mlogit)

install.packages("nnet")
?nnet
#a neural network classifier which is used to see the accuracy of the model and make predictions on input data  
require(nnet)

#load dataset
mdata <- read.csv(file.choose())
View(mdata)

#remove x and id variables
mdata1 <-subset(mdata,select = -c(1,2)) 

head(mdata1)#gives first 6 rows

tail(mdata1)#gives last 6 rows
View(mdata1)
table(mdata$prog)#tabular representation of y categories
#academic  general vocation 
#105       45       50 

table(mdata1$ses,mdata1$prog)
#         prog
#ses      academic general vocation
    #high         42       9        7
    #low          19      16       12
    #middle       44      20       3
#other way=>with(mdata1,table(ses,prog))

colnames(mdata1)
mdata1$prog <- relevel(mdata1$prog,ref = "academic")#selecting academic as baseline

model.prog <- multinom(prog~female+ses+schtyp+read+write+math+science+honors,data = mdata1)
summary(model.prog)

z <- summary(model.prog)$coefficients/summary(model.prog)$standard.errors
p_value <-(1-pnorm(abs(z),0,1))*2
p_value

summary(model.prog)$coefficients

#odds ratio
exp(coef(model.prog))

#predict probability
prob <- fitted(model.prog)
prob

#find the accuracy of the model
class(prob)
prob <- data.frame(prob)
View(prob)
prob["pred"] <- NULL

#custom func that returns the predicted value based on probability
get_names <- function(i){
  return(names(which.max(i)))
}

pred_name <- apply(prob,1,get_names)
?apply
prob$pred <- pred_name
View(prob)

#confusion matrix
table(pred_name,mdata1$prog)
#pred_name  academic general vocation
#academic       86      22       17
#general        11      10        4
#vocation        8      13       29

#confusion matrix visualization
barplot(table(pred_name,mdata1$prog),beside = T,col=c("red","green","blue"),legend=c("academic","general","vocation"),main = "Predicted(x-axis)-legend(actual)",ylab="count")

#accuracy
mean(pred_name==mdata1$prog)
#0.625