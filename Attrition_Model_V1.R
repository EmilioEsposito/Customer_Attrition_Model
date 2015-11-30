library(plyr)
library(ggplot2)
library(sqldf)
library(ROSE)
library(tscount)
library(DataCombine)
library(zoo)
source("CV_functions_V1.R")

# set working directory
setwd(".")
getwd()
# read in ecard data
ecard <- read.csv("ltv.csv", sep = ",", header = TRUE)

############################
############################
#Data prep
############################
############################

# make binary column for indep var, cancel
## 1==cancel event
## 0==not cancelled
ecard$cancel <- mapvalues(ecard$status, c(0,1,2), c(0,0,1))

# convert date column to date format
ecard$date <- as.Date(ecard$date , "%Y-%m-%d")

# get opening_date of each cust
opening_date <- sqldf("
                      select 
                      id, 
                      date as opening_date
                      from ecard
                      where status=0
                      group by id 
                      ")

# convert date column to date format
opening_date$opening_date <- as.Date(opening_date$opening_date)

#merge opening date field onto ecard
ecard <- merge(ecard, opening_date, by.x = "id", by.y = "id")

#calc days since account opening
ecard$days_open <- ecard$date - ecard$opening_date + 1
ecard$days_open <- as.numeric(ecard$days_open)

#ensure sorted by id and date
ecard <- ecard[with(ecard, order(id, date)), ]

#calc days inactive
ecard <- slide(ecard, Var = "days_open", GroupVar = "id", slideBy = -1)
colnames(ecard)[ncol(ecard)] <- "prior_days_open"

ecard$days_btwn_login <- ecard$days_open - ecard$prior_days_open -1
ecard$days_btwn_login <- as.numeric(ecard$days_btwn_login)

#get rid of -1s
ecard$days_btwn_login <-mapvalues(ecard$days_btwn_login,c(-1,NA),c(0,0))
ecard$prior_days_open <-mapvalues(ecard$prior_days_open,c(-1,NA),c(0,0))


#############################
#############################
#Quarterly with Uneven spacing
#############################
#############################

ecard$period <- as.yearqtr(ecard$date)
ecard_q <- sqldf("
                 select 
                 id,
                 gender,
                 period,
                 sum(pages) as pages,
                 sum(onsite) as onsite,
                 sum(entered) as entered,
                 sum(completed) as completed,
                 sum(holiday) as holiday,
                 max(days_open) as days_open,
                 avg(days_btwn_login) as avg_days_btwn_login,
                 count(id) as num_trns,
                 sum(cancel) as cancel
                 from ecard
                 group by id, gender, period
                 order by id, period
                 ")


#lagged vars
lg <- function(x) c(x[1], x[1:(length(x)-1)])

ecard_q <- ddply(ecard_q, .(id), .fun = transform, pages_scale=scale(pages))
ecard_q <- ddply(ecard_q, .(id), .fun = transform, onsite_scale=scale(onsite))
ecard_q <- ddply(ecard_q, .(id), .fun = transform, entered_scale=scale(entered))
ecard_q <- ddply(ecard_q, .(id), .fun = transform, completed_scale=scale(completed))
ecard_q <- ddply(ecard_q, .(id), .fun = transform, holiday_scale=scale(holiday))
ecard_q <- ddply(ecard_q, .(id), .fun = transform, avg_days_btwn_login_scale=scale(avg_days_btwn_login))
ecard_q <- ddply(ecard_q, .(id), .fun = transform, days_open_scale=scale(days_open))


ecard_q$pages_change <- ave(ecard_q$pages_scale, ecard_q$id, FUN = function(x) x-lg(x))
ecard_q$onsite_change <- ave(ecard_q$onsite_scale, ecard_q$id, FUN = function(x) x-lg(x))
ecard_q$entered_change <- ave(ecard_q$entered_scale, ecard_q$id, FUN = function(x) x-lg(x))
ecard_q$completed_change <- ave(ecard_q$completed_scale, ecard_q$id, FUN = function(x) x-lg(x))
ecard_q$holiday_change <- ave(ecard_q$holiday_scale, ecard_q$id, FUN = function(x) x-lg(x))
ecard_q$avg_days_btwn_login_change <- ave(ecard_q$avg_days_btwn_login_scale, ecard_q$id, FUN = function(x) x-lg(x))

#replace 0s
ecard_q[is.na(ecard_q)] <- 0


write.csv(ecard_q,"ecard_q.csv", row.names = FALSE)

############################
############################
#feature selection
############################
############################



ecard_model_full <- ecard_q[, c("id",
                           "gender",
                           "pages_scale",
                           "onsite_scale",
                           "entered_scale",
                           "completed_scale", 
                           "holiday_change",
                           "pages_change",
                           "onsite_change",
                           "entered_change",
                           "completed_change",
                           "holiday_change",
                           "avg_days_btwn_login_change",
                           "days_open",
                           "days_open_scale",
                           "avg_days_btwn_login",
                           "avg_days_btwn_login_scale",
                           "num_trns",
                           "cancel")]

library(leaps)

#find subsets wth leaps package
subs <- regsubsets(cancel~., data=ecard_model_full[-1], 
                   nvmax = 8, nbest=1, 
                   method  ='exhaustive',
                   force.in="days_open")
#get summary
subs_summ <- summary(subs)
#find best 5 subsets
best <- subs_summ$which
keep <- t(best)
keep[which(keep[,8]),8,drop=F]
keep

library(MASS)

#get full lm of bmi
ecard_q_glm <- glm(cancel ~ . ,data = ecard_model_full[-1])

#perform backward stepwise regression
backstep <- stepAIC(object = ecard_q_glm, direction = "forward")


filter.features.by.cor(ecard_model_full)


ecard_model <- ecard_q[, c("id","period",
                                "gender",
                                "entered_scale",
                                "completed_scale",
                                "days_open_scale",
                                "avg_days_btwn_login_change",
                                "cancel")]

############################
############################
#Cross Validation
############################
############################

knnn <- do_cv_user(ecard_model, 10, "50nn")
logreg <- do_cv_user(ecard_model, 10, "logreg")
nb <- do_cv_user(ecard_model, 10, "nb")
tree<- do_cv_user(ecard_model, 10, "tree")
#randomForest<- do_cv_user(ecard_model, 10, "randomForest")
default <- do_cv_user(ecard_model, 10, "default")
write.csv(logreg,"logreg.csv")


roc.curve(knnn$truth, knnn$pred)
roc.curve(logreg$truth, logreg$pred)
roc.curve(nb$truth, nb$pred)
roc.curve(tree$truth, tree$pred)
#roc.curve(randomForest$truth, randomForest$pred)
roc.curve(default$truth, default$pred)

get_metrics(knnn,.08)
get_metrics(logreg,.08)
get_metrics(nb,.08)
get_metrics(tree,.08)


############################
############################
#Out of Sample & Out of Time Validation
############################
############################

#train the model on first 2 years of data

ecard_model_final <- ecard_model[, c("gender",
                                "entered_scale",
                                    "completed_scale",
                                    "days_open_scale",
                                    "avg_days_btwn_login_change",
                                    "cancel")]
train <- ecard_model_final[which(ecard_model$period<="2012 Q4" & (ecard_model$id %in% c(1:5000))),,drop=FALSE]
test <- ecard_model_final[which(ecard_model$period>"2012 Q4" & (ecard_model$id %in% c(5001:10000))),,drop=FALSE]

#all models
logreg_oot <- as.data.frame(get_pred_logreg(train,test))
knn_oot <- as.data.frame(get_pred_knn(train,test,10))
nb_oot <- as.data.frame(get_pred_nb(train,test))
tree_oot <- as.data.frame(get_pred_tree(train,test))
default_oot <- as.data.frame(get_pred_default(train,test))

roc.curve(response = logreg_oot$truth, predicted = logreg_oot$pred,main="Attrition Model ROC curves", col="red")
roc.curve(response = knn_oot$truth, predicted = knn_oot$pred,main="Attrition ROC curves", add.roc = TRUE, col="blue")
roc.curve(response = nb_oot$truth, predicted = nb_oot$pred,main="Attrition ROC curves", add.roc = TRUE, col="green")
roc.curve(response = tree_oot$truth, predicted = tree_oot$pred,main="Attrition ROC curves", add.roc = TRUE, col="orange")
legend("right", c("LogReg","Knn","NaiveBayes","DecTree"), col=c("red","blue","green","orange"), lwd=10)

get_metrics(logreg_oot, .09)
get_metrics(knn_oot, .09)
get_metrics(nb_oot, .09)
get_metrics(tree_oot, .09)
get_metrics(default_oot, .09)


############################
############################
#Model Objects (in progress)
############################
############################

ecard_glm <- glm(cancel~.,data = train, family = binomial)
summary(ecard_glm)
pred <- predict(ecard_glm, newdata = test, type = "response")

#test the model on the last 2 years of data
truth <- test[,"cancel"]
test$pred<-pred

write.csv(logreg,"test.csv")
roc.curve(response = truth, predicted = pred)

pred_truth <- data.frame(pred=pred, truth=truth)
head(pred_truth)

get_metrics(pred_truth,.05)


#tree time cv

tree_oot <- as.data.frame(get_pred_tree(train,test))
roc.curve(tree_oot$truth, tree_oot$pred)

get_metrics(tree_oot,.02)




