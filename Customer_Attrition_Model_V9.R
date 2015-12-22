#By Emilio Esposito

library(plyr)
library(ggplot2)
library(sqldf)
library(ROSE)
library(DataCombine)
library(zoo)
source("Functions_CrossValidation_etc_V1.R")

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
#Aggregate Data into Monthly records (still with irregular spacing)
#############################
#############################

ecard$period <- as.yearmon(ecard$date)
ecard_m <- sqldf("
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



############################
############################
#feature selection
############################
############################



ecard_model_full <- ecard_m[, c("id",
                           "gender",
                           "pages",
                           "onsite",
                           "entered",
                           "completed",
                           "days_open",
                           "avg_days_btwn_login",
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
keep

#get full glm
ecard_m_glm <- glm(cancel ~ . ,data = ecard_model_full[-1])

#perform backward stepwise regression
backstep <- stepAIC(object = ecard_m_glm, direction = "backward")


library(MASS)

filter.features.by.cor(ecard_model_full)

#.874 num
ecard_model <- ecard_m[, c("id","period",
                           "gender",
                           "entered",
                           "pages",
                           "days_open",
                           "holiday",
                           "cancel")]

                                    
############################
############################
#10 fold Cross Validation with different models
############################
############################

logreg <- do_cv_user(ecard_model, 10, "logreg")
knn <- do_cv_user(ecard_model, 10, "5nn")
tree <- do_cv_user(ecard_model, 10, "tree")
nb <- do_cv_user(ecard_model, 10, "nb")
default <- do_cv_user(ecard_model, 10, "default")

#make ROC curves of CV results
roc.curve(response = logreg$truth, predicted = logreg$pred,main="Attrition Model Performance curves", col="red", cex.lab=1.2, cex.axis=1.2)
roc.curve(response = knn$truth, predicted = knn$pred,main="Attrition Model Performance curves", col="blue",add.roc = T, cex.lab=1.2, cex.axis=1.2)
roc.curve(response = tree$truth, predicted = tree$pred,main="Attrition Model Performance curves", col="green",add.roc = T, cex.lab=1.2, cex.axis=1.2)
roc.curve(response = nb$truth, predicted = nb$pred,main="Attrition Model Performance curves", col="orange",add.roc = T, cex.lab=1.2, cex.axis=1.2)
test <- roc.curve(response = default$truth, predicted = default$pred,main="Attrition Model Performance curves", col="black",add.roc = T, cex.lab=1.2, cex.axis=1.2)
legend("right", c("LogReg","Knn","DecTree","NaiveBayes", "Default"), col=c("red","blue","green","orange","black"), lwd=5)

#get metrics for logreg model
get_metrics(logreg,.05)


#output results to csv
ecard_m_ActualvPred <- merge(ecard_m, logreg[,c("id","period","pred")], by.x=c("id","period"),by.y=c("id","period"))
ecard_m_ActualvPred$pred_rounded.05<-aaply(.data = ecard_m_ActualvPred[,"pred"], .margins = 1,.fun = function(x) if(x>=.05) 1 else 0 )
small_output <- format(ecard_m_ActualvPred, digits=2, nsmall=2)
write.csv(small_output,"Attrition_Model_Monthly_Final_Output_10FoldCV_Logreg.csv")

############################
############################
#Out of Sample & Out of Time Validation
############################
############################

#train the model on first 2 years of data

#subset the df
ecard_model_final <- ecard_m[, c(
  "gender",
  "entered",
  "pages",
  "days_open",
  "holiday",
  "cancel")]

#randomly pick users for training
set.seed(2)
users.avail <- c(1:10000)
train_users <- sample(users.avail,size=5000,replace=FALSE)
test_users <- users.avail[!users.avail %in% train_users]

#assign train and test users
train <- ecard_model_final[which(ecard_model$id %in% train_users),,drop=FALSE]
test <- ecard_model_final[which(ecard_model$id %in% test_users),,drop=FALSE]

#perform out of time tests
logreg_oot <- as.data.frame(get_pred_logreg(train,test))
default_oot <- as.data.frame(get_pred_default(train,test))

#plot ROC curve
roc.curve(response = logreg_oot$truth, predicted = logreg_oot$pred,main="Attrition Model Performance Curve", col="red")

#get confusion matrix of models
get_metrics(logreg_oot, .05)
get_metrics(default_oot, .05)

#flip train and test
logreg_oot <- as.data.frame(get_pred_logreg(test,train))
get_metrics(logreg_oot, .05)

############################
############################
#Final Model Objects
############################
############################

#train on ALL the data
ecard_glm <- glm(cancel~.,data = ecard_model_final, family = binomial)
summary(ecard_glm)


############################
############################
#find rank ordered users who will cancel
############################
############################

last_user_month <- sqldf("select
id,
max(period) as period
from ecard_m
group by id
order by id")

last_user_month$period <- as.yearmon(last_user_month$period)

last_act_user_record <- join(last_user_month, ecard_m, by=c("id","period"))

last_act_user_record <- last_act_user_record[which(last_act_user_record$cancel==0),,drop=F]
pred <- predict(ecard_glm, last_act_user_record, type = "response")
rank <- data.frame(id=last_act_user_record$id, pred=pred)
rank <- rank[order(rank$pred,decreasing = T),]
rank <- rank[which(rank$pred>=.05),]
nrow(rank)
View(rank)

