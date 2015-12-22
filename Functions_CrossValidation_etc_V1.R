#BY Emilio Esposito

library(FNN)
library(e1071)
library(plyr)
library(randomForest)

do_cv_user <- function(df, num_folds, model_name,  frm="ytrain~.") {
  
  #setseed
  set.seed(1) #guarantees same random order for each model
  
  #get number of columns
  nf <- ncol(df)
  
  #if output is factor, convert to numeric binary (0,1)
  if(is.factor(df[ , nf])) {
    df[ , nf] <- as.numeric(df[ , nf])
    
    ##first level will be mapped to 1, second level will be 0
    #df[ , nf] <- mapvalues(df[ , nf], c(1,2), c(1,0))
  }
  
  #make empty vector to store all pred_truth
  pred_truth_df <- data.frame()
  pred_truth_all <- data.frame()
  users.avail <- unique(df$id)
  #subset the df into test and train
  for(i in 1:num_folds) {
   
    rand.test.users <- sample(users.avail,size=min(length(unique(df$id))/num_folds,length(users.avail)),replace=FALSE)
    length(rand.test.users)
    users.avail <- users.avail[!users.avail %in% rand.test.users]
    length(users.avail)
    
    #subset the df
    train <- df[-which(df[,"id"] %in% rand.test.users), ,drop=FALSE]
    test <- df[which(df[,"id"] %in% rand.test.users), ,drop=FALSE]
    test_full <- test
    train_full <- test
    
    #drop id and period
    drop_index <- which(names(train)=="id")
    drop_index <- c(drop_index, which(names(train)=="period"))
    drop_index <- c(drop_index, which(names(train)=="date"))
    drop_index <- c(drop_index, which(names(train)=="quarter"))
    drop_index <- c(drop_index, which(names(train)=="month"))
    train <- train[-drop_index]
    test <- test[-drop_index]
    
    #determine model from model_name string
    if(grepl(pattern = "nn", x= model_name)) {
      k <- as.numeric(strsplit(model_name,"nn"))
      
      #run the model 
      pred_truth <- get_pred_knn(train,test,k)
    }
    else if(grepl(pattern = "glmer", x= model_name)) {
      #run the model 
      pred_truth <- get_pred_glmer(train,test,frm)
      message("glmer model")
    } else if(grepl(pattern = "frm", x= model_name)) {
      #run the model 
      pred_truth <- get_pred_log_regfrm(train,test,frm)
      message("glmer model")
    } else if(grepl(pattern = "cox", x= model_name)) {
      #run the model 
      pred_truth <- get_pred_cox(train_full, test_full)
      message("cox model")
    } else {
      #run the model 
      model <- get(paste("get_pred_", model_name, sep = ""), mode="function") 
      pred_truth <- model(train, test)
      message("running fold")
    }
    pred_truth_df <- cbind(test_full, pred_truth)
    #accumulate results for all folds
    #pred_truth_all <- rbind(pred_truth_all, pred_truth)
    pred_truth_all <- rbind(pred_truth_all, pred_truth_df)
  }
  
  return(pred_truth_all)
}
#PART 1

# Logistic regression model
# Assumes the last column of data is the output dimension, and that it's numeric binary
get_pred_logreg <- function(train,test){
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ytrain <- train[ , nf]
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  logreg_mod <- glm(ytrain ~ ., data = xtrain, family = "binomial")
  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(logreg_mod, xtest, type = "response")
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

# SVM model
# Assumes the last column of data is the output dimension
# Assumes numeric binary output column (0 and 1)
get_pred_svm <- function(train,test){
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ##convet to factor
  ytrain <- as.factor(train[ , nf])
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  svm_mod <- svm(ytrain ~., data = xtrain, probability = TRUE)
  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(svm_mod, xtest, probability = TRUE)
  #extract probabilities
  pred <- attributes(pred)
  pred <- pred$probabilities
  #only keep probability of "1" factor
  pred <- pred[,"1"]
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

# Naive Bayes model
# Assumes the last column of data is the output dimension
# Assumes numeric binary output column (0 and 1)
get_pred_nb <- function(train,test){
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ##convet to factor
  ytrain <- as.factor(train[ , nf])
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  nb_mod <- naiveBayes(ytrain ~., data = xtrain)
  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(nb_mod, xtest, type = "raw")
  #only keep probability of factor "1"
  pred <- pred[,"1"]
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

# knn
# Assumes the last column of data is the output dimension
# Assumes numeric binary output column (0 and 1)
get_pred_knn <- function(train, test, k){
  #remove non-numeric columns:
  origcol <- ncol(train)
  num <- sapply(train, is.numeric)
  train <- train[ , num]
  test <- test[ , num]
  removed <- origcol-ncol(train)
  if(removed>0) {
    message(origcol-ncol(train)," non-numeric cols were removed")
  } 
  nf <- ncol(train)
  input <- train[,-nf]
  query <- test[,-nf]
  my.knn <- get.knnx(input,query,k=k) # Get k nearest neighbors
  nn.index <- my.knn$nn.index
  pred <- rep(NA,nrow(test))
  truth <- rep(NA,nrow(test))
  for (ii in 1:nrow(test)){
    neighborIndices <- nn.index[ii,]
    neighborYs <- train[neighborIndices, nf]
    pred[ii] <- mean(neighborYs)
    
    #get true values of test
    truth[ii] = test[ii, nf]
  }
  pred_truth <- cbind(pred, truth)
  
  
  return(pred_truth)  
}


# Default predictor model
# Assumes the last column of data is the output dimension
get_pred_default <- function(train,test){
  # Your implementation goes here
  #find average of output ind var (last col)
  pred <- mean(train[ ,ncol(train)])
  truth <- test[,ncol(test)]
  pred_truth <- cbind(pred, truth)
  return(pred_truth)
}

get_pred_logreg_frm <- function(train,test,frm){
  
  logreg_mod <- glm(as.formula(paste(frm)), data = train, family = "binomial")

  #get predicted values based on test df
  pred <- predict(logreg_mod, test, type = "response")
  
  #get true values of test
  truth <- as.factor(test[,"cancel"])
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

#PART 3
# set default cutoff to 0.5 (this can be overridden at invocation time)
get_metrics <- function(pred_truth, cutoff=0.5) {
  
  #transform probabilities into 0/1
  pred_truth[,"pred"] <- aaply(.data = pred_truth[,"pred"], .margins = 1,.fun = function(x) if(x>=cutoff) 1 else 0 )
  
  #find true positives
  pred_truth$tp <- with(pred_truth, as.numeric(pred==1 & truth==1))
  
  #find true negatives
  pred_truth$tn <- with(pred_truth, as.numeric(pred==0 & truth==0))
  
  #find false positives
  pred_truth$fp <- with(pred_truth, as.numeric(pred==1 & truth==0))
  
  tpr <- with(pred_truth, sum(tp)/sum(truth))
  fpr <- with(pred_truth, sum(fp)/sum(!truth))
  acc <- with(pred_truth, sum(tn+tp)/nrow(pred_truth))
  precision <- with(pred_truth, sum(tp)/sum(tp+fp)) 
  recall <- tpr
  
  metrics <- data.frame(tpr=tpr, fpr=fpr, acc=acc, precision=precision, recall=recall)
  return(metrics)
}



filter.features.by.cor <- function(df) {
  #remove non-numeric columns:
  origcol <- ncol(df)
  num <- sapply(df, is.numeric)
  df <- df[ , num]
  removed <- origcol-ncol(df)
  if(removed>0) {
    message(removed," non-numeric cols were removed")
  } 
  
  #get last column index to id the output col
  lc <- ncol(df)
  
  #get value correlations
  cors <- cor(df[,1:lc])
  
  #only keep cors with output column
  cors <- as.data.frame(cors[,lc])
  
  #remove correlatin of output vs output
  output_index <- which(rownames(cors)==names(df[lc]))
  cors <- cors[-output_index,,drop=FALSE]
  
  #reorder in descending abs correlation order
  cors <- as.data.frame(cors[order(abs(cors), decreasing = TRUE),,drop=FALSE])
  
  #add a column equal to rownames just for convenience 
  cors <- data.frame(var=rownames(cors), weight=cors[,1])
  
  return(cors)
}


# Classification Tree with rpart
library(rpart)

get_pred_tree <- function(train, test) {
  
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ##convet to factor
  ytrain <- as.factor(train[ , nf])
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  # grow tree 
  fit <- rpart(ytrain ~ .,
               method="class", data=xtrain)

  pfit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
  
  printcp(pfit) # display the results 
  plotcp(pfit) # visualize cross-validation results 
  summary(pfit) # detailed summary of splits
  
  # plot tree 
  plot(pfit, uniform=TRUE, 
       main="Classification Tree for Ecard Attrition")
  text(pfit, use.n=TRUE, all=TRUE, cex=.8, pretty=TRUE)
  
  # create attractive postscript plot of tree 
  post(pfit, file = "tree.ps", 
       title = "Classification Tree for Ecard Attrition")
  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(pfit, xtest, type = "prob")
  #only keep probability of factor "1"
  pred <- pred[,"1"]
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

get_pred_randomForest <- function(train, test) {
  
  
  #get number of columns
  nf <- ncol(train)
  
  #get independent variable for training
  ##convet to factor
  ytrain <- as.factor(train[ , nf])
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  # grow tree 
  fit <- randomForest(ytrain ~ ., data=xtrain)
  

  
  #get dependent variables for testing
  xtest <- data.frame(test[ , -nf])
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
 

    # plot tree 
    plot(fit, uniform=TRUE, 
         main="Classification Tree for Kyphosis")
    text(fit, use.n=TRUE, all=TRUE, cex=.8)
    

  #get predicted values based on test df
  pred <- predict(fit, xtest, type = "prob")
  #only keep probability of factor "1"
  pred <- pred[,"1"]
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}

write.csv <- function(ob, filename) {
  write.table(ob, filename, quote = FALSE, sep = ",", row.names = FALSE)
}


get_rmse <- function(pred_truth) {
  #calculate predicted output minus test actual output:
  error <- pred_truth$pred - pred_truth$truth
  
  #calc Sum Squared Errors (SSE)
  SSE <- sum(error*error)
  #calc Mean Squared Errors (MSE)
  MSE <- SSE/nrow(pred_truth)
  RMSE <- MSE^.5
  return(RMSE)
}

get_pred_lr <- function(train,test){
  # Your implementation goes here
  # You may leverage lm function available in R
  
  #get number of features
  nf <- ncol(train)
  
  #get independent variable for training
  ytrain <- train[ , nf]
  #colnames(ytrain) <- colnames(train)[ncol(ytrain)]
  
  #get dependent variables for training
  xtrain <- data.frame(train[ , -nf])
  colnames(xtrain) <- colnames(train)[1:ncol(xtrain)]
  
  lin_mod <- lm(ytrain~., data = xtrain)
  
  #get dependent variables for testing
  xtest <- test
  colnames(xtest) <- colnames(test)[1:ncol(xtest)]
  
  #get predicted values based on test df
  pred <- predict(lin_mod, xtest)
  
  #get true values of test
  truth <- test[,nf]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}


library(survival)

get_pred_cox <- function(train,test){
  # Your implementation goes here
  # You may leverage lm function available in R
  
  #get number of features
  nf <- ncol(train)
  
  model <- coxph(Surv(days_open, cancel) ~ entered + onsite + num_trns + num_trns + holiday + cluster(id), train) 
  
  #get predicted values based on test df
  pred <- predict(model)
  pred <- as.data.frame(pred, test)
  
  #get true values of test
  truth <- test[,"cancel"]
  
  pred_truth <- cbind(pred, truth)
  
  return(pred_truth)
}
