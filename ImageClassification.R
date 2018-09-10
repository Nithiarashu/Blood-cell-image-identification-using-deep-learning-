library(jpeg)
library(EBImage)

#Path of the data from where the images should be fetched
Data.dir <- ("C:/Nithi/Masters Data Analytics/ADM/ADM CA2/x17154154-ca2/JPEGTest")

#Get all the file names in a vector
pic1 <- as.vector(list.files(path = Data.dir, full.names = TRUE, include.dirs = FALSE))

mydata <- c()

#Read al the JPEG files
for (i in 1:length(pic1)) {mydata[[i]] <- readJPEG(pic1[[i]])}

#Resize the images to 28*28 and convert them to a grayscale image
for (i in 1:length(mydata)) {mydata[[i]] <- resize(Image(data = mydata[[i]], dim = dim(mydata[[i]]), colormode = "Grayscale"), 28, 28)}

#Getting the features of the images in a vector
for (i in 1:length(mydata)) {mydata[[i]] <- as.vector(mydata[[i]])}

#Converting the vector to a data frame for further processing
dfr <- as.data.frame(do.call(rbind, mydata))

#Read the labels of the images from a csv
lab <- read.csv(file = "C:/Nithi/Masters Data Analytics/ADM/ADM CA2/x17154154-ca2/labelsTest.csv", header = T)

#Merge the label vector to the data frame we have created earlier
dfr <- data.frame(cbind(lab$Category, dfr))
table(dfr$lab.Category)
dfr$lab.Category <- factor(dfr$lab.Category)
plot(dfr$lab.Category, xlab = 'Class',ylab = 'Count of Images', lwd = 2, col="Navy",main = "Class Imbalance")

#To make the dataset balanced
library(UBL)
df <- SmoteClassif(lab.Category ~ ., dfr, C.perc = "balance", repl = FALSE)

#Changing the labels of the dependent variable
df$lab.Category <- factor(df$lab.Category, levels = c("BASOPHIL", "EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"), labels = c(0,1,2,3,4))
names(df)[1] <- c("Label")
plot(df$Label, xlab = 'Class',ylab = 'Count of Images', lwd = 2, col="Navy",main = "Class Balance")
str(df)

#Check if any feature could be removed
columnsKeep <- names(which(colSums(df[,-1]) > 0))
df <- df[c("Label", columnsKeep)]

library(caret)
library(e1071)
library(ggfortify)

set.seed(17154154)
idx <- createDataPartition(df$Label, p=0.75, list = FALSE)

#PCA on the features
pca <- prcomp(df[idx,-1], scale. = F, center = F)
autoplot(pca, data = df[idx,], colour='Label')
screeplot(pca, type = "lines", npcs = 70, main = 'Screeplot of PCA')

var.pca <- pca$sdev ^ 2
x.var.pca <- var.pca/sum(var.pca)
cum.var.pca <- cumsum(x.var.pca)

#Graph to determine the number of pcs required
plot(cum.var.pca[1:100], xlab = "No. of PCs", 
     ylab = "Cumulative Proportion of variance explained", ylim = c(0,1), type = 'b')

#PCA rotation on the data
pcs <- 30
indata <- as.data.frame(as.matrix(df[,-1]) %*% pca$rotation[,1:pcs])
indata <- data.frame(cbind(df[,1], indata))
names(indata)[1] <- c("Label")
hist(indata$PC6, xlab = 'PC6', main = "Normal Distribution of PC6", breaks = 25, col = rgb(0.1, 0.1, 0.9, 0.9))
qqnorm(indata$PC6)

#Splitting the data into train and test
train <- indata[idx,]
test <- indata[-idx,]


#SVM model
train$Label <- factor(train$Label)
svmModel <- svm(train[,-1], indata[idx,1], kernel = "polynomial")
results <- predict(svmModel,test[,-1])
#Confusion Matrix
a <- (confusionMatrix(results, indata[-idx,1]))
plot(a$table, xlab = 'Predicted', ylab = 'Observed', main = 'Confusion Matrix', col = 'Navy')
str(a$byClass[,c(5,6,7,11)])

#To plot confusion matrix
library(ggplot2)
ggplot(data =  a$table, mapping = aes(x = Prediction, y = Reference))+
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f",Freq)), vjust = 1) +
  scale_fill_gradient(low = "Sky Blue", high = "Blue",trans = "log") +
  ggtitle("Confusion Matrix of SVM") +
  labs(y = 'Observed Class', x='Predicted Class')



#Deep Learning using mxnet
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
library(mxnet)
library(mlbench)
indata$Label <- factor(indata$Label)

data(indata, package = "mlbench")
train.ind <- seq(1,344,4)

str(train)

train.x <- data.matrix(indata[-train.ind, -1])
train.y <- indata[-train.ind,1]
test.x <- data.matrix(indata[train.ind, -1])
test.y <- indata[train.ind,1]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, num_hidden = 128, name = "fc1")
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

mx.set.seed(0)
mnetModel <- mx.model.FeedForward.create(softmax, X = train.x, y = train.y, array.batch.size=50,
                                         ctx = mx.cpu(),momentum = 0.9,
                                         learning.rate = 0.2, num.round=50,
                                         eval.metric=mx.metric.accuracy)

mxPred <- predict(mnetModel, test.x)



#Hyperparameter Optimization using Deep Learning
library(h2o)
hidden_opt <- list(c(5,5,5,5,5), c(10, 10, 10, 10), c(50, 50, 50, 50)) 
l1 = c(0, 0.00001, 0.0001)
l2 = c(0, 0.00001, 0.0001)
rate = c(0, 01, 0.005, 0.001)
activations <- c("Rectifier", "Maxout", "Tanh", "RectifierWithDropout", "MaxoutWithDropout", "TanhWithDropout")
hyper_params <- list(hidden = hidden_opt, l1 = l1, activation=activations, l2 = l2, momentum_start = c(0, 0.5),
                     momentum_stable = c(0.99, 0.5, 0), rate = rate)

h2o.init(ip = "localhost", port = 54321)
train$Label <- factor(train$Label)
h2otrain <- train
h2otrain <- as.h2o(h2otrain)
h2otest <- test
h2otest <- as.h2o(h2otest)

#Model Grid
model_grid <- h2o.grid("deeplearning",
                       hyper_params = hyper_params,
                       x = c(2:length(h2otrain)),  # column numbers for predictors
                       y = 1,   # column number for label
                       training_frame = h2otrain,
                       validation_frame = h2otest)

dlPerf <- c()
str(test)
test$Label <- as.integer(test$Label)
for (model_id in model_grid@model_ids){
  model <- h2o.getModel(model_id)
  pred <- h2o.predict(model, h2otest)
  pred <- as.data.frame(pred)
  dlPerformance <- 1 - mean(pred$predict != test$Label)
  dlPerf <- rbind(dlPerf, dlPerformance)
}
#The best accuracy
(bestDL <- max(dlPerf))


#The best model's performance
str(model_grid)
Optmodel <- h2o.getModel('Grid_DeepLearning_h2otrain_model_R_1533032935338_1_model_2678') #Model id which gave the highest accuracy
Optpred <- h2o.predict(Optmodel, h2otest)
Optpred <- as.data.frame(Optpred)
OptPerformance <- 1 - mean(Optpred$predict != test$Label)


str(test)
test$Label <- factor(test$Label)
cm <- confusionMatrix(Optpred$predict,test$Label)

#To plot confusion matrix
library(ggplot2)
ggplot(data =  cm$table, mapping = aes(x = Prediction, y = Reference))+
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f",Freq)), vjust = 1) +
  scale_fill_gradient(low = "White", high = "Blue",trans = "log") +
  ggtitle("Confusion Matrix of Neural Network - Deep Learning") +
  labs(y = 'Observed Class', x='Predicted Class')
