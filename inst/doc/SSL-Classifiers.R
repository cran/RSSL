## ----echo=FALSE, include=FALSE-------------------------------------------
library(RSSL)
library(dplyr)
library(ggplot2)
library(ggthemes)

## ----echo=FALSE----------------------------------------------------------
set.seed(42)
data_2gauss <- data.frame(generate2ClassGaussian(n=100,d=2,var=0.2)) %>% 
  add_missinglabels_mar(formula=Class~.,prob=0.9)
problem_2gauss <-  data_2gauss %>% df_to_matrices(Class~.)

problem1 <- problem_2gauss

p1 <- ggplot(data_2gauss,aes(x=X1,y=X2,shape=Class,color=Class)) +
  geom_point(size=6,alpha=0.8) +
  coord_equal() +
  theme_tufte(base_family = "sans",base_size = 18) +
  scale_shape_stata(na.value=16) +
  scale_color_colorblind(na.value="grey") +
  theme(axis.title.y=element_text(angle = 0, hjust = 0)) +
  scale_linetype_stata() +
  labs(y="",x="")
print(p1)

## ----echo=FALSE----------------------------------------------------------
set.seed(42)
data_slicedcookie <- data.frame(generateSlicedCookie(100,expected=TRUE))[sample(1:100),] %>% 
  add_missinglabels_mar(prob=0.9,formula=Class~.)
problem_slicedcookie <- data_slicedcookie %>% df_to_matrices

problem2 <- problem_slicedcookie

p2 <- ggplot(data_slicedcookie,aes(x=X1,y=X2,shape=Class,color=Class)) +
  geom_point(size=6,alpha=0.8) +
  coord_equal() +
  theme_tufte(base_family = "sans",base_size = 18) +
  scale_shape_stata(na.value=16) +
  scale_color_colorblind(na.value="grey") +
  theme(axis.title.y=element_text(angle = 0, hjust = 0)) +
  scale_linetype_stata() +
  labs(y="",x="")
print(p2)

## ------------------------------------------------------------------------
g_ls <- LeastSquaresClassifier(problem1$X,problem1$y)
g_self <- SelfLearning(problem1$X,problem1$y,problem1$X_u,LeastSquaresClassifier)

## ----echo=FALSE, warning=FALSE, fig.width=8, fig.height=5, dev="svg"-----


p1 + geom_classifier("LS"=g_ls,"Self Learning"=g_self)

## ------------------------------------------------------------------------
g_ls <- LeastSquaresClassifier(problem2$X,problem2$y)
g_self <- SelfLearning(problem2$X,problem2$y,problem2$X_u,LeastSquaresClassifier)

## ----echo=FALSE, warning=FALSE, fig.width=8, fig.height=5, dev="svg"-----
p2 + geom_classifier("LS"=g_ls,"Self Learning"=g_self)

## ------------------------------------------------------------------------
g_lda <- LinearDiscriminantClassifier(problem1$X,problem1$y)
g_emlda <- EMLinearDiscriminantClassifier(problem1$X,problem1$y,problem1$X_u)
g_sllda <- SelfLearning(problem1$X,problem1$y,problem1$X_u,LinearDiscriminantClassifier)

## ----echo=FALSE, warning=FALSE, fig.width=8, fig.height=5, dev="svg"-----
p1 + geom_classifier("LDA"=g_lda,"EMLDA"=g_emlda,"SLLDA"=g_sllda)

## ------------------------------------------------------------------------
g_lda <- LinearDiscriminantClassifier(problem2$X,problem2$y)
g_emlda <- EMLinearDiscriminantClassifier(problem2$X,problem2$y,problem2$X_u)
g_sllda <- SelfLearning(problem2$X,problem2$y,problem2$X_u,LinearDiscriminantClassifier)

## ----echo=FALSE, warning=FALSE, fig.width=8, fig.height=5, dev="svg"-----
p2 + geom_classifier("LDA"=g_lda,"EMLDA"=g_emlda,"SLLDA"=g_sllda)

## ------------------------------------------------------------------------
g_svm <- LinearSVM(problem2$X,problem2$y,scale=TRUE)
g_tsvm <- TSVMcccp_lin(problem2$X,problem2$y,problem2$X_u,C=1,Cstar=100)
g_tsvm2 <- TSVMcccp(problem2$X,problem2$y,problem2$X_u,C=1,Cstar=1)

## ------------------------------------------------------------------------
p2 + geom_classifier("SVM"=g_svm,"TSVM"=g_tsvm)

g_ls <-LeastSquaresClassifier(problem2$X,problem2$y,scale=TRUE,y_scale=TRUE)

Xt <- as.matrix(expand.grid(X1=seq(-4,4,length.out = 100),X2=seq(-4,4,length.out = 100)))
data.frame(Xt,pred=decisionvalues(g_ls,Xt)) %>% 
  ggplot(aes(x=X1,y=X2,z=pred)) + geom_contour(size=5,breaks=c(0.5)) +
  geom_classifier(g_ls)

Xt <- as.matrix(expand.grid(X1=seq(-4,4,length.out = 100),X2=seq(-4,4,length.out = 100)))
data.frame(Xt,pred=decisionvalues(g_svm,Xt)) %>% 
  ggplot(aes(x=X1,y=X2,z=pred)) + geom_contour(size=5,breaks=c(0)) +
  geom_classifier(g_svm)


## ------------------------------------------------------------------------
#g_s4vm <- S4VM(problem1$X,problem1$y,problem1$X_u) # Returns NULL

## ------------------------------------------------------------------------
  g_lr <- LogisticLossClassifier(problem2$X,problem2$y)
  g_erlr <- ERLogisticLossClassifier(problem2$X,problem2$y,problem2$X_u,lambda_entropy = 0,init = rnorm(3))

## ----echo=FALSE, warning=FALSE, fig.width=8, fig.height=5, dev="svg"-----
p2 + geom_classifier("LL"=g_lr,"ERLR"=g_erlr)

## ------------------------------------------------------------------------
set.seed(42)
generateTwoCircles(200,0.05) %>%
  ggplot(aes(X1,X2,color=Class)) %>%
  + geom_point() %>%
  + coord_equal()

problem_circle <- 
  generateTwoCircles(200, 0.05) %>% 
  SSLDataFrameToMatrices(Class~.,.) 
problem_circle <- split_dataset_ssl(problem_circle$X, problem_circle$y,frac_ssl=0.98)

g_grf <- GRFClassifier(problem_circle$X,problem_circle$y,problem_circle$X_u,sigma=0.05)

data.frame(problem_circle$X,Class=problem_circle$y) %>% 
  ggplot(aes(x=X1,y=X2,color=Class)) +
  geom_responsibilities(g_grf,problem_circle$X_u) +
  geom_point(aes(shape=Class,color=Class),size=6) +
  coord_equal()

## ------------------------------------------------------------------------
g_sup <- LeastSquaresClassifier(problem1$X,problem1$y)
g_proj_sup <- ICLeastSquaresClassifier(problem1$X,problem1$y,problem1$X_u,projection = "supervised")
g_proj_semi <- ICLeastSquaresClassifier(problem1$X,problem1$y,problem1$X_u,projection = "semisupervised")
g_proj_euc <- ICLeastSquaresClassifier(problem1$X,problem1$y,problem1$X_u,projection = "euclidean")

## ----echo=FALSE, warning=FALSE, fig.width=8, fig.height=5, dev="svg"-----
p1 + geom_classifier("LS"=g_sup,"Proj_sup"=g_proj_sup,"Proj_semi"=g_proj_semi,"Proj_Euc"=g_proj_euc)

## ------------------------------------------------------------------------
g_sup <- LinearDiscriminantClassifier(problem1$X, problem1$y)
g_semi <- ICLinearDiscriminantClassifier(problem1$X,problem1$y,problem1$X_u)

## ----echo=FALSE, warning=FALSE, fig.width=8, fig.height=5, dev="svg"-----
p1 + geom_classifier("LDA"=g_sup,"ICLDA"=g_semi)

## ------------------------------------------------------------------------
problem <-
  generate2ClassGaussian(n=1000,d = 2,var = 0.3,expected = TRUE) %>% 
  SSLDataFrameToMatrices(Class~.,.) %>% 
  {split_dataset_ssl(.$X,.$y,frac_train=0.5,frac_ssl=0.98) }

sum(loss(LeastSquaresClassifier(problem$X,problem$y),problem$X_test,problem$y_test))
sum(loss(USMLeastSquaresClassifier(problem$X,problem$y,problem$X_u),problem$X_test,problem$y_test))
sum(loss(ICLeastSquaresClassifier(problem$X,problem$y,problem$X_u),problem$X_test,problem$y_test))
sum(loss(ICLeastSquaresClassifier(problem$X,problem$y,problem$X_u,projection="semisupervised"),problem$X_test,problem$y_test))
mean(predict(ICLeastSquaresClassifier(problem$X,problem$y,problem$X_u,projection="semisupervised"),problem$X_test)==problem$y_test)
mean(predict(LeastSquaresClassifier(problem$X,problem$y),problem$X_test)==problem$y_test)

sum(loss(SelfLearning(X=problem$X,y=problem$y,X_u=problem$X_u,method=LeastSquaresClassifier),problem$X_test,problem$y_test))

# Nearest Mean
sum(loss(NearestMeanClassifier(X=problem$X,y=problem$y),problem$X_test,problem$y_test))
sum(loss(ICNearestMeanClassifier(X=problem$X,y=problem$y,X_u=problem$X_u),problem$X_test,problem$y_test))
sum(loss(EMNearestMeanClassifier(X=problem$X,y=problem$y,X_u=problem$X_u),problem$X_test,problem$y_test))
sum(loss(SelfLearning(X=problem$X,y=problem$y,X_u=problem$X_u,method=NearestMeanClassifier),problem$X_test,problem$y_test))


# LDA
sum(loss(LinearDiscriminantClassifier(X=problem$X,y=problem$y),problem$X_test,problem$y_test))
sum(loss(ICLinearDiscriminantClassifier(X=problem$X,y=problem$y,X_u=problem$X_u),problem$X_test,problem$y_test))
sum(loss(EMLinearDiscriminantClassifier(X=problem$X,y=problem$y,X_u=problem$X_u),problem$X_test,problem$y_test))
sum(loss(SelfLearning(X=problem$X,y=problem$y,X_u=problem$X_u,method=LinearDiscriminantClassifier),problem$X_test,problem$y_test))


# Logistic Regression
#sum(loss(LogisticRegression(X=problem$X,y=problem$y),problem$X_test,problem$y_test))
sum(loss(LogisticLossClassifier(X=problem$X,y=problem$y,x_center=TRUE),problem$X_test,problem$y_test))
sum(loss(ERLogisticLossClassifier(X=problem$X,y=problem$y,X_u=problem$X_u),problem$X_test,problem$y_test))
mean(predict(LogisticLossClassifier(X=problem$X,y=problem$y),problem$X_test)==problem$y_test)
mean(predict(ERLogisticLossClassifier(X=problem$X,y=problem$y,X_u=problem$X_u),problem$X_test)==problem$y_test)

# SVM
sum(loss(LinearSVM(X=problem$X,y=problem$y),problem$X_test,problem$y_test))

