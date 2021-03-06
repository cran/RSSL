% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MCNearestMeanClassifier.R
\name{MCNearestMeanClassifier}
\alias{MCNearestMeanClassifier}
\title{Moment Constrained Semi-supervised Nearest Mean Classifier}
\usage{
MCNearestMeanClassifier(X, y, X_u, update_sigma = FALSE, prior = NULL,
  x_center = FALSE, scale = FALSE)
}
\arguments{
\item{X}{matrix; Design matrix for labeled data}

\item{y}{factor or integer vector; Label vector}

\item{X_u}{matrix; Design matrix for unlabeled data}

\item{update_sigma}{logical; Whether the estimate of the variance should be updated after the means have been updated using the unlabeled data}

\item{prior}{matrix; Class priors for the classes}

\item{x_center}{logical;  Should the features be centered?}

\item{scale}{logical; Should the features be normalized? (default: FALSE)}
}
\description{
Update the means based on the moment constraints as defined in Loog (2010). The means estimated using the labeled data are updated by making sure their weighted mean corresponds to the overall mean on all (labeled and unlabeled) data. Optionally, the estimated variance of the classes can be re-estimated after this update is applied by setting update_sigma to \code{TRUE}. To get the true nearest mean classifier, rather than estimate the class priors, set them to equal priors using, for instance \code{prior=matrix(0.5,2)}.
}
\references{
Loog, M., 2010. Constrained Parameter Estimation for Semi-Supervised Learning: The Case of the Nearest Mean Classifier. In Proceedings of the 2010 European Conference on Machine learning and Knowledge Discovery in Databases. pp. 291-304.
}
\seealso{
Other RSSL classifiers: 
\code{\link{EMLeastSquaresClassifier}},
\code{\link{EMLinearDiscriminantClassifier}},
\code{\link{GRFClassifier}},
\code{\link{ICLeastSquaresClassifier}},
\code{\link{ICLinearDiscriminantClassifier}},
\code{\link{KernelLeastSquaresClassifier}},
\code{\link{LaplacianKernelLeastSquaresClassifier}()},
\code{\link{LaplacianSVM}},
\code{\link{LeastSquaresClassifier}},
\code{\link{LinearDiscriminantClassifier}},
\code{\link{LinearSVM}},
\code{\link{LinearTSVM}()},
\code{\link{LogisticLossClassifier}},
\code{\link{LogisticRegression}},
\code{\link{MCLinearDiscriminantClassifier}},
\code{\link{MCPLDA}},
\code{\link{MajorityClassClassifier}},
\code{\link{NearestMeanClassifier}},
\code{\link{QuadraticDiscriminantClassifier}},
\code{\link{S4VM}},
\code{\link{SVM}},
\code{\link{SelfLearning}},
\code{\link{TSVM}},
\code{\link{USMLeastSquaresClassifier}},
\code{\link{WellSVM}},
\code{\link{svmlin}()}
}
\concept{RSSL classifiers}
