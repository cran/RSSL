% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MCPLDA.R
\name{projection_simplex}
\alias{projection_simplex}
\title{Project an n-dim vector y to the simplex Dn}
\usage{
projection_simplex(y)
}
\arguments{
\item{y}{matrix with vectors to be projected onto the simplex}
}
\value{
projection of y onto the simplex
}
\description{
Where \eqn{Dn = \{ 0 <= x <= 1, sum(x) = 1\}}. 
R translation of Loog's version of Xiaojing Ye's initial implementation.
The algorithm works row-wise.
}
\references{
Algorithm is explained in http://arxiv.org/abs/1101.6081
}
