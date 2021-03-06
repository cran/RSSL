% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/GenerateSSLData.R
\name{generateSlicedCookie}
\alias{generateSlicedCookie}
\title{Generate Sliced Cookie dataset}
\usage{
generateSlicedCookie(n = 100, expected = FALSE, gap = 1)
}
\arguments{
\item{n}{integer; number of observations to generate}

\item{expected}{logical; TRUE if the large margin equals the class boundary, FALSE if the class boundary is perpendicular to the large margin}

\item{gap}{numeric; Size of the gap}
}
\value{
A data.frame with n objects from the sliced cookie example
}
\description{
Generate a sliced cookie dataset: a circle with a large margin in the middle.
}
\examples{
data <- generateSlicedCookie(1000,expected=FALSE)
plot(data[,1],data[,2],col=data$Class,asp=1)
}
\seealso{
Other RSSL datasets: 
\code{\link{generate2ClassGaussian}()},
\code{\link{generateABA}()},
\code{\link{generateCrescentMoon}()},
\code{\link{generateFourClusters}()},
\code{\link{generateParallelPlanes}()},
\code{\link{generateSpirals}()},
\code{\link{generateTwoCircles}()}
}
\concept{RSSL datasets}
