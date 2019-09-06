[Go to main page](../README.md)

## Download R and R Studio
1. Go to the [CRAN website](https://cran.r-project.org) and follow the instructions to download and install R.
2. Download and install [RStudio](https://www.rstudio.com/products/rstudio/download/#download).

## Installing additional packages
Open RStudio and install the following packages.

```{r}
## To install needed CRAN packages:
install.packages("tidyverse")
install.packages("GGally")
install.packages("caret")
install.packages("devtools")
install.packages("gmodels")
install.packages("dendextend")
install.packages("randomForest")

## To install needed Bioconductor packages:
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("limma", "edgeR"))

# To install libraries from GitHub source
library(devtools)
install_github("vqv/ggbiplot")
```

Load them to make sure they were successfully installed.
```
# Load packages
library(tidyverse) # working with data frames, plotting
library(GGally)
library(caret)
library(gmodels)
library(ggbiplot)
library(dendextend)
library(randomForest)

library(edgeR)      # cpm, etc -- RNA-Seq normalization
library(limma)      # lmFit, etc -- fitting many models
```
