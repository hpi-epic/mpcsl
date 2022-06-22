
#
#
#
# <-------------------------------------------------
# CRAN dependencies
packages <- c('optparse', 'stringi', 'httr', 'jsonlite', 'ParallelPC', 'infotheo', 'bnlearn', 'vctrs', 'dplyr', 'hash','Ckmeans.1d.dp','abind', 'igraph', 'ggm', 'corpcor', 'robustbase', 'vcd', 'bdsmatrix', 'sfsmisc', 'fastICA', 'clue', 'RcppArmadillo')
# non-CRAN dependencies
bioc_packages <- c('graph', 'RBGL','bnlearn')
# <-------------------------------------------------
#
#
#

not_installed <- function(pkg){
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    return(length(new.pkg) > 0)
}

#source('http://www.bioconductor.org/biocLite.R');
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
for (pkg in bioc_packages){
    if(not_installed(pkg)){
        BiocManager::install(pkg);
        sapply(pkg, require, character.only = TRUE)
    }
}

for (pkg in packages){
    if(not_installed(pkg)){
        install.packages(pkg)
        sapply(pkg, require, character.only = TRUE)
    }
}
