
#
#
#
# <-------------------------------------------------
# CRAN dependencies
packages <- c('optparse', 'stringi', 'httr', 'jsonlite', 'ParallelPC', 'infotheo', 'vctrs', 'dplyr', 'hash')
# non-CRAN dependencies
bioc_packages <- c('graph', 'RBGL')
# <-------------------------------------------------
#
#
#

not_installed <- function(pkg){
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    return(length(new.pkg) > 0)
}

source('http://www.bioconductor.org/biocLite.R');
for (pkg in bioc_packages){
    if(not_installed(pkg)){
        biocLite(pkg);
        sapply(pkg, require, character.only = TRUE)
    }
}

for (pkg in packages){
    if(not_installed(pkg)){
        install.packages(pkg)
        sapply(pkg, require, character.only = TRUE)
    }
}
