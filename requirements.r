ipak <- function(pkg){
    new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
    if (length(new.pkg)) 
        install.packages(new.pkg, dependencies = TRUE)
    sapply(pkg, require, character.only = TRUE)
}


# <-------------------------------------------------
# non-CRAN dependencies
bioc_packages <- c('graph', 'RBGL')
# CRAN dependencies
packages <- c('optparse', 'pcalg', 'httr', 'jsonlite')
# <-------------------------------------------------




source('http://www.bioconductor.org/biocLite.R');
for (pkg in ipak(bioc_packages)){
    biocLite(pkg);
}
install.packages(ipak(packages))
