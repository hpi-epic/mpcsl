library(optparse, quietly = T)
library(pcalg, quietly = T)
library(ParallelPC, quietly = T)
source("src/master/executor/algorithms/r/mpci_utils.r")

option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("--api_host"), type="character",
                                help="API Host/Port", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-t", "--independence_test"), type="character", default="gaussCI",
                                help="Independence test used for the pcalg", metavar=""),
                    make_option(c("-a", "--alpha"), type="double", default=0.05,
                                help="This is a hyperparameter", metavar=""),
                    make_option(c("-c", "--cores"), type="integer", default=1,
                                help="The number of cores to run the pc-algorithm on", metavar=""),
                    make_option(c("--fixed_gaps"), type="character", default=FALSE,
                                help="The connections that are removed via prior knowledge", metavar=""),
                    make_option(c("--fixed_edges"), type="character", default=FALSE,
                                help="The connections that are fixed via prior knowledge", metavar="")
);

indepTestDict <- list(gaussCI=gaussCItest, binCI=binCItest, disCI=disCItest)

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

df <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id)

matrix_df <- data.matrix(df)

if (opt$independence_test == "gaussCI") {
    sufficient_stats <- list(C=cor(matrix_df), n=nrow(matrix_df))
} else if (opt$independence_test == "binCI") {
    sufficient_stats <- list(dm=matrix_df, adaptDF=FALSE)
} else if (opt$independence_test == "disCI"){
    p <- ncol(matrix_df)
    nlev <- vapply(seq_len(p), function(j) length(levels(factor(matrix_df[,j]))), 1L)
    sufficient_stats <- list(dm=matrix_df, adaptDF=FALSE, nlev=nlev)
} else{
    stop("No valid independence test specified")
}

result = pc_parallel(suffStat=sufficient_stats,
            indepTest=indepTestDict[[opt$independence_test]],
            p=ncol(matrix_df), alpha=opt$alpha, num.cores=opt$cores, skel.method="parallel", verbose=TRUE)

graph_request <- store_graph_result(opt$api_host, result@'graph', df, opt$job_id, opt)
