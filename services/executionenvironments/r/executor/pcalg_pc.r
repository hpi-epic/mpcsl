library(optparse, quietly = T)
library(pcalg, quietly = T)
source("/scripts/mpci_utils.r")
# include for CMI-adaptive tests
require("parallel")
require("SCCI")
require(dplyr)
library("MASS")
Sys.setenv("_R_CHECK_LENGTH_1_CONDITION_" = "true")

source("/scripts/CMI-adaptive-hist/algorithm/generate_candidate_cuts.R")
source("/scripts/CMI-adaptive-hist/algorithm/refine_candidate_cuts_exclude_empty_bins.R")
source("/scripts/CMI-adaptive-hist/algorithm/modified_log.R")
source("/scripts/CMI-adaptive-hist/algorithm/iterative_cmi_greedy_flexible_parallel.R")

source("/scripts/CMI-adaptive-hist/algorithm/multi_hist_splitting_seed_based_simple.R")
source("/scripts/CMI-adaptive-hist/algorithm/oned_hist_iterative.R")
source("/scripts/CMI-adaptive-hist/algorithm/CMI_estimates.R")
source("/scripts/CMI-adaptive-hist/algorithm/CMI_pvals.R")
source("/scripts/CMI-adaptive-hist/algorithm/utils.R")

source("/scripts/kcit_wrapper.r")

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
                    make_option(c("-s", "--subset_size"), type="integer", default=-1,
                                help="The maximal size of the conditioning sets that are considered", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided (with impact on performance)", metavar=""),
                    make_option(c("--skeleton_method"), type="character", default="stable.fast",
                                help="Method used within skeleton, C++ or R", metavar=""),
                    make_option(c("--sampling_factor"), type="double", default=1.0,
                                help="Data sampling factor to select a random subset, between 0 and 1", metavar=""),
                    make_option(c("--sampling_method"), type="character", default="random",
                                help="Data sampling method either random or otherwise top", metavar=""),
                    make_option(c("--discrete_limit"), type="integer", default=11,
                                help="Maximum unique values per variable considered as discrete", metavar="")
                    #make_option(c("--fixed_gaps"), type="character", default=NULL,
                    #            help="The connections that are removed via prior knowledge", metavar=""),
                    #make_option(c("--fixed_edges"), type="character", default=NULL,
                    #            help="The connections that are fixed via prior knowledge", metavar=""),

);

indepTestDict <- list(gaussCI=gaussCItest, binCI=binCItest, disCI=disCItest,
                    CMIpqNML=CMIp.qNML, CMIpfNML=CMIp.fNML, CMIpChisq99=CMIp.Chisq99, CMIpChisq95=CMIp.Chisq95,
                    KCIT=kcitWrapper)

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

tmp_result <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id, opt$sampling_method, opt$sampling_factor)
df <- tmp_result[[1]]
dataset_loading_time <- tmp_result[[2]]

if (opt$independence_test == "gaussCI") {
    matrix_df <- data.matrix(df)
    sufficient_stats <- list(C=cor(matrix_df), n=nrow(matrix_df))
} else if (opt$independence_test == "binCI" || opt$independence_test == "disCI"){
    # Map categories to numbers if not done yet
    df[] <- lapply(df, factor)
    df <- df[sapply(df, function(x) !is.factor(x) | nlevels(x) > 1)]
    matrix_df <- data.matrix(df) - 1

    if (opt$independence_test == "binCI") {
        sufficient_stats <- list(dm=matrix_df, adaptDF=FALSE)
    } else {
        p <- ncol(matrix_df)
        nlev <- vapply(seq_len(p), function(j) length(levels(factor(matrix_df[,j]))), 1L)
        sufficient_stats <- list(dm=matrix_df, adaptDF=FALSE, nlev=nlev)
    }
    # avoid segfaults in C++ extension, limit numCores to 1
    if (opt$independence_test == "binCI"){
        opt$cores <- 1
    }
} else if (opt$independence_test == "CMIpqNML" || opt$independence_test == "CMIpfNML" ||
           opt$independence_test == "CMIpChisq99" || opt$independence_test == "CMIpChisq95"){
    matrix_df <- data.matrix(df)
    type <- rep(1, ncol(matrix_df))
    for(i in 1:ncol(matrix_df)) {
        if(length(unique(matrix_df[ , i])) < opt$discrete_limit) {
            type[i] = 0
        }
    }
    sufficient_stats = list(dm=matrix_df, type=type, n=nrow(df))
} else if (opt$independence_test == "KCIT") {
    sufficient_stats <- list(dm=df)
} else {
    stop("No valid independence test specified")
}

subset_size <- if(opt$subset_size < 0) Inf else opt$subset_size
verbose <- opt$verbose > 0
start <- Sys.time()
result = pc(suffStat=sufficient_stats, verbose=verbose,
            indepTest=indepTestDict[[opt$independence_test]], m.max=subset_size,
            p=ncol(matrix_df), alpha=opt$alpha, numCores=opt$cores, skel.method=opt$skeleton_method)
end <- Sys.time()
taken <- as.double(difftime(end,start,unit="s"))
colorize_log('\033[32m',taken)
graph_request <- store_graph_result(opt$api_host, result@'graph', result@'sepset', df, opt$job_id, opt$independence_test, opt$send_sepsets, opt, taken, dataset_loading_time)
