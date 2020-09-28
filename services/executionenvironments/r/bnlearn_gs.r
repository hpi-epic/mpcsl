library(optparse, quietly = T)
library(bnlearn, quietly = T)
library(parallel, quietly = T)
library(dplyr, quietly = T)
source("/scripts/mpci_utils.r")


option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("--api_host"), type="character",
                                help="API Host/Port", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-t", "--independence_test"), type="character", default="mi-cg",
                                help="Independence test used for the bnlearn pc.stable", metavar=""),
                    make_option(c("-a", "--alpha"), type="double", default=0.05,
                                help="This is a hyperparameter", metavar=""),
                    make_option(c("-c", "--cores"), type="integer", default=1,
                                help="The number of cores to run bnlearn in cluster mode on localhost", metavar=""),
                    make_option(c("-s", "--subset_size"), type="integer", default=-1,
                                help="The maximal size of the conditioning sets that are considered", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided (with impact on performance)", metavar=""),
                    make_option(c("--discrete_limit"), type="integer", default=4,
                                help="Maximum unique values per variable considered as discrete", metavar=""),
                    make_option(c("--B"), type="integer", default=NULL,
                                help="The number of permutations considered for each permutation test", metavar=""),
                    make_option(c("--undirected"), type="integer", default=0,
                                help="The number of permutations considered for each permutation test", metavar=""),
                    make_option(c("--sampling_factor"), type="double", default=1.0,
                                help="Data sampling factor to select a random subset, between 0 and 1", metavar="")

);

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

tmp_result <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id, opt$sampling_factor)
df <- tmp_result[[1]]
dataset_loading_time <- tmp_result[[2]]

if (opt$independence_test == "mi-cg") {
	matrix_df <- df%>%dplyr::mutate_all(funs(if(length(unique(.))<opt$discrete_limit) as.factor(.)  else as.numeric(as.numeric(.))))
} else if ( opt$independence_test == "cor" || opt$independence_test == "zf" || 
            opt$independence_test == "mi-g" || opt$independence_test == "mi-g-sh") {
    matrix_df <- df
} else if ( opt$independence_test == "x2" || opt$independence_test == "x2-adf" || 
            opt$independence_test == "mi" || opt$independence_test == "mi-adf" ||
            opt$independence_test == "mi-sh") {
    df[] <- lapply(df, factor)
    before <- ncol(df)
    df <- df[sapply(df, function(x) !is.factor(x) | nlevels(x) > 1)]
    if (ncol(df) < before){
        colorize_log('\033[31m',paste('Removed ',(before - ncol(df))))
    }
    matrix_df <- df
} else {
	stop("No valid independence test specified")
}

subset_size <- if(opt$subset_size < 0) Inf else opt$subset_size
verbose <- opt$verbose > 0
undirected <- opt$undirected > 0
taken <- double()
if (opt$cores == 1) {
    start <- Sys.time()
    result = gs(matrix_df, debug=verbose, test=opt$independence_test, alpha=opt$alpha, B=opt$B, undirected=undirected, max.sx=subset_size)
    end <- Sys.time()
    taken <- as.double(difftime(end,start,unit="s"))
    colorize_log('\033[32m',taken)
} else {
    cl = makeCluster(opt$cores, type = "PSOCK")
    start <- Sys.time()
    result = gs(matrix_df, debug=verbose, test=opt$independence_test, alpha=opt$alpha, B=opt$B, undirected=undirected, max.sx=subset_size, cluster=cl)
    end <- Sys.time()
    taken <- as.double(difftime(end,start,unit="s"))
    colorize_log('\033[32m',taken)
}



graph_request <- store_graph_result_bn(opt$api_host, result, df, opt$job_id, opt$independence_test, opt, taken, dataset_loading_time)
