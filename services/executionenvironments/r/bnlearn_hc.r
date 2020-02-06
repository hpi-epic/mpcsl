library(optparse, quietly = T)
library(bnlearn, quietly = T)
library(dplyr, quietly = T)
source("/scripts/mpci_utils.r")


option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("--api_host"), type="character",
                                help="API Host/Port", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-s", "--score"), type="character",
                                help="Score used for the bnlearn hc", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("--restarts"), type="integer", default=0,
                                help="The number of random restarts", metavar=""),
                    make_option(c("--perturb"), type="integer", default=1,
                                help="The number of attempts to randomly insert/remove/reverse an arc on every random restart", metavar=""),
                    make_option(c("--maxiter"), type="integer", default=Inf,
                                help="The maximum number of iterations", metavar=""),
                    make_option(c("--maxp"), type="integer", default=Inf,
                                help="The maximum number of parents for a node", metavar=""),
                    make_option(c("-o", "--optimized"), type="integer", default=1,
                                help="If TRUE (the default), score caching is used to speed up structure learning", metavar=""),
                    make_option(c("--discrete_limit"), type="integer", default=4,
                                help="Maximum unique values per variable considered as discrete", metavar=""));

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

tmp_result <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id)
df <- tmp_result[[1]]
dataset_loading_time <- tmp_result[[2]]

if (opt$score == "loglik-cg" || opt$score == "aic-cg" || opt$score == "bic-cg" || opt$score == "pred-loglik-cg") {
	matrix_df <- df%>%dplyr::mutate_all(funs(if(length(unique(.))<opt$discrete_limit) as.factor(.)  else as.numeric(as.numeric(.))))
} else if ( opt$score == "loglik-g" || opt$score == "aic-g" || 
            opt$score == "bic-g" || opt$score == "pred-loglik-g" || opt$score == "bge") {
    matrix_df <- df
} else if ( opt$score == "loglik" || opt$score == "aic" || 
            opt$score == "bic" || opt$score == "pred-loglik" ||
            opt$score == "bde" || opt$score == "bds" ||
            opt$score == "mbde" || opt$score == "bdla" || opt$score == "k2") {
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

verbose <- opt$verbose > 0
optimized <- opt$optimized > 0
maxiter_value <- opt$maxiter
if(maxiter_value == -1){
    maxiter_value <- Inf
}
maxp_value <- opt$maxp
if(maxp_value == -1){
    maxp_value <- Inf
}

taken <- double()
start <- Sys.time()
result = hc(matrix_df, score=opt$score, debug=verbose, restart=opt$restart, perturb=opt$perturb, max.iter=maxiter_value, maxp=maxp_value, optimized=optimized)
end <- Sys.time()
taken <- as.double(difftime(end,start,unit="s"))
colorize_log('\033[32m',taken)

graph_request <- store_graph_result_bnlearn_hc(opt$api_host, result, df, opt$job_id, opt, taken, dataset_loading_time)
