library(optparse, quietly = T)
library(pcalg, quietly = T)
## TODO CHANGE TO CUPC
source("/scripts/cuPC.R")
source("/scripts/mpci_utils.r")


option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("--api_host"), type="character",
                                help="API Host/Port", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    #make_option(c("-t", "--independence_test"), type="character", default="gaussCI",
                    #            help="Independence test used for the pcalg", metavar=""),
                    make_option(c("-a", "--alpha"), type="double", default=0.05,
                                help="This is a hyperparameter", metavar=""),
                    #make_option(c("-c", "--cores"), type="integer", default=1,
                    #            help="The number of cores to run the pc-algorithm on", metavar=""),
                    make_option(c("-s", "--subset_size"), type="integer", default=-1,
                                help="The maximal size of the conditioning sets that are considered", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided (with impact on performance)", metavar=""),
                    make_option(c("--sampling_factor"), type="double", default=1.0,
                                help="Data sampling factor to select a random subset, between 0 and 1", metavar="")
                    #make_option(c("--skeleton_method"), type="character", default="stable.fast",
                    #            help="Method used within skeleton, C++ or R", metavar="")
                    #make_option(c("--fixed_gaps"), type="character", default=NULL,
                    #            help="The connections that are removed via prior knowledge", metavar=""),
                    #make_option(c("--fixed_edges"), type="character", default=NULL,
                    #            help="The connections that are fixed via prior knowledge", metavar=""),

);

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

tmp_result <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id, opt$sampling_factor)
df <- tmp_result[[1]]
dataset_loading_time <- tmp_result[[2]]


matrix_df <- data.matrix(df)
### warm up GPU
tmp <- list(C=cor(cbind(c(1,0.5,0.5),c(0.3,1,0.3),c(0.5,0.3,1))), n=3)
cu_pc(suffStat=tmp, p=3, m.max=1, alpha=0.01, verbose=verbose)
### warm up end

sufficient_stats <- list(C=cor(matrix_df), n=nrow(matrix_df))
subset_size <- if(opt$subset_size < 0) Inf else opt$subset_size
verbose <- opt$verbose > 0

start <- Sys.time()
result = cu_pc(suffStat=sufficient_stats, p=ncol(matrix_df), m.max=subset_size, alpha=opt$alpha, verbose=verbose)
end <- Sys.time()
taken <- as.double(difftime(end,start,unit="s"))
colorize_log('\033[32m',taken)
graph_request <- store_graph_result(opt$api_host, result@'graph', result@'sepset', df, opt$job_id, opt$independence_test, opt$send_sepsets, opt, taken, dataset_loading_time)
