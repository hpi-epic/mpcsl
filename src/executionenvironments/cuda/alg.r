library(optparse, quietly = T)
library('gpucausalrcpp', quietly = T)
source("/scripts/mpci_utils.r")

option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("--api_host"), type="character",
                                help="API Host/Port", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-a", "--alpha"), type="double", default=0.05,
                                help="This is a hyperparameter", metavar=""),
                    make_option(c("-s", "--subset_size"), type="integer", default=-1,
                                help="The maximal size of the conditioning sets that are considered", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided (with impact on performance)", metavar="")

);

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

df <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id)

matrix_df <- data.matrix(df)

verbose <- opt$verbose > 0

result = estimateSkeleton(matrix_df, alpha=opt$alpha, maxCondSize=opt$subset_size, verbose=verbose)

# graph_request <- store_graph_result(opt$api_host, [], result, df, opt$job_id, "gaussCI", opt$send_sepsets, opt)
