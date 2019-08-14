library(bnlearn)
library(parallel)
library(dplyr)
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
#TODO                    make_option(c("-c", "--cores"), type="integer", default=1,
#TODO                                help="The number of cores to run the pc-algorithm on", metavar=""),
                    make_option(c("-s", "--subset_size"), type="integer", default=-1,
                                help="The maximal size of the conditioning sets that are considered", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided (with impact on performance)", metavar="")
                    #make_option(c("--fixed_gaps"), type="character", default=NULL,
                    #            help="The connections that are removed via prior knowledge", metavar=""),
                    #make_option(c("--fixed_edges"), type="character", default=NULL,
                    #            help="The connections that are fixed via prior knowledge", metavar=""),

);

indepTestDict <- list(gaussCI=gaussCItest, binCI=binCItest, disCI=disCItest)

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

df <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id)
#df <- read.csv(url('http://192.168.30.184:5000/api/dataset/17/load'), header=TRUE, check.names=FALSE, row.names = NULL, colClasses = "character")

if (opt$independence_test == "mi-cg") {
	matrix_df <- df%>%dplyr::mutate_all(funs(if(length(unique(.))<4) as.factor(.)  else as.numeric(as.numeric(.))))
} else {
	stop("No valid independence test specified")
}

subset_size <- if(opt$subset_size < 0) Inf else opt$subset_size
verbose <- opt$verbose > 0
result = pc.stable(matrix_df, debug=verbose, test=opt$independence_test, alpha=opt$alpha, max.sx=subset_size)


graph_request <- store_graph_result_bn(opt$api_host, result, df, opt$job_id, opt$independence_test, opt)
