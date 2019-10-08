library(optparse, quietly = T)
library(pcalg, quietly = T)
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

t_start <- Sys.time()
tmp = estimateSkeleton(matrix_df, alpha=opt$alpha, maxCondSize=opt$subset_size, verbose=verbose)
t_skel <- Sys.time()

p <- ncol(matrix_df)
seq_p <- seq_len(p)
labels <- as.character(seq_len(p))
G <- tmp$amat
sepset <- lapply(seq_p, function(i) c(
    lapply(tmp$sepset[[i]], function(v) if(identical(v, as.integer(-1))) NULL else v),
    vector("list", p - length(tmp$sepset[[i]])))) # TODO change convention: make sepset triangular
pMax <- tmp$pMax
n.edgetests <- tmp$n.edgetests
ord <- length(n.edgetests) - 1L

Gobject <-
    if (sum(G) == 0) {
      new("graphNEL", nodes = labels)
    } else {
      colnames(G) <- rownames(G) <- labels
      as(G,"graphNEL")
    }

## final object
skel <- new("pcAlgo", graph = Gobject, call = match.call(), n = integer(0),
    max.ord = as.integer(ord - 1), n.edgetests = n.edgetests,
    sepset = sepset, pMax = pMax, zMin = matrix(NA, 1, 1))

result = udag2pdag(skel)
t_end <- Sys.time()
estimation <- as.double(difftime(t_skel,t_start,unit="s"))
total <- as.double(difftime(t_end,t_start,unit="s"))
colorize_log('\033[32m',paste('Time Skeleton Estimation ', estimation))
colorize_log('\033[32m',paste('Time Total ',total))
graph_request <- store_graph_result(opt$api_host, result@'graph', result@'sepset', df, opt$job_id, "gaussCI", opt$send_sepsets, opt)
