library(optparse, quietly = T)
library(pcalg, quietly = T)
library(dplyr)
library(bnlearn)
source("/scripts/mpci_utils.r")

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
                                help="Data sampling factor to select a random subset, between 0 and 1", metavar="")
                    #make_option(c("--fixed_gaps"), type="character", default=NULL,
                    #            help="The connections that are removed via prior knowledge", metavar=""),
                    #make_option(c("--fixed_edges"), type="character", default=NULL,
                    #            help="The connections that are fixed via prior knowledge", metavar=""),

);

discrete_node_limit <- 10
'%notin%' <- function(x,y)!('%in%'(x,y))
micgCItest <- function(x, y, S, suffStat) {
  if(x %notin% colnames(suffStat$dm)){
    print(paste("notin", x))
  }
  data_types <- sapply(suffStat$dm, class) # list of data types per columns

  #Determine test
  all_variabes <- as.character(append(list(x, y), S))
  found_types <- list()
  for (var in all_variabes) {
    found_types <- append(data_types[var], found_types)
  }
  distinct_types <- sort(unlist(unique(found_types), use.names=FALSE))
  types_string <- paste(distinct_types, collapse = '_')
  ci_test <- switch(
    types_string,
    "numeric" = "cor", # only continuous values
    "factor" = "x2", # only discrete values
    "factor_numeric" = "mi-cg" # mixed values
  )

  # Execute test
  htest <- ci.test(
    as.character(x),
    as.character(y),
    as.character(S),
    suffStat$dm,
    ci_test
  )

  #return p-value of test
  htest$"p.value"
}

indepTestDict <- list(gaussCI=gaussCItest, binCI=binCItest, disCI=disCItest, micg=micgCItest)

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

tmp_result <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id, opt$sampling_factor)
df <- tmp_result[[1]]
dataset_loading_time <- tmp_result[[2]]
matrix <- data.matrix(df)
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
} else if (opt$independence_test == "micg") {
    # convert datatypes numeric (continuous) -> if exceed discrete node_limit else factor (discrete)
    matrix_df <- df %>%
      dplyr::mutate_all(
        funs(
          if(length(unique(.)) < discrete_node_limit)
            as.factor(.)
          else as.numeric(as.numeric(.))
          )
        ) %>%
      as.data.frame()

    # increase column names by one because R starts counting at 1
    colnames(matrix_df) <- as.character(as.numeric(colnames(matrix_df)) + 1)
    sufficient_stats <- list(dm = matrix_df)
}else {
    stop("No valid independence test specified")
}
subset_size <- if(opt$subset_size < 0) Inf else opt$subset_size
verbose <- opt$verbose > 0
start <- Sys.time()
result = pc(suffStat=sufficient_stats, verbose=verbose,
            indepTest=indepTestDict[[opt$independence_test]], m.max=subset_size,
            p=ncol(matrix), alpha=opt$alpha, numCores=opt$cores, skel.method=opt$skeleton_method)
end <- Sys.time()
taken <- as.double(difftime(end,start,unit="s"))
colorize_log('\033[32m',taken)
graph_request <- store_graph_result(opt$api_host, result@'graph', result@'sepset', df, opt$job_id, opt$independence_test, opt$send_sepsets, opt, taken, dataset_loading_time)
