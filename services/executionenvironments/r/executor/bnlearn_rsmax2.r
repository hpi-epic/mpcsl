library(optparse, quietly = T)
library(bnlearn, quietly = T)
library(dplyr, quietly = T)
library(hash, quietly)
source("/scripts/mpci_utils.r")


option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("--api_host"), type="character",
                                help="API Host/Port", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("--restrict"), type="character",
                                help="The constraint-based or local search algorithm to be used in the “restrict” phase", metavar=""),
                    make_option(c("--maximize"), type="character",
                                help="The score-based algorithm to be used in the “maximize” phase", metavar=""),
                    make_option(c("--restrict_args"), type="character",
                                help="A list of arguments to be passed to the algorithm specified by restrict", metavar=""),
                    make_option(c("--maximize_args"), type="character",
                                help="A list of arguments to be passed to the algorithm specified by maximize", metavar=""),
                    make_option(c("--discrete_limit"), type="integer", default=4,
                                help="Maximum unique values per variable considered as discrete", metavar=""),
                    make_option(c("--sampling_factor"), type="double", default=1.0,
                                help="Data sampling factor to select a random subset, between 0 and 1", metavar=""),
                    make_option(c("--sampling_method"), type="character", default="random",
                                help="Data sampling method either random or otherwise top", metavar="")
); 

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

restrict_hash <- hash()
maximize_hash <- hash()
restrict_split <- strsplit(opt$restrict_args, ",")
maximize_split <- strsplit(opt$maximize_args, ",")

for(i in 1:length(restrict_split[[1]])){
    split_elem <- strsplit(restrict_split[[1]][i], "=")
    restrict_hash[[split_elem[[1]][1]]] = split_elem[[1]][2]
}

for(i in 1:length(maximize_split[[1]])){
    split_elem <- strsplit(maximize_split[[1]][i], "=")
    maximize_hash[[split_elem[[1]][1]]] = split_elem[[1]][2]
}

# General Parameters
verbose <- opt$verbose > 0

# Optional Constrained-Based Parameters
B <- NULL
max_sx <- NULL
undirected <- FALSE
if(opt$restrict != "pc.stable" && opt$restrict != "gs"){
    stop("Restriction Function not supported yet")
}else{
    if (!is.null(restrict_hash[["B"]])) B <- restrict_hash[["B"]];
    if (!is.null(restrict_hash[["max.sx"]])) max_sx <- restrict_hash[["max.sx"]];
    if (!is.null(restrict_hash[["undirected"]])) undirected <- restrict_hash[["undirected"]];
}
# Optional Score-Based Parameters
restart <- 0
perturb <- 1
max_iter <- Inf
maxp <- Inf
optimized <- TRUE 
if(opt$maximize != "hc"){
    stop("Maximize Function not supported yet")
}else{
    if (!is.null(maximize_hash[["restart"]])) restart <- maximize_hash[["restart"]]; 
    if (!is.null(maximize_hash[["perturb"]])) perturb <- maximize_hash[["perturb"]];
    if (!is.null(maximize_hash[["max.iter"]])) max_iter <- maximize_hash[["max.iter"]];
    if (!is.null(maximize_hash[["maxp"]])) maxp <- maximize_hash[["maxp"]];
    if (!is.null(maximize_hash[["optimized"]])) optimized <- maximize_hash[["optimized"]]; 
}

tmp_result <- get_dataset(opt$api_host, opt$dataset_id, opt$job_id, opt$sampling_method, opt$sampling_factor)
df <- tmp_result[[1]]
dataset_loading_time <- tmp_result[[2]]

if (restrict_hash[["test"]] == "mi-cg") {
	matrix_df <- df%>%dplyr::mutate_all(funs(if(length(unique(.))<opt$discrete_limit) as.factor(.)  else as.numeric(as.numeric(.))))
} else if ( restrict_hash[["test"]] == "cor" || restrict_hash[["test"]] == "zf" || 
            restrict_hash[["test"]] == "mi-g" ||restrict_hash[["test"]] == "mi-g-sh") {
    matrix_df <- df
} else if ( restrict_hash[["test"]] == "x2" || restrict_hash[["test"]] == "x2-adf" || 
            restrict_hash[["test"]] == "mi" || restrict_hash[["test"]] == "mi-adf" ||
            restrict_hash[["test"]] == "mi-sh") {
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

taken <- double()
start <- Sys.time()
if(opt$restrict == "pc.stable" && opt$maximize == "hc"){
    result = rsmax2(matrix_df, 
        restrict=opt$restrict, restrict.args=list(test=restrict_hash[["test"]], 
                                                  alpha=as.numeric(restrict_hash[["alpha"]]),
                                                  max.sx=as.numreric(max_sx)), 
        maximize=opt$maximize, maximize.args=list(score=maximize_hash[["score"]],
                                                  restart=as.numeric(restart),
                                                  perturb=as.numeric(perturb),
                                                  max.iter=as.numeric(max_iter),
                                                  maxp=as.numeric(maxp),
                                                  optimized=as.logical(optimized)),
        debug=verbose)
}
if(opt$restrict == "gs" && opt$maximize == "hc"){
    result = rsmax2(matrix_df, 
        restrict=opt$restrict, restrict.args=list(test=restrict_hash[["test"]], 
                                                  alpha=as.numeric(restrict_hash[["alpha"]]),
                                                  max.sx=as.numeric(max_sx),
                                                  B=as.numeric(B),
                                                  undirected=undirected), 
        maximize=opt$maximize, maximize.args=list(score=maximize_hash[["score"]],
                                                  restart=as.numeric(restart),
                                                  perturb=as.numeric(perturb),
                                                  max.iter=as.numeric(max_iter),
                                                  maxp=as.numeric(maxp),
                                                  optimized=as.logical(optimized)),
        debug=verbose)
}
end <- Sys.time()
taken <- as.double(difftime(end,start,unit="s"))
colorize_log('\033[32m',taken)

graph_request <- store_graph_result_bnlearn_hc(opt$api_host, result, df, opt$job_id, opt, taken, dataset_loading_time)