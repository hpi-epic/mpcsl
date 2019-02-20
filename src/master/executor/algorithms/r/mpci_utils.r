library(httr, quietly = T)
library(graph, quietly = T)
library(jsonlite, quietly = T)
library(stringi, quietly = T)
library(causaleffect, quietly = T)
library(infotheo, quietly = T)
options(show.error.messages = FALSE)
options(error = function() {
    err <- stri_replace_all_regex(geterrmessage(), '\n', paste0('\n', ANSI_RED))
    colorize_log(ANSI_RED, err)
    colorize_log(ANSI_RED, 'Execution halted')
    quit(save='no', status=1, runLast=FALSE)
    })

ANSI_RED <- '\033[31m'
ANSI_GREEN <- '\033[32m'
ANSI_RESET <- '\033[0m\n'

colorize_log <- function(color, string) {
    # ANSI coloring
    cat(paste0(color, string, ANSI_RESET))
}

check_request <- function(api_host, request, job_id) {
    if (http_error(request)) {
        save(request, file=paste0('logs/job_', job_id, '_error.RData'))
        error_request <- PUT(paste0('http://', api_host, '/api/job/', job_id))
        warn_for_status(error_request)
        stop_for_status(request)
    }
}

get_dataset <- function(api_host, dataset_id, job_id) {
    url <- paste0('http://', api_host, '/api/dataset/', dataset_id, '/load')
    colorize_log(ANSI_GREEN, paste0('Load dataset from ', url))
    start_time <- Sys.time()
    df_request <- GET(url)
    check_request(api_host, df_request, job_id)
    colorize_log(ANSI_GREEN, paste('Successfully loaded dataset (size ', headers(df_request)$`x-content-length`,
                ' bytes) in', (Sys.time() - start_time), 'sec'))

    df <- read.csv(text=content(df_request, 'text'))
    return(df)
}

estimate_weight <- function(from_node, to_node, graph, df, continuous=FALSE) {
    disc_df <- if(regression) discretize(df) else df
    from_node_name <- colnames(df)[strtoi(from_node)]
    to_node_name <- colnames(df)[strtoi(to_node)]

    # expression <- causal.effect(to_node_name, from_node_name, G=igraph.from.graphNEL(graph), expr=FALSE)
    # parents <- sapply(unlist(inEdges(from_node, graph)), function(x) colnames(df)[strtoi(x)])

    # Only mutual information for now
    mi <- round(mutinformation(disc_df[[from_node_name]], disc_df[[to_node_name]]), digits = 4)
    # cmi <- condinformation(disc_df[[from_node_name]], disc_df[[to_node_name]], disc_df[parents])
    return mi
}

store_graph_result <- function(api_host, graph, sepsets, df, job_id, opt) {
    edges <- edges(graph)
    edge_list <- list(from_node=c(), to_node=c())
    node_list <- c()
    i <- 1
    for (node in names(edges)){
        node_list <- c(node_list, colnames(df)[strtoi(node)])
        for (edge in edges[[node]]){
            edge_list[['from_node']][[i]] <- colnames(df)[strtoi(node)]
            edge_list[['to_node']][[i]] <- colnames(df)[strtoi(edge)]

            weight <- estimate_weight(
                node, edge, graph, df,
                continuous=(opt$independence_test != 'binCI' && opt$independence_test != 'disCI')
            )
            edge_list[['weight']][[i]] <- weight

            i <- i + 1
        }
    }
    edge_list <- data.frame(edge_list)
    
    sepset_list <- list(from_node=c(), to_node=c(), statistic=c(), level=c())
    ss_nodes_list <- list()
    i <- 1
    for(from_node in 1:length(sepsets)){
        for(to_node in 1:length(sepsets)){
            sepset <- sapply(sepsets[[from_node]][[to_node]], (function (x) colnames(df)[x]))
            if(length(sepset) > 0){
                sepset_list[['from_node']][[i]] <- colnames(df)[from_node]
                sepset_list[['to_node']][[i]] <- colnames(df)[to_node]
                ss_nodes_list[[i]] <- if(length(sepset) > 1) sepset else list(sepset)
                sepset_list[['statistic']][[i]] <- 0
                sepset_list[['level']][[i]] <- length(sepset)
                i <- i + 1
            }
        }
    }
    sepset_list <- data.frame(sepset_list)
    sepset_list$nodes <- ss_nodes_list
    
    result_json <- jsonlite::toJSON(list(
        job_id=strtoi(job_id),
        node_list=node_list,
        edge_list=if(nrow(edge_list) == 0) list() else edge_list,
        meta_results=opt,
        sepset_list=if(nrow(sepset_list) == 0) list() else sepset_list
    ), auto_unbox=TRUE)
    
    graph_request <- POST(paste0('http://', api_host, '/api/job/', job_id, '/result'),
                                 body=result_json, 
                                 add_headers("Content-Type" = "application/json"))
    check_request(api_host, graph_request, job_id)
    colorize_log(ANSI_GREEN, paste0('Successfully executed job ', job_id))
    return(graph_request)
}
