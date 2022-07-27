library(httr, quietly = T)
library(graph, quietly = T)
library(jsonlite, quietly = T)
library(stringi, quietly = T)
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
        save(request, file=paste0('/logs/job_', job_id, '_error.RData'))
        error_request <- RETRY("PUT", paste0('http://', api_host, '/api/job/', job_id), times = 5, quiet=FALSE)
        warn_for_status(error_request)
        stop_for_status(request)
    }
}

get_dataset <- function(api_host, dataset_id, job_id, sampling_factor=1) {
    url <- paste0('http://', api_host, '/api/dataset/', dataset_id, '/loadwithids')
    colorize_log(ANSI_GREEN, paste0('Load dataset from ', url))
    start_time <- Sys.time()
    df_request <- RETRY("GET", url, times = 5, quiet=FALSE)
    check_request(api_host, df_request, job_id)
    dataset_loading_time <- as.double(difftime(Sys.time(),start_time,unit="s"))
    colorize_log(ANSI_GREEN, paste('Successfully loaded dataset (size ', headers(df_request)$`x-content-length`,
                ' bytes) in', dataset_loading_time))

    df <- read.csv(text=content(df_request, 'text'), header=TRUE, check.names=FALSE)
    df <- df[sample(nrow(df), nrow(df) * sampling_factor), ]
    return(list(df,dataset_loading_time))
}

estimate_weight <- function(from_node, to_node, graph, df, continuous=FALSE) {
    disc_df <- if(continuous) discretize(df) else df
    from_node_name <- colnames(df)[strtoi(from_node)]
    to_node_name <- colnames(df)[strtoi(to_node)]

    # expression <- causal.effect(to_node_name, from_node_name, G=igraph.from.graphNEL(graph), expr=FALSE)
    # parents <- sapply(unlist(inEdges(from_node, graph)), function(x) colnames(df)[strtoi(x)])

    # Only mutual information for now
    mi <- round(mutinformation(disc_df[[from_node_name]], disc_df[[to_node_name]]), digits = 4)
    # cmi <- condinformation(disc_df[[from_node_name]], disc_df[[to_node_name]], disc_df[parents])
    return(mi)
}

store_graph_result <- function(api_host, graph, sepsets, df, job_id, independence_test, send_sepsets, meta_results, execution_time, dataset_loading_time) {
    edges <- edges(graph)
    edge_list <- list(from_node=c(), to_node=c())
    i <- 1
    for (node in names(edges)){
        for (edge in edges[[node]]){
            edge_list[['from_node']] <- c(edge_list[['from_node']],as.numeric(colnames(df)[strtoi(node)]))
            edge_list[['to_node']] <- c(edge_list[['to_node']],as.numeric(colnames(df)[strtoi(edge)]))

            weight <- estimate_weight(
                node, edge, graph, df,
                continuous=(independence_test != 'binCI' && independence_test != 'disCI')
            )
            edge_list[['weight']] <- c(edge_list[['weight']],weight)

            i <- i + 1
        }
    }
    edge_list <- data.frame(edge_list)

    sepset_list <- list(from_node=c(), to_node=c(), statistic=c(), level=c())

    if (send_sepsets > 0) {
        ss_nodes_list <- list()
        i <- 1
        for(from_node in 1:length(sepsets)){
            for(to_node in 1:length(sepsets)){
                sepset <- sapply(sepsets[[from_node]][[to_node]], (function (x) colnames(df)[x]))
                if(length(sepset) > 0){
                    sepset_list[['from_node']] <- c(sepset_list[['from_node']],as.numeric(colnames(df)[from_node]))
                    sepset_list[['to_node']] <- c(sepset_list[['to_node']],as.numeric(colnames(df)[to_node]))
                    ss_nodes_list[[i]] <- if(length(sepset) > 1) sepset else list(sepset)
                    sepset_list[['statistic']] <- c(sepset_list[['statistic']],0)
                    sepset_list[['level']] <- c(sepset_list[['level']],length(sepset))
                    i <- i + 1
                }
            }
        }
    }
    sepset_list <- data.frame(sepset_list)

    if (send_sepsets > 0) {
        sepset_list$nodes <- ss_nodes_list
    }
    
    result_json <- jsonlite::toJSON(list(
        job_id=strtoi(job_id),
        edge_list=if(nrow(edge_list) == 0) list() else edge_list,
        meta_results=meta_results,
        sepset_list=if(nrow(sepset_list) == 0) list() else sepset_list,
        execution_time =execution_time,
        dataset_loading_time=dataset_loading_time
    ), auto_unbox=TRUE)
    
    graph_request <- RETRY("POST", paste0('http://', api_host, '/api/job/', job_id, '/result'),
                                 body=result_json, 
                                 add_headers("Content-Type" = "application/json"), times = 5, quiet=FALSE)
    check_request(api_host, graph_request, job_id)
    colorize_log(ANSI_GREEN, paste0('Successfully executed job ', job_id))
    return(graph_request)
}

store_graph_result_bn <- function(api_host, bn_result, df, job_id, independence_test, meta_results, execution_time, dataset_loading_time) {
    edge_list <- list(from_node=c(), to_node=c())
    
    if (length(result$'arcs'[,1]) != 0) {
        for (i in 1:(length(result$'arcs'[,1])) ){
            edge_list[['from_node']] <- c(edge_list[['from_node']],as.numeric(result$'arcs'[i,][1]))
            edge_list[['to_node']] <- c(edge_list[['to_node']],as.numeric(result$'arcs'[i,][2]))
            edge_list[['weight']] <- c(edge_list[['weight']],0)
        }
    }
    edge_list <- data.frame(edge_list)

    
    result_json <- jsonlite::toJSON(list(
        job_id=strtoi(job_id),
        edge_list=if(nrow(edge_list) == 0) list() else edge_list,
        meta_results=meta_results,
        sepset_list=list(),
        execution_time =execution_time,
        dataset_loading_time=dataset_loading_time
    ), auto_unbox=TRUE)

    graph_request <- RETRY("POST", paste0('http://', api_host, '/api/job/', job_id, '/result'),
                                 body=result_json, 
                                 add_headers("Content-Type" = "application/json"), times = 5, quiet=FALSE)
    check_request(api_host, graph_request, job_id)
    colorize_log(ANSI_GREEN, paste0('Successfully executed job ', job_id))
    return(graph_request)
}

store_graph_result_bnlearn_hc <- function(api_host, bn_result, df, job_id, meta_results, execution_time, dataset_loading_time){
    edge_list <- list(from_node=c(), to_node=c())

    if (length(result$'arcs'[,1]) != 0) {
        for (i in 1:(length(result$'arcs'[,1])) ){
            edge_list[['from_node']] <- c(edge_list[['from_node']],as.numeric(result$'arcs'[i,][1]))
            edge_list[['to_node']] <- c(edge_list[['to_node']],as.numeric(result$'arcs'[i,][2]))
            edge_list[['weight']] <- c(edge_list[['weight']],0)
        }
    }
    edge_list <- data.frame(edge_list)

    result_json <- jsonlite::toJSON(list(
        job_id=strtoi(job_id),
        edge_list=if(nrow(edge_list) == 0) list() else edge_list,
        meta_results=meta_results,
        sepset_list=list(),
        execution_time =execution_time,
        dataset_loading_time=dataset_loading_time
    ), auto_unbox=TRUE)

    graph_request <- RETRY("POST", paste0('http://', api_host, '/api/job/', job_id, '/result'),
                                 body=result_json, 
                                 add_headers("Content-Type" = "application/json"), times = 5, quiet=FALSE)
    check_request(api_host, graph_request, job_id)
    colorize_log(ANSI_GREEN, paste0('Successfully executed job ', job_id))
    return(graph_request)
}
