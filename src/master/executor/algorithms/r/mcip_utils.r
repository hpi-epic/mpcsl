library(httr, quietly = T)
library(graph, quietly = T)
library(jsonlite, quietly = T)


get_dataset <- function(api_host, dataset_id) {
    df_request <- GET(paste0('http://', api_host, '/api/dataset/', dataset_id, '/load'))
    # print(df_request)
    df <- read.csv(text=content(df_request, 'text'))
    return(df)
}

store_graph_result <- function(api_host, graph, df, job_id, opt) {
    edges <- edges(graph)
    edge_list <- list(from_node=c(), to_node=c())
    node_list <- c()
    i <- 1
    for (node in names(edges)){
        node_list <- c(node_list, colnames(df)[strtoi(node)])
        for (edge in edges[[node]]){
            edge_list[['from_node']][[i]] <- colnames(df)[strtoi(node)]
            edge_list[['to_node']][[i]] <- colnames(df)[strtoi(edge)]
            i <- i + 1
        }
    }
    edge_list <- data.frame(edge_list)
    
    sepsets <- result@'sepset'
    sepset_list <- list(from_node=c(), to_node=c(), statistic=c(), level=c())
    ss_nodes_list <- list()
    i <- 1
    for(from_node in 1:length(sepsets)){
        for(to_node in 1:length(sepsets)){
            sepset <- sapply(sepsets[[from_node]][[to_node]], (function (x) colnames(df)[x]))
            if(length(sepset) > 0){
                sepset_list[['from_node']][[i]] <- colnames(df)[from_node]
                sepset_list[['to_node']][[i]] <- colnames(df)[to_node]
                ss_nodes_list[[i]] <- list(sepset)
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
    # print(result_json)
    
    graph_request <- POST(paste0('http://', api_host, '/api/job/', job_id, '/result'),
                                 body=result_json, 
                                 add_headers("Content-Type" = "application/json"))

    return(graph_request)
}
