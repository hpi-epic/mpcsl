library(optparse, quietly = T)
library(httr, quietly = T)
library(pcalg, quietly = T)
library(graph, quietly = T)
library(jsonlite, quietly = T)

option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-t", "--independence_test"), type="character", default="gaussCI",
                                help="Independence test used for the pcalg", metavar=""),
                    make_option(c("-a", "--alpha"), type="double", default=0.05,
                                help="This is a hyperparameter", metavar=""),
                    make_option(c("-c", "--cores"), type="integer", default=1,
                                help="The number of cores to run the pc-algorithm on", metavar=""),
                    make_option(c("-fg", "--fixed_gaps"), type="character", default=FALSE,
                                help="The connections that are removed via prior knowledge", metavar=""),
                    make_option(c("-fe", "--fixed_edges"), type="character", default=FALSE,
                                help="The connections that are fixed via prior knowledge", metavar="")
);

indepTestDict <- list(gaussCI=gaussCItest, binCI=binCItest, disCI=disCItest)

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

print(opt$independence_test)

df_request <- GET(paste0('http://localhost:5000/dataset/', opt$dataset_id, '/load'))
df <- read.csv(text=content(df_request, 'text'))

matrix_df <- data.matrix(df)
sufficient_stats <- list(C=cor(matrix_df),n=nrow(matrix_df))
result = pc(suffStat=sufficient_stats,
            indepTest=indepTestDict[[opt$independence_test]],
            p=ncol(matrix_df), alpha=opt$alpha, numCores=opt$cores)
graph <- result@'graph'

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
    job_id=strtoi(opt$job_id),
    node_list=node_list,
    edge_list=if(nrow(edge_list) == 0) list() else edge_list,
    meta_results=opt,
    sepset_list=if(nrow(sepset_list) == 0) list() else sepset_list
), auto_unbox=TRUE)
# print(result_json)

graph_request <- POST(paste0('http://localhost:5000/job/', opt$job_id, '/result'), body=result_json, add_headers("Content-Type" = "application/json"))
# print(graph_request$request)
