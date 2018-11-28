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
from_list <- c()
to_list <- c()
node_list <- c()
for (node in names(edges)){
    node_list <- c(node_list, colnames(df)[strtoi(node)])
    for (edge in edges[[node]]){
        from_list <- c(from_list, colnames(df)[strtoi(node)])
        to_list <- c(to_list, colnames(df)[strtoi(edge)])
    }
}

edge_list <- data.frame(from_node=from_list, to_node=to_list)

result_json <- jsonlite::toJSON(list(
    job_id=strtoi(opt$job_id),
    node_list=node_list,
    edge_list=if(nrow(edge_list) == 0) list() else edge_list,
    meta_results=opt
), auto_unbox=TRUE)
# print(result_json)

graph_request <- POST('http://localhost:5000/results', body=result_json, add_headers("Content-Type" = "application/json"))
# print(graph_request$request)
