library(optparse)
library(httr)
library(pcalg)
library(graph)

option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-a", "--alpha"), type="double", default=0.05,
                                help="This is a hyperparameter", metavar=""),
                    make_option(c("-c", "--cores"), type="integer", default=1,
                                help="The number of cores to run the pc-algorithm on", metavar=""),
                    make_option(c("-fg", "--fixed_gaps"), type="character", default=FALSE,
                                help="The connections that are removed via prior knowledge", metavar=""),
                    make_option(c("-fe", "--fixed_edges"), type="character", default=FALSE,
                                help="The connections that are fixed via prior knowledge", metavar="")
);

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

df_request <- GET(paste0('http://localhost:5000/datasets/', opt$dataset_id))
df <- read.csv(text=content(df_request, 'text'))

matrix_df <- data.matrix(df)
sufficient_stats <- list(C=cor(matrix_df),n=nrow(matrix_df))
result = pc(suffStat=sufficient_stats, indepTest=gaussCItest, p=ncol(matrix_df),
            alpha=opt$alpha, numCores=opt$cores)
graph <- result@'graph'

edges <- edges(graph)
edge_list <- list()
for (node in names(edges)){
    for (edge in edges[[node]]){
        edge_list <- c(edge_list, list(from=colnames(df)[strtoi(node)], to=colnames(df)[strtoi(edge)]))
    }
}
result_json <- list(
    job_id=job_id,
    node_list=names(edges),
    edge_list=edge_list,
    meta_results=opt
)

graph_request <- POST('http://localhost:5000/results', encode='json', body=result_json)
