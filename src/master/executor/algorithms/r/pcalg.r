library(optparse)
library(httr)
library(pcalg)
library(graph)

# TODO: Read Command line arguments and hyperparameters

option_list_v <- list(
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
print(content(df_request, 'text'))
df <- read.csv(text=content(df_request, 'text'))

matrix_df <- data.matrix(df)
# print(matrix_df)
sufficient_stats <- list(C=cor(matrix_df),n=nrow(matrix_df))
result = pc(suffStat=sufficient_stats, indepTest=gaussCItest, p=ncol(matrix_df),
            alpha=opt$alpha, numCores=opt$cores)
graph <- result@'graph'

edges <- edges(graph)
from <- c()
to <- c()
for (node in names(edges)){
    for (edge in edges[[node]]){
        from <- c(from, c(colnames(df)[strtoi(node)]))
        to <- c(to, colnames(df)[strtoi(edge)])
    }
}
edge_df = data.frame(start_node=from, end_node=to)
write.csv(edge_df, row.names=FALSE)

graph_request <- POST('http://localhost:5000/results', encode='json', body=list(result=capture.output(write.csv(edge_df, row.names=FALSE))))
# print(content(graph_request, 'text'))
