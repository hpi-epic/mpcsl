library(bnlearn)
library(pcalg)
library(igraph)
library(dplyr)


input_graph_file <- "/home/jonas/Code/mpci2/services/executionenvironments/r/executor/ground_truth (1).gml"

graph <- read.graph(input_graph_file, format="gml")
graph_nel <- igraph.to.graphNEL(graph)

# df <- rmvDAG(nSamples,graph_nel)
dataset_file <- "/home/jonas/Code/mpci2/services/executionenvironments/r/executor/fuer_jonas (2).csv"
df <- read.csv(dataset_file, header=TRUE, check.names=FALSE)
matrix <- data.matrix(df)
matrix <- df%>%dplyr::mutate_all(funs(if(length(unique(.))<discrete_node_limit) as.factor(.)  else as.numeric(as.numeric(.))))
df <- as.data.frame(matrix)
colnames(df) <- as.character(as.numeric(colnames(df)) + 1)

subset_size <- Inf
verbose <- TRUE
result = pc(suffStat=df, verbose=verbose,
            indepTest=, m.max=subset_size,
            p=ncol(matrix), alpha=0.05, numCores=1, skel.method="stable.fast")
graph <- result@'graph'
igraphDAG <- igraph.from.graphNEL(graph)
tmpGraphFile <- "/home/jonas/Code/mpci2/services/executionenvironments/r/executor/out2.gml"
write_graph(igraphDAG, tmpGraphFile, "gml")
