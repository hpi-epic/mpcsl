library(bnlearn)
library(pcalg)
library(igraph)
library(dplyr)

discrete_node_limit <- 4

micgCallback <- function(x, y, S, suffStat) {
  data_types <- sapply(df, class)
  concerned_variables <- as.character(append(list(x, y), S))
  found_types <- list()
  for (var in concerned_variables) {
    found_types <- append(data_types[var], found_types)
  }
  distinct_types <- sort(unlist(unique(found_types), use.names=FALSE))
  types_string <- paste(distinct_types, collapse = '_')
  ci_test <- switch(
  types_string,
  "numeric" = "cor",
  "factor" = "x2",
  "factor_numeric" = "mi-cg"
  )
  htest <- ci.test(as.character(x), as.character(y), as.character(S), suffStat, ci_test)
  htest$"p.value"
}

input_graph_file <- "/home/jonas/Code/mpci2/services/executionenvironments/r/executor/ground_truth.gml"

graph <- read.graph(input_graph_file, format="gml")
graph_nel <- igraph.to.graphNEL(graph)

# df <- rmvDAG(nSamples,graph_nel)
dataset_file <- "/home/jonas/Code/mpci2/services/executionenvironments/r/executor/fuer_jonas (1).csv"
df <- read.csv(dataset_file, header=TRUE, check.names=FALSE)
matrix <- data.matrix(df)
matrix <- df%>%dplyr::mutate_all(funs(if(length(unique(.))<discrete_node_limit) as.factor(.)  else as.numeric(as.numeric(.))))
df <- as.data.frame(matrix)
colnames(df) <- as.character(as.numeric(colnames(df)) + 1)

subset_size <- Inf
verbose <- TRUE
result = pc(suffStat=df, verbose=verbose,
            indepTest=micgCallback, m.max=subset_size,
            p=ncol(matrix), alpha=0.05, numCores=1, skel.method="stable.fast")
graph <- result@'graph'
igraphDAG <- igraph.from.graphNEL(graph)
tmpGraphFile <- "/home/jonas/Code/mpci2/services/executionenvironments/r/generator/out.gml"
write_graph(igraphDAG, tmpGraphFile, "gml")
