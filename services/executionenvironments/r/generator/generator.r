library("pcalg")
library(optparse)
library(httr)

tmpDataFile <- 'df.csv'

option_list_v <- list(
  # optparse does not support mandatory arguments so I set a value to NA by default to verify later if it was provided.
  make_option("--uploadEndpoint", type = "character", help = "API Host/Port", default = NA),
  make_option("--nSamples", type = "integer", default = NA, help = "number of samples to be generated"),
  make_option("--nNodes", type = "integer", default = NA, help = "number of variables"),
  make_option("--edgeProbability", type = "double", default = NA, help = "probability that a given edge is in the graph"),
  make_option("--edgeValueLowerBound", type = "double", default = NA, help = "lowest possible edge value"),
  make_option("--edgeValueUpperBound", type = "double", default = NA, help = "highest possible edge value")
)

option_parser <- OptionParser(option_list = option_list_v)
opt <- parse_args(option_parser)

for (name in names(opt)) {
  if(is.na(opt[[name]])){
    stop(paste0("Paramater --", name, " is required"))
  }
}

dag <- randomDAG(opt$nNodes, opt$edgeProbability, opt$edgeValueLowerBound, opt$edgeValueUpperBound)
dataset <- rmvDAG(opt$nSamples,dag)
write.csv(dataset, tmpDataFile)


put_dataset <- function(uploadEndpoint) {
    url <- paste0(uploadEndpoint)
    df_request <- RETRY("PUT", url, body = list(file = upload_file(tmpDataFile)), encode = "multipart", times = 1, quiet=FALSE)
    return(df_request)
}

put_dataset(opt$uploadEndpoint)
#post_ground_truth()
#igraphDAG <- igraph.from.graphNEL(dag)
#> write_graph(igraphDAG, "graph.gml", "gml")
