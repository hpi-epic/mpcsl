library("pcalg")
library(optparse)
library(httr)
library(igraph)

tmpDataFile <- 'df.csv'
tmpGraphFile <- 'graph.gml'

option_list_v <- list(
  # optparse does not support mandatory arguments so I set a value to NA by default to verify later if it was provided.
  make_option("--apiHost", type="character", help="API Host/Port", default=NA),
  make_option("--uploadEndpoint", type = "character", help = "Dataset Upload Url", default = NA),
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

igraphDAG <- igraph.from.graphNEL(dag)
write_graph(igraphDAG, tmpGraphFile, "gml")

upload_dataset <- function(uploadEndpoint, apiHost) {
  url <- paste0(uploadEndpoint)
  response <- RETRY("PUT", url, body = list(file = upload_file(tmpDataFile)), encode = "multipart", times = 1, quiet=FALSE)
  stop_for_status(response)
  responseBody <- content(response)
  if (is.null(responseBody$id)){
    stop(paste0("Response did not contain dataset id", responseBody))
  }
  datasetId <- responseBody$id

  groundTruthEndpoint <- paste0("http://", apiHost, "/api/dataset/", datasetId, "/ground-truth")
  response <- RETRY(
    "POST",
    groundTruthEndpoint,
    body = list(graph_file = upload_file(tmpGraphFile)),
    encode = "multipart",
    times = 1,
    quiet=FALSE
  )
  stop_for_status(response)
}

upload_dataset(opt$uploadEndpoint, opt$apiHost)
