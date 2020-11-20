library("pcalg")
library(optparse)
library(httr)

tmpDataFile <- 'df.csv'

option_list_v <- list(
  # optparse does not support mandatory arguments so I set api_host to NA by default to verify later if it was provided.
  make_option("--apiHost", type = "character", help = "API Host/Port", default = NA),
  make_option("--nSamples", type = "integer", default = 100, help = "number of samples to be generated"),
  make_option("--nNodes", type = "integer", default = 5, help = "number of variables"),
  make_option("--edgeProbability", type = "double", default = 0.5, help = "probability that a given edge is in the graph"),
  make_option("--edgeValueLowerBound", type = "double", default = -1, help = "lowest possible edge value"),
  make_option("--edgeValueUpperBound", type = "double", default = 1, help = "highest possible edge value")
)

option_parser <- OptionParser(option_list = option_list_v)
opt <- parse_args(option_parser)

if (is.na(opt$apiHost)) {
  stop("Argument --apiHost is required")
}

dag <- randomDAG(opt$nNodes, opt$edgeProbability, opt$edgeValueLowerBound, opt$edgeValueUpperBound)
dataset <- rmvDAG(opt$nSamples,dag)
write.csv(dataset, tmpDataFile)


post_dataset <- function(apiHost) {
    url <- paste0('http://', apiHost, '/api/dataset_csv_upload')
    df_request <- RETRY("POST", url, body = list(file = upload_file(tmpDataFile)), encode = "multipart", times = 1, quiet=FALSE)
    return(df_request)
}

post_dataset(opt$apiHost)


