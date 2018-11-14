library("optparse")

# TODO: Read Command line arguments and hyperparameters

option_list_v <- list(make_option(c("-a", "--alpha"), type="double", default=0,
                                help="This is a hyperparameter", metavar=""),
                    make_option(c("-c", "--cores"), type="integer", default=1,
                                help="The number of cores to run the pc-algorithm on", metavar=""),
                    make_option(c("-fg", "--fixed-gaps"), type="character", default="",
                                help="The connections that are removed via prior knowledge", metavar=""),
                    make_option(c("-fe", "--fixed-edges"), type="character", default="",
                                help="The connections that are fixed via prior knowledge", metavar="")
);

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

# TODO: Load Relevant data

# TODO: Bring into relavant format

# result_graph <- pc()

# TODO: Write Result Graph to endpoint