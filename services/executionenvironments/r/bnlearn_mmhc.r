library(optparse, quietly = T)
library(bnlearn, quietly = T)
library(dplyr, quietly = T)
source("/scripts/mpci_utils.r")


option_list_v <- list(
                    make_option(c("-j", "--job_id"), type="character",
                                help="Job ID", metavar=""),
                    make_option(c("--api_host"), type="character",
                                help="API Host/Port", metavar=""),
                    make_option(c("-d", "--dataset_id"), type="character",
                                help="Dataset ID", metavar=""),
                    make_option(c("-v", "--verbose"), type="integer", default=0,
                                help="More detailed output is provided", metavar=""),
                    make_option(c("--send_sepsets"), type="integer", default=0,
                                help="If 1, sepsets will be sent with the results", metavar=""),
                    make_option(c("--restrict"), type="character",
                                help="The constraint-based or local search algorithm to be used in the “restrict” phase", metavar=""),
                    make_option(c("--maximize"), type="character",
                                help="The score-based algorithm to be used in the “maximize” phase", metavar=""),
                    make_option(c("--restrict_args"), type="character",
                                help="A list of arguments to be passed to the algorithm specified by restrict", metavar=""),
                    make_option(c("--maximize_args"), type="character",
                                help="A list of arguments to be passed to the algorithm specified by maximize", metavar="")
); 

option_parser <- OptionParser(option_list=option_list_v)
opt <- parse_args(option_parser)

print(opt$restrict)
print(opt$maximize)
print(opt$restrict_args)
print(opt$maximize_args)

# Force job to fail with this instruction
hc(matrix_df, score=opt$score, debug=verbose, restart=opt$restart, perturb=opt$perturb, max.iter=maxiter_value, maxp=maxp_value, optimized=optimized)
