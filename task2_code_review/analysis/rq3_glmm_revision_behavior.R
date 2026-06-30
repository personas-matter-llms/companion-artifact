this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
script_dir <- dirname(normalizePath(this_file))
root_dir <- normalizePath(file.path(script_dir, ".."))

source(file.path(script_dir, "glmm_common.R"))

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else file.path(root_dir, "data", "shared54_bleu_instances.csv")
output <- if (length(args) >= 2) args[[2]] else file.path(root_dir, "results", "csv", "rq3_glmm_revision_results.csv")

run_binary_glmm(input, output, "revised")
