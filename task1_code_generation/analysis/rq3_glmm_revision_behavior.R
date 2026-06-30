this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
script_dir <- dirname(normalizePath(this_file))
root_dir <- normalizePath(file.path(script_dir, ".."))

source(file.path(script_dir, "glmm_common.R"))

args <- commandArgs(trailingOnly = TRUE)
input <- if (length(args) >= 1) args[[1]] else file.path(root_dir, "data", "shared54_pass1_instances.csv")
output_dir <- if (length(args) >= 2) args[[2]] else file.path(root_dir, "results", "csv")

run_binary_glmm(
  input,
  file.path(output_dir, "rq3_glmm_revised_results.csv"),
  "revised"
)

overrevised_data <- read.csv(input, stringsAsFactors = FALSE)
overrevised_data <- subset(overrevised_data, model != "qwen2_5_1_5b")
overrevised_input <- tempfile(fileext = ".csv")
write.csv(overrevised_data, overrevised_input, row.names = FALSE, quote = FALSE, na = "NA")
on.exit(unlink(overrevised_input), add = TRUE)

run_binary_glmm(
  overrevised_input,
  file.path(output_dir, "rq3_glmm_overrevised_results.csv"),
  "overrevised"
)
