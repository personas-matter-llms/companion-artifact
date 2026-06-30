suppressPackageStartupMessages(library(lme4))

MODEL_LABELS <- c(
  "qwen2_5_1_5b" = "Qwen 1.5B",
  "llama_3_1_8b" = "Llama 8B",
  "mistral_small_3_24b_awq" = "Mistral 24B",
  "qwen2.5-32b-awq" = "Qwen 32B"
)

EMOTION_LABELS <- c(
  "emotionanger" = "Anger vs happiness",
  "emotiondisgust" = "Disgust vs happiness",
  "emotionfear" = "Fear vs happiness",
  "emotionneutral" = "Neutral vs happiness",
  "emotionsadness" = "Sadness vs happiness"
)

TRAIT_LABELS <- c(
  "conscientiousnessH" = "Conscientiousness",
  "opennessH" = "Openness",
  "extraversionH" = "Extraversion"
)

scope_label <- function(scope) {
  if (scope == "overall") {
    return("Overall")
  }
  if (startsWith(scope, "model:")) {
    model_key <- sub("^model:", "", scope)
    if (model_key %in% names(MODEL_LABELS)) {
      return(paste("Model:", MODEL_LABELS[[model_key]]))
    }
    return(paste("Model:", model_key))
  }
  scope
}

write_result <- function(rows, scope, result_type, effect, comparison, reference = "",
                         chi_square = NA, df = NA, estimate = NA, se = NA, z = NA,
                         odds_ratio = NA, p_value = NA, bh_family = "", note = "") {
  rows[[length(rows) + 1]] <- data.frame(
    analysis_scope = scope_label(scope),
    result_type = result_type,
    effect = effect,
    comparison = comparison,
    reference = reference,
    chi_square = chi_square,
    df = df,
    log_odds_estimate = estimate,
    std_error = se,
    z_value = z,
    odds_ratio = odds_ratio,
    p_value = p_value,
    q_value_bh = NA_real_,
    bh_family = bh_family,
    note = note,
    stringsAsFactors = FALSE
  )
  rows
}

fit_glmm <- function(data, include_model = TRUE) {
  data$model <- factor(data$model)
  data$emotion <- relevel(factor(data$emotion), ref = "happiness")
  data$conscientiousness <- relevel(factor(data$conscientiousness), ref = "L")
  data$openness <- relevel(factor(data$openness), ref = "L")
  data$extraversion <- relevel(factor(data$extraversion), ref = "L")
  data$question_id <- factor(data$question_id)

  ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  model_term <- if (include_model) "model + " else ""
  f <- function(rhs) as.formula(paste("outcome ~", model_term, rhs, "+ (1 | question_id)"))

  full <- glmer(
    f("emotion + conscientiousness + openness + extraversion"),
    data = data,
    family = binomial,
    control = ctrl
  )
  no_emotion <- glmer(
    f("conscientiousness + openness + extraversion"),
    data = data,
    family = binomial,
    control = ctrl
  )
  no_c <- glmer(
    f("emotion + openness + extraversion"),
    data = data,
    family = binomial,
    control = ctrl
  )
  no_o <- glmer(
    f("emotion + conscientiousness + extraversion"),
    data = data,
    family = binomial,
    control = ctrl
  )
  no_e <- glmer(
    f("emotion + conscientiousness + openness"),
    data = data,
    family = binomial,
    control = ctrl
  )

  list(full = full, no_emotion = no_emotion, no_c = no_c, no_o = no_o, no_e = no_e)
}

add_lrt_row <- function(rows, scope, effect, reduced, full) {
  a <- anova(reduced, full, test = "Chisq")
  write_result(
    rows,
    scope,
    "Main effect",
    effect,
    "Overall effect",
    chi_square = a$Chisq[2],
    df = a$Df[2],
    p_value = a$`Pr(>Chisq)`[2],
    bh_family = "main_effects"
  )
}

add_contrast_rows <- function(rows, scope, model) {
  co <- coef(summary(model))
  for (name in rownames(co)) {
    if (name %in% names(EMOTION_LABELS)) {
      rows <- write_result(
        rows,
        scope,
        "Emotion contrast",
        "Emotion",
        EMOTION_LABELS[[name]],
        reference = "Happiness",
        estimate = co[name, "Estimate"],
        se = co[name, "Std. Error"],
        z = co[name, "z value"],
        p_value = co[name, "Pr(>|z|)"],
        odds_ratio = exp(co[name, "Estimate"]),
        bh_family = "planned_contrasts"
      )
    } else if (name %in% names(TRAIT_LABELS)) {
      rows <- write_result(
        rows,
        scope,
        "Trait contrast",
        TRAIT_LABELS[[name]],
        "High vs low",
        reference = "Low",
        estimate = co[name, "Estimate"],
        se = co[name, "Std. Error"],
        z = co[name, "z value"],
        p_value = co[name, "Pr(>|z|)"],
        odds_ratio = exp(co[name, "Estimate"]),
        bh_family = "planned_contrasts"
      )
    }
  }
  rows
}

add_bh <- function(results) {
  results$q_value_bh <- NA_real_
  for (scope in unique(results$analysis_scope)) {
    for (family in c("main_effects", "planned_contrasts")) {
      idx <- which(results$analysis_scope == scope & results$bh_family == family)
      idx <- idx[!is.na(results$p_value[idx])]
      if (length(idx) > 0) {
        results$q_value_bh[idx] <- p.adjust(results$p_value[idx], method = "BH")
      }
    }
  }
  results
}

run_scope <- function(data, scope, include_model = TRUE) {
  if (sum(data$outcome) == 0 || sum(data$outcome) == nrow(data)) {
    message("Skipping ", scope_label(scope), ": outcome has no variation.")
    return(NULL)
  }

  fits <- fit_glmm(data, include_model = include_model)
  rows <- list()
  rows <- add_lrt_row(rows, scope, "Emotion", fits$no_emotion, fits$full)
  rows <- add_lrt_row(rows, scope, "Conscientiousness", fits$no_c, fits$full)
  rows <- add_lrt_row(rows, scope, "Openness", fits$no_o, fits$full)
  rows <- add_lrt_row(rows, scope, "Extraversion", fits$no_e, fits$full)
  rows <- add_contrast_rows(rows, scope, fits$full)
  add_bh(do.call(rbind, rows))
}

run_binary_glmm <- function(input, output, outcome) {
  data <- read.csv(input, stringsAsFactors = FALSE)
  required <- c(
    "model", "assignment", "question_id", "emotion", "personality",
    "conscientiousness", "openness", "extraversion", outcome
  )
  missing <- setdiff(required, names(data))
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }

  shared <- subset(
    data,
    assignment == "shared54" &
      conscientiousness != "N" &
      openness != "N" &
      extraversion != "N"
  )
  shared$outcome <- as.integer(shared[[outcome]])

  overall <- run_scope(shared, "overall", include_model = TRUE)

  by_model <- list()
  for (current_model in sort(unique(shared$model))) {
    by_model[[length(by_model) + 1]] <- run_scope(
      subset(shared, model == current_model),
      paste("model", current_model, sep = ":"),
      include_model = FALSE
    )
  }
  by_model <- do.call(rbind, Filter(Negate(is.null), by_model))
  combined <- do.call(rbind, Filter(Negate(is.null), list(overall, by_model)))

  dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
  by_model_output <- sub("\\.csv$", "_by_model.csv", output)
  combined_output <- sub("\\.csv$", "_combined.csv", output)

  write.csv(overall, output, row.names = FALSE, quote = FALSE, na = "NA")
  write.csv(by_model, by_model_output, row.names = FALSE, quote = FALSE, na = "NA")
  write.csv(combined, combined_output, row.names = FALSE, quote = FALSE, na = "NA")

  cat("input=", input, "\n", sep = "")
  cat("output=", output, "\n", sep = "")
  cat("outcome=", outcome, "\n", sep = "")
  cat("by_model_output=", by_model_output, "\n", sep = "")
  cat("combined_output=", combined_output, "\n", sep = "")
  cat("rows=", nrow(shared), "\n", sep = "")
  cat("positive=", sum(shared$outcome), "\n", sep = "")
}
