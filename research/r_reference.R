# Independent R check of the SAME saved input matrices, not R-generated fixtures.
# From repository root: Rscript research/r_reference.R
# Prerequisite: CCA 1.2.2 in the project library or an existing R library.

# Record a deterministic timezone without a sandbox-restricted system probe.
if (!nzchar(Sys.getenv("TZ"))) Sys.setenv(TZ = "UTC")

if (dir.exists(".r-library")) {
    .libPaths(c(normalizePath(".r-library"), .libPaths()))
}

if (!requireNamespace("CCA", quietly = TRUE)) {
    stop("Install the R CCA package first: install.packages('CCA')")
}
if (as.character(utils::packageVersion("CCA")) != "1.2.2") {
    stop("This audit targets CCA 1.2.2; review conventions before changing version")
}

read_matrix <- function(path) as.matrix(utils::read.csv(path, header = FALSE))

out <- "reports/stage1/r_reference"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
cases <- list(classical = c(0, 0), ridge_lowdim = c(0.3, 1.2),
              ridge_wide = c(0.3, 1.2))
for (case in names(cases)) {
    input <- file.path("statsmodels-checkout/statsmodels/multivariate/tests/results/ridge_cca", case)
    x <- read_matrix(file.path(input, "x.csv"))
    y <- read_matrix(file.path(input, "y.csv"))
    alpha <- cases[[case]]
    # No implicit scaling: these input fixtures use raw sample covariance.
    lambda_x <- alpha[1] * sum(diag(stats::cov(x))) / ncol(x)
    lambda_y <- alpha[2] * sum(diag(stats::cov(y))) / ncol(y)
    fit <- CCA::rcc(x, y, lambda1 = lambda_x, lambda2 = lambda_y)
    roots <- fit$cor[1:3]
    correlations <- diag(stats::cor(fit$scores$xscores[, 1:3],
                                  fit$scores$yscores[, 1:3]))
    # Committed, independently computed fixtures: pytest needs no R at runtime.
    utils::write.csv(data.frame(root = roots, train_correlation = correlations),
                     file.path(input, "r_results.csv"), row.names = FALSE)
    utils::write.table(fit$xcoef[, 1:3], file.path(input, "r_x_weights.csv"),
                       sep = ",", row.names = FALSE, col.names = FALSE)
    utils::write.table(fit$ycoef[, 1:3], file.path(input, "r_y_weights.csv"),
                       sep = ",", row.names = FALSE, col.names = FALSE)
    cat(case, "R reference generated\n")
}

# Export the original proposal's real dataset and an independent ridge reference.
data("nutrimouse", package = "CCA")
raw <- "data/raw/nutrimouse"
dir.create(raw, recursive = TRUE, showWarnings = FALSE)
x <- as.matrix(nutrimouse$gene)
y <- as.matrix(nutrimouse$lipid)
utils::write.table(x, file.path(raw, "gene.csv"), sep = ",", row.names = FALSE,
                   col.names = FALSE)
utils::write.table(y, file.path(raw, "lipid.csv"), sep = ",", row.names = FALSE,
                   col.names = FALSE)
utils::write.csv(data.frame(row_index = seq_len(nrow(x)) - 1L,
                           row_name = rownames(x), diet = nutrimouse$diet,
                           genotype = nutrimouse$genotype),
                 file.path(raw, "metadata.csv"), row.names = FALSE)
xs <- scale(x, center = TRUE, scale = TRUE)
ys <- scale(y, center = TRUE, scale = TRUE)
lambda_x <- sum(diag(stats::cov(xs))) / ncol(xs)
lambda_y <- sum(diag(stats::cov(ys))) / ncol(ys)
fit <- CCA::rcc(xs, ys, lambda1 = lambda_x, lambda2 = lambda_y)
correlations <- diag(stats::cor(fit$scores$xscores[, 1:3], fit$scores$yscores[, 1:3]))
utils::write.csv(data.frame(root = fit$cor[1:3], train_correlation = correlations),
                 file.path(out, "nutrimouse.csv"), row.names = FALSE)
utils::write.table(fit$xcoef[, 1:3], file.path(out, "nutrimouse_x_weights.csv"),
                   sep = ",", row.names = FALSE, col.names = FALSE)
utils::write.table(fit$ycoef[, 1:3], file.path(out, "nutrimouse_y_weights.csv"),
                   sep = ",", row.names = FALSE, col.names = FALSE)
utils::write.csv(installed.packages()[, c("Package", "Version")],
                 file.path(out, "packages.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
cat("Nutrimouse exported: ", nrow(x), " x (", ncol(x), ", ", ncol(y), ")\n", sep = "")
