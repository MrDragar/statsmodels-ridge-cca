# Persistent independent R worker for CCA::rcc/cc and PMA::CCA comparisons.
# JSON requests arrive on stdin; stdout contains one JSON response per request.
Sys.setenv(TZ = "UTC")
RNGkind("Mersenne-Twister", "Inversion", "Rejection")
.libPaths(c(normalizePath(".r-library"), .libPaths()))
stopifnot(requireNamespace("CCA"), requireNamespace("PMA"), requireNamespace("jsonlite"))

paired_cor <- function(x, y) {
    result <- diag(cor(x, y))
    if (any(!is.finite(result))) stop("Undefined score correlation")
    result
}
preprocess <- function(x, y) {
    mx <- colMeans(x); my <- colMeans(y)
    sx <- apply(x, 2, sd); sy <- apply(y, 2, sd)
    sx[sx == 0] <- 1; sy[sy == 0] <- 1
    list(x = sweep(sweep(x, 2, mx), 2, sx, "/"),
         y = sweep(sweep(y, 2, my), 2, sy, "/"),
         mx = mx, my = my, sx = sx, sy = sy)
}
package_fit <- function(z, method, penalty, k, initial_v = NULL) {
    if (method == "r_ridge") {
        lambda <- penalty * c(mean(diag(cov(z$x))), mean(diag(cov(z$y))))
        fit <- CCA::rcc(z$x, z$y, lambda1 = lambda[1], lambda2 = lambda[2])
        ax <- fit$xcoef[, seq_len(k), drop = FALSE]
        ay <- fit$ycoef[, seq_len(k), drop = FALSE]
        roots <- fit$cor[seq_len(k)]
    } else if (method == "r_classical") {
        fit <- CCA::cc(z$x, z$y)
        ax <- fit$xcoef[, seq_len(k), drop = FALSE]
        ay <- fit$ycoef[, seq_len(k), drop = FALSE]
        roots <- fit$cor[seq_len(k)]
    } else {
        fit <- PMA::CCA(z$x, z$y, typex = "standard", typez = "standard",
                        penaltyx = penalty[1], penaltyz = penalty[2],
                        K = k, niter = 100, standardize = FALSE, trace = FALSE,
                        v = initial_v)
        ax <- fit$u; ay <- fit$v; roots <- NULL
    }
    list(x_mean = z$mx, y_mean = z$my,
         x_coef = sweep(ax, 1, z$sx, "/"),
         y_coef = sweep(ay, 1, z$sy, "/"), roots = roots,
         train_correlations = paired_cor(z$x %*% ax, z$y %*% ay))
}
# Reuse each training fold's SVD across the grid. This exact ridge grid scorer
# is checked against the actual CCA::rcc package on a validation fold per job.
# All reported final R fits use CCA::rcc itself, including their runtime.
ridge_cache <- function(z) {
    sx <- svd(z$x); sy <- svd(z$y)
    rx <- which(sx$d > max(1e-8, sx$d[1] * max(dim(z$x)) * .Machine$double.eps))
    ry <- which(sy$d > max(1e-8, sy$d[1] * max(dim(z$y)) * .Machine$double.eps))
    list(dx = sx$d[rx], dy = sy$d[ry], vx = sx$v[, rx, drop = FALSE],
         vy = sy$v[, ry, drop = FALSE],
         cross = crossprod(sx$u[, rx, drop = FALSE], sy$u[, ry, drop = FALSE]),
         n = nrow(z$x), p = ncol(z$x), q = ncol(z$y))
}
cached_ridge <- function(cache, alpha) {
    hx <- sqrt(cache$dx^2 + alpha[1] * sum(cache$dx^2) / cache$p)
    hy <- sqrt(cache$dy^2 + alpha[2] * sum(cache$dy^2) / cache$q)
    kernel <- sweep(sweep(cache$cross, 1, cache$dx / hx, "*"),
                    2, cache$dy / hy, "*")
    fit <- svd(kernel, nu = 1, nv = 1)
    list(ax = sweep(cache$vx, 2, hx, "/") %*% fit$u,
         ay = sweep(cache$vy, 2, hy, "/") %*% fit$v)
}
best_index <- function(scores, grid, ridge) {
    means <- rowMeans(scores)
    eligible <- which(is.finite(means))
    if (!length(eligible)) stop("All candidates failed")
    best <- eligible[1]
    for (i in eligible[-1]) {
        tied <- abs(means[i] - means[best]) <= 1e-12 + 1e-12 * abs(means[best])
        stronger <- if (ridge) sum(grid[i, ]) > sum(grid[best, ]) else
                               sum(grid[i, ]) < sum(grid[best, ])
        if ((means[i] > means[best] && !tied) || (tied && stronger)) best <- i
    }
    best
}
tune <- function(x, y, fold_ids, grid, method, k, seed) {
    start <- proc.time()[[3]]
    scores <- matrix(NA_real_, nrow(grid), max(fold_ids))
    failures <- list()
    check_error <- NULL
    for (j in seq_len(max(fold_ids))) {
        train <- which(fold_ids != j); valid <- which(fold_ids == j)
        z <- preprocess(x[train, , drop = FALSE], y[train, , drop = FALSE])
        xv <- sweep(sweep(x[valid, , drop = FALSE], 2, z$mx), 2, z$sx, "/")
        yv <- sweep(sweep(y[valid, , drop = FALSE], 2, z$my), 2, z$sy, "/")
        if (method == "r_ridge") cache <- ridge_cache(z) else
            initial_v <- svd(crossprod(z$x, z$y), nu = 0, nv = 1)$v
        for (i in seq_len(nrow(grid))) {
            set.seed(seed + j * 1000 + i)
            value <- tryCatch({
                if (method == "r_ridge") {
                    fit <- cached_ridge(cache, grid[i, ])
                    paired_cor(xv %*% fit$ax, yv %*% fit$ay)[1]
                } else {
                    fit <- PMA::CCA(z$x, z$y, typex = "standard", typez = "standard",
                                    penaltyx = grid[i, 1], penaltyz = grid[i, 2],
                                    K = 1, niter = 100, standardize = FALSE,
                                    trace = FALSE, v = initial_v)
                    paired_cor(xv %*% fit$u, yv %*% fit$v)[1]
                }
            }, error = function(e) {
                failures[[length(failures) + 1]] <<- list(candidate = i - 1, fold = j - 1,
                                                          reason = conditionMessage(e))
                NA_real_
            })
            scores[i, j] <- value
        }
        if (j == 1 && method == "r_ridge") {
            # Independent package comparison, on validation rather than train.
            ref <- package_fit(z, method, grid[1, ], 1)
            pr <- paired_cor(sweep(x[valid, , drop = FALSE], 2, ref$x_mean) %*% ref$x_coef,
                             sweep(y[valid, , drop = FALSE], 2, ref$y_mean) %*% ref$y_coef)[1]
            check_error <- abs(pr - scores[1, j])
            stopifnot(check_error < 1e-8)
        }
    }
    best <- best_index(scores, grid, method == "r_ridge")
    penalty <- grid[best, ]
    z <- preprocess(x, y)
    set.seed(seed)
    fit_start <- proc.time()[[3]]
    fit <- package_fit(z, method, penalty, k)
    fit_seconds <- proc.time()[[3]] - fit_start
    c(fit, list(penalty = penalty, cv_scores = scores,
                mean_cv = mean(scores[best, ]), failures = failures,
                fit_seconds = fit_seconds,
                selection_and_refit_seconds = proc.time()[[3]] - start,
                cached_scorer_package_validation_error = check_error))
}
run <- function(req) {
    x <- as.matrix(read.csv(req$x_path, header = FALSE))
    y <- as.matrix(read.csv(req$y_path, header = FALSE))
    set.seed(req$seed)
    safe <- function(fun) tryCatch(fun(), error = function(e) list(error = conditionMessage(e)))
    if (isTRUE(req$reference_only)) {
        z <- preprocess(x, y)
        return(list(correctness_reference = safe(function() package_fit(
            z, "r_ridge", req$our_penalty, min(nrow(x) - 1, ncol(x), ncol(y))))))
    }
    ridge <- safe(function() tune(x, y, req$fold_ids, req$ridge_grid,
                                 "r_ridge", req$k, req$seed))
    pma <- safe(function() tune(x, y, req$fold_ids, req$pma_grid,
                               "r_pma", req$k, req$seed))
    z <- preprocess(x, y)
    reference <- safe(function() package_fit(z, "r_ridge", req$our_penalty,
                                            min(nrow(x) - 1, ncol(x), ncol(y))))
    classical <- if (min(nrow(x) - 1, nrow(y) - 1) < max(ncol(x), ncol(y)))
        list(error = "Classical CCA requires full column ranks") else safe(function() {
            t0 <- proc.time()[[3]]
            fit <- package_fit(z, "r_classical", NULL, req$k)
            c(fit, list(fit_seconds = proc.time()[[3]] - t0))
        })
    list(r_ridge = ridge, r_pma = pma, r_classical = classical,
         correctness_reference = reference)
}
cat(jsonlite::toJSON(list(ready = TRUE, R = R.version.string,
                         CCA = as.character(packageVersion("CCA")),
                         PMA = as.character(packageVersion("PMA"))), auto_unbox = TRUE), "\n")
flush(stdout())
con <- file("stdin", open = "r")
repeat {
    line <- readLines(con, n = 1, warn = FALSE)
    if (!length(line)) break
    response <- tryCatch(run(jsonlite::fromJSON(line)),
                         error = function(e) list(error = conditionMessage(e)))
    cat(jsonlite::toJSON(response, auto_unbox = TRUE, digits = 17, null = "null",
                        na = "null", matrix = "rowmajor"), "\n")
    flush(stdout())
}
