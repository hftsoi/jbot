from imports import *

def get_cls_embeddings(backbone, x):
    ds = tf.data.Dataset.from_tensor_slices(x).batch(4096)
    out = []
    for i, xb in enumerate(ds):
        cls, _ = backbone(xb, training=False)
        out.append(cls.numpy())
    return np.concatenate(out, axis=0)

def l2_normalize(z, axis=1):
    n = np.linalg.norm(z, axis=axis, keepdims=True)
    return z / np.maximum(n, 1e-12)

def make_bank(z_train, M, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(z_train.shape[0], size=M, replace=False)
    return z_train[idx]

def knn_distances(z_bank, z_test, l2norm, kmax):
    if l2norm:
        z_bank = l2_normalize(z_bank)
        z_test = l2_normalize(z_test)

    knn = NearestNeighbors(n_neighbors=kmax, metric="euclidean", n_jobs=-1)
    knn.fit(z_bank)
    
    # out = (N_test, kmax), with kmax sorted by increasing distance
    dists, _ = knn.kneighbors(z_test, return_distance=True)
    return dists

def knn_aggregate(dists, k, agg):
    d = dists[:, :k]
    if agg == "mean":
        return d.mean(axis=1)
    if agg == "median":
        return np.median(d, axis=1)
    if agg == "max":
        return d.max(axis=1)
    raise ValueError("agg = 'mean' / 'median' / 'max'")

def mahalanobis(z_bank, z_test, l2norm, reg):
    if l2norm:
        z_bank = l2_normalize(z_bank)
        z_test = l2_normalize(z_test)

    mu = z_bank.mean(axis=0, keepdims=True)
    Xc = z_bank - mu
    cov = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)

    L = np.linalg.cholesky(cov)
    Yc = (z_test - mu).T
    v = np.linalg.solve(L, Yc) 
    score = np.sum(v*v, axis=0)
    return score

def cosine_similarities(z_bank, z_test, kmax):
    z_bank = l2_normalize(z_bank)
    z_test = l2_normalize(z_test)
    nn = NearestNeighbors(n_neighbors=kmax, metric="cosine", n_jobs=-1)
    nn.fit(z_bank)
    cos_dist, _ = nn.kneighbors(z_test, return_distance=True)
    sims = 1 - cos_dist
    return sims

def cosine_aggregate(sims, k, agg, temp):
    s = sims[:, :k]
    if agg == "max":
        return 1 - s.max(axis=1)
    if agg == "softmax_mean":
        m = s.max(axis=1, keepdims=True)
        w = np.exp((s - m) / temp)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        return 1 - (w * s).sum(axis=1)
    if agg == "logsumexp":
        sim_lse = temp*np.log(np.mean(np.exp(s / temp), axis=1) + 1e-12)
        return - sim_lse
    raise ValueError("agg = 'max' / 'softmax_mean' / 'logsumexp'")

def make_bank_with_labels(z_train, y_train5, M, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(z_train.shape[0], size=M, replace=False)
    return z_train[idx], y_train5[idx]

def _maha_md2_from_chol(z, mu, L):
    # z: (N,d), mu: (1,d), L: cholesky(cov) (d,d)
    Yc = (z - mu).T
    v = np.linalg.solve(L, Yc)
    return np.sum(v * v, axis=0) # (N,)

def cc_mahalanobis_min_qg_score(z_bank, y_bank5, z_test, l2norm, reg):
    if l2norm:
        z_bank = l2_normalize(z_bank)
        z_test = l2_normalize(z_test)

    yi = np.argmax(y_bank5, axis=1)
    m_q = (yi == 0)
    m_g = (yi == 1)

    # shared covariance from q+g bank
    mu0 = z_bank.mean(axis=0, keepdims=True)
    Xc = z_bank - mu0
    cov = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)
    L = np.linalg.cholesky(cov)

    mu_q = z_bank[m_q].mean(axis=0, keepdims=True)
    mu_g = z_bank[m_g].mean(axis=0, keepdims=True)

    mdq = _maha_md2_from_chol(z_test, mu_q, L)
    mdg = _maha_md2_from_chol(z_test, mu_g, L)

    return np.minimum(mdq, mdg)

def relative_mahalanobis_min_qg_score(z_bank, y_bank5, z_test, l2norm, reg, variant):
    if l2norm:
        z_bank = l2_normalize(z_bank)
        z_test = l2_normalize(z_test)

    yi = np.argmax(y_bank5, axis=1)
    m_q = (yi == 0)
    m_g = (yi == 1)

    mu0 = z_bank.mean(axis=0, keepdims=True)
    Xc = z_bank - mu0
    cov = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))
    cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)
    L = np.linalg.cholesky(cov)

    mu_q = z_bank[m_q].mean(axis=0, keepdims=True)
    mu_g = z_bank[m_g].mean(axis=0, keepdims=True)

    md0 = _maha_md2_from_chol(z_test, mu0, L)
    mdq = _maha_md2_from_chol(z_test, mu_q, L)
    mdg = _maha_md2_from_chol(z_test, mu_g, L)

    # min(MDq^2 - MD0^2, MDg^2 - MD0^2)
    if variant == "sub":
        return np.minimum(mdq - md0, mdg - md0)

    # min(MDq^2/(MD0^2), MDg^2/(MD0^2))
    if variant == "ratio":
        return np.minimum(mdq / md0, mdg / md0)

    # log(min(MDq^2,MDg^2)) - log(MD0^2)
    if variant == "logratio":
        return np.log(np.minimum(mdq, mdg)) - np.log(md0)

def gmm_score(z_bank, z_test, K, cov_type, l2norm, reg_covar, seed):
    if l2norm:
        z_bank = l2_normalize(z_bank)
        z_test = l2_normalize(z_test)

    gm = GaussianMixture(
        n_components=K,
        covariance_type=cov_type, # full,tied,diag
        reg_covar=reg_covar,
        random_state=seed,
        max_iter=300,
        n_init=1,
    )
    gm.fit(z_bank)

    # anomaly score = -log p(x)
    return -gm.score_samples(z_test)
    
def run_ad_scan(
    z_train, z_test, y5,
    seed,
    fpr_targets,
    scan_maha,
    scan_knn,
    scan_cos,
    scan_ccmd,
    scan_rmd,
    scan_gmm,
    y_train5,
    highlight_top,
):
    yi = np.argmax(y5, axis=1)

    # metrics
    def _tpr_at_fpr(y_true, score, fpr_target):
        fpr, tpr, _ = roc_curve(y_true, score)
        ok = np.where(fpr <= fpr_target)[0]
        return 0 if ok.size == 0 else np.max(tpr[ok])

    def _eval_sig(score, sig_idx):
        bg = (yi == 0) | (yi == 1)
        sig = (yi == sig_idx)
        m = bg | sig
        if bg.sum() == 0 or sig.sum() == 0:
            return tuple([np.nan] * (1 + len(fpr_targets)))
        y_true = sig[m].astype(np.int32)
        s = np.asarray(score, dtype=np.float32)[m]
        aucv = roc_auc_score(y_true, s)
        tprs = tuple(_tpr_at_fpr(y_true, s, t) for t in fpr_targets)
        return (aucv,) + tprs

    def _eval_all(score):
        bkg = (yi == 0) | (yi == 1)
        sig = (yi == 2) | (yi == 3) | (yi == 4)
        m = bkg | sig
        if bkg.sum() == 0 or sig.sum() == 0:
            return tuple([np.nan] * (1 + len(fpr_targets)))
        y_true = sig[m].astype(np.int32)
        s = np.asarray(score, dtype=np.float32)[m]
        aucv = roc_auc_score(y_true, s)
        tprs = tuple(_tpr_at_fpr(y_true, s, t) for t in fpr_targets)
        return (aucv,) + tprs

    # formatting / highlighting
    def _strip_ansi(s):
        import re
        return re.sub(r"\x1b\[[0-9;]*m", "", s)

    def _pad_vis(s, width):
        vis = len(_strip_ansi(s))
        if vis >= width:
            return s
        return s + " " * (width - vis)

    # all highlighted entries are bold red
    H_PRE, H_POST = ("\033[1;91m", "\033[0m")

    def _fmt_num(v, rank=None):
        s = f"{v:.4f}"
        if rank is None:
            return s
        return f"{H_PRE}{s}{H_POST}"

    def _default_tag(prefix, cfg):
        keys = [k for k in cfg.keys()]
        parts = [f"{k}={cfg[k]}" for k in keys]
        return f"{prefix}(" + ", ".join(parts) + ")"

    rows = [] # dict(group, tag, stats)

    # naive maha
    if scan_maha:
        prefix = scan_maha.get("name", "maha")
        Ms = scan_maha.get("M", [])
        l2s = scan_maha.get("l2", [])
        regs = scan_maha.get("reg", [])
        tag_fn = scan_maha.get("tag", None)

        for M in Ms:
            bank = make_bank(z_train, M=M, seed=seed)
            for l2 in l2s:
                for reg in regs:
                    score = mahalanobis(bank, z_test, l2norm=l2, reg=reg)
                    stats = {"W": _eval_sig(score, 2), "Z": _eval_sig(score, 3), "t": _eval_sig(score, 4), "all": _eval_all(score)}
                    cfg = {"M": M, "l2": l2, "reg": reg}
                    tag = tag_fn(cfg) if callable(tag_fn) else _default_tag(prefix, cfg)
                    rows.append({"group": "maha", "tag": tag, "stats": stats})

    # kNN
    if scan_knn:
        prefix = scan_knn.get("name", "knn")
        Ms = scan_knn.get("M", [])
        l2s = scan_knn.get("l2", [])
        kmaxs = scan_knn.get("kmax", [])
        ks = scan_knn.get("k", [])
        aggs = scan_knn.get("agg", [])
        tag_fn = scan_knn.get("tag", None)

        for M in Ms:
            bank = make_bank(z_train, M=M, seed=seed)
            for l2 in l2s:
                for kmax in kmaxs:
                    dists = knn_distances(bank, z_test, l2norm=l2, kmax=kmax)
                    for k in ks:
                        for agg in aggs:
                            score = knn_aggregate(dists, k=k, agg=agg)
                            stats = {"W": _eval_sig(score, 2), "Z": _eval_sig(score, 3), "t": _eval_sig(score, 4), "all": _eval_all(score)}
                            cfg = {"M": M, "l2": l2, "kmax": kmax, "k": k, "agg": agg}
                            tag = tag_fn(cfg) if callable(tag_fn) else _default_tag(prefix, cfg)
                            rows.append({"group": "knn", "tag": tag, "stats": stats})

    # cosine sims
    if scan_cos:
        prefix = scan_cos.get("name", "cos")
        Ms = scan_cos.get("M", [])
        kmaxs = scan_cos.get("kmax", [])
        ks = scan_cos.get("k", [])
        aggs = scan_cos.get("agg", [])
        temps = scan_cos.get("temp", [])
        tag_fn = scan_cos.get("tag", None)

        for M in Ms:
            bank = make_bank(z_train, M=M, seed=seed)
            for kmax in kmaxs:
                sims = cosine_similarities(bank, z_test, kmax=kmax)
                for k in ks:
                    for agg in aggs:
                        use_temps = [None] if agg == "max" else temps
                        for T in use_temps:
                            score = cosine_aggregate(sims, k=k, agg=agg, temp=T)
                            stats = {"W": _eval_sig(score, 2), "Z": _eval_sig(score, 3), "t": _eval_sig(score, 4), "all": _eval_all(score)}
                            cfg = {"M": M, "kmax": kmax, "k": k, "agg": agg}
                            if T is not None:
                                cfg["T"] = T
                            tag = tag_fn(cfg) if callable(tag_fn) else _default_tag(prefix, cfg)
                            rows.append({"group": "cos", "tag": tag, "stats": stats})

    # class-conditional MD (min q/g)
    if scan_ccmd:
        prefix = scan_ccmd.get("name", "ccMD")
        Ms = scan_ccmd.get("M", [])
        l2s = scan_ccmd.get("l2", [])
        regs = scan_ccmd.get("reg", [])
        tag_fn = scan_ccmd.get("tag", None)

        for M in Ms:
            z_bank, y_bank = make_bank_with_labels(z_train, y_train5, M=M, seed=seed)
            for l2 in l2s:
                for reg in regs:
                    score = cc_mahalanobis_min_qg_score(z_bank, y_bank, z_test, l2norm=l2, reg=reg)
                    stats = {"W": _eval_sig(score, 2), "Z": _eval_sig(score, 3), "t": _eval_sig(score, 4), "all": _eval_all(score)}
                    cfg = {"M": M, "l2": l2, "reg": reg}
                    tag = tag_fn(cfg) if callable(tag_fn) else _default_tag(prefix, cfg)
                    rows.append({"group": "ccmd", "tag": tag, "stats": stats})

    # relative MD
    if scan_rmd:
        prefix = scan_rmd.get("name", "RMD")
        Ms = scan_rmd.get("M", [])
        l2s = scan_rmd.get("l2", [])
        regs = scan_rmd.get("reg", [])
        variants = scan_rmd.get("variant", [])
        tag_fn = scan_rmd.get("tag", None)

        for M in Ms:
            z_bank, y_bank = make_bank_with_labels(z_train, y_train5, M=M, seed=seed)
            for l2 in l2s:
                for reg in regs:
                    for var in variants:
                        score = relative_mahalanobis_min_qg_score(
                            z_bank, y_bank, z_test,
                            l2norm=l2, reg=reg, variant=var
                        )
                        stats = {"W": _eval_sig(score, 2), "Z": _eval_sig(score, 3), "t": _eval_sig(score, 4), "all": _eval_all(score)}
                        cfg = {"M": M, "l2": l2, "reg": reg, "variant": var}
                        tag = tag_fn(cfg) if callable(tag_fn) else _default_tag(prefix, cfg)
                        rows.append({"group": "rmd", "tag": tag, "stats": stats})

    # GMM
    if scan_gmm:
        prefix = scan_gmm.get("name", "GMM")
        Ms = scan_gmm.get("M", [])
        l2s = scan_gmm.get("l2", [])
        Ks = scan_gmm.get("K", [])
        cov_types = scan_gmm.get("cov_type", [])
        reg_covars = scan_gmm.get("reg_covar", [])
        gmm_seed = scan_gmm.get("seed", seed)
        tag_fn = scan_gmm.get("tag", None)

        for M in Ms:
            bank = make_bank(z_train, M=M, seed=seed)
            for l2 in l2s:
                for K in Ks:
                    for cov_type in cov_types:
                        for reg_covar in reg_covars:
                            score = gmm_score(
                                bank, z_test, K=K,
                                cov_type=cov_type,
                                l2norm=l2,
                                reg_covar=reg_covar,
                                seed=gmm_seed
                            )
                            stats = {"W": _eval_sig(score, 2), "Z": _eval_sig(score, 3), "t": _eval_sig(score, 4), "all": _eval_all(score)}
                            cfg = {"M": M, "l2": l2, "K": K, "cov": cov_type, "reg": reg_covar}
                            tag = tag_fn(cfg) if callable(tag_fn) else _default_tag(prefix, cfg)
                            rows.append({"group": "gmm", "tag": tag, "stats": stats})

    # compute top-N
    signals = ["W", "Z", "t", "all"]
    n_metrics = 1 + len(fpr_targets)
    rank_map = {}

    for sig in signals:
        for j in range(n_metrics):
            vals = np.array([r["stats"][sig][j] for r in rows], dtype=np.float32)
            finite = np.isfinite(vals)
            if finite.sum() == 0:
                continue
            idxs = np.where(finite)[0]
            order = idxs[np.argsort(vals[idxs])[::-1]]

            topk = max(0, min(highlight_top, order.size))
            top = order[:topk]
            for rank, ridx in enumerate(top):
                rank_map[(ridx, sig, j)] = rank

    # printing
    fpr_lbl = ",".join([f"{t:g}" for t in fpr_targets])
    col_names = [f"W(AUC,{fpr_lbl})", f"Z(AUC,{fpr_lbl})", f"t(AUC,{fpr_lbl})", f"all(AUC,{fpr_lbl})"]

    tag_w = max(len(r["tag"]) for r in rows)
    cell_w = 6 + (n_metrics - 1) * 8
    header = (
        f"{'tag':<{tag_w}} | "
        f"{col_names[0]:<{cell_w}} | {col_names[1]:<{cell_w}} | {col_names[2]:<{cell_w}} | {col_names[3]:<{cell_w}}"
    )
    print(header)
    print("-" * len(_strip_ansi(header)))

    last_group = None
    for i, r in enumerate(rows):
        if last_group is not None and r["group"] != last_group:
            print("")
        last_group = r["group"]

        def _cell(sig):
            vals = r["stats"][sig]
            parts = []
            for j, v in enumerate(vals):
                rank = rank_map.get((i, sig, j), None)
                parts.append(_fmt_num(v, rank))
            return "  ".join(parts)

        cW = _pad_vis(_cell("W"), cell_w)
        cZ = _pad_vis(_cell("Z"), cell_w)
        ct = _pad_vis(_cell("t"), cell_w)
        ca = _pad_vis(_cell("all"), cell_w)

        line = f"{r['tag']:<{tag_w}} | {cW} | {cZ} | {ct} | {ca}"
        print(line)

def report_ad_kfold(scores_dict, y5, kfold, eff_b_targets, seed):
    yi = y5.argmax(1)
    bkg = (yi == 0) | (yi == 1)
    sig_masks = {
        "W": (yi == 2),
        "Z": (yi == 3),
        "t": (yi == 4),
        "all": (yi == 2) | (yi == 3) | (yi == 4),
    }

    def _metrics(y_bin, s):
        fpr, tpr, _ = roc_curve(y_bin, s)
        aucv = auc(fpr, tpr)
        effs = []
        for eb in eff_b_targets:
            ok = np.where(fpr <= eb)[0]
            effs.append(tpr[ok].max() if ok.size else 0)
        return aucv, effs

    for sig_name, sig in sig_masks.items():
        m = bkg | sig
        y_bin = sig[m]

        print(f"\n{sig_name} vs QCD  (k={kfold})")
        print(f"{'method':10s}  {'AUC':>14s}  " + "  ".join([f"eff@{eb:g}".rjust(14) for eb in eff_b_targets]))
        print("-" * (10 + 2 + 14 + 2 + (14 + 2) * len(eff_b_targets)))

        for name, score in scores_dict.items():
            s_all = np.asarray(score)[m]

            skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            aucs, effs = [], []
            for _, te_idx in skf.split(np.zeros_like(y_bin), y_bin):
                aucv, e = _metrics(y_bin[te_idx], s_all[te_idx])
                aucs.append(aucv)
                effs.append(e)

            aucs = np.asarray(aucs)
            effs = np.asarray(effs)

            mu_auc, sd_auc = aucs.mean(), aucs.std()
            mu_eff, sd_eff = effs.mean(axis=0), effs.std(axis=0)

            cells = [f"{mu_auc:.4f}+/-{sd_auc:.4f}".rjust(14)]
            cells += [f"{mu_eff[i]:.4f}+/-{sd_eff[i]:.4f}".rjust(14) for i in range(len(eff_b_targets))]
            print(f"{name:10s}  " + "  ".join(cells))