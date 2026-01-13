from imports import *

def load_jetnet(
    mode, # "5class" / "tqg" / "qg"
    n_train_pool=None,
    n_test_pool=None,
    n_train=None,
    n_val=None,
    n_test=None,
    qg_balance_anoms=True,
    seed=42
):
    classes = ["q", "g", "w", "z", "t"]
    name_to_i = {c: i for i, c in enumerate(classes)}

    def _one_hot(i, n=5):
        v = np.zeros((n,), dtype=np.float32)
        v[i] = 1.0
        return v

    def _load_class(c):
        path = os.path.join("datasets/jetnet", f"{c}.hdf5")
        with h5py.File(path, "r") as f:
            x = f["particle_features"][:].astype(np.float32)
        y = np.tile(_one_hot(name_to_i[c])[None, :], (x.shape[0], 1)).astype(np.float32)
        return x, y

    def _counts5(y):
        if y is None or y.shape[0] == 0:
            return {c: 0 for c in classes}
        yi = np.argmax(y, axis=1)
        return {c: int(np.sum(yi == name_to_i[c])) for c in classes}

    def _fmt_shape(a):
        return "None" if a is None else str(tuple(a.shape))

    def _make_row(name, x, y, y_top=None):
        c = _counts5(y)
        N = 0 if x is None else x.shape[0]
        row = {
            "name": name,
            "x_shape": _fmt_shape(x),
            "y_shape": _fmt_shape(y),
            "N": N,
            "q": c["q"], "g": c["g"], "w": c["w"], "z": c["z"], "t": c["t"],
            "top": None, "qg": None,
            "has_top": (y_top is not None),
        }
        if y_top is not None:
            row["top"] = int(np.sum(y_top == 1))
            row["qg"] = int(np.sum(y_top == 0))
        return row

    def _print_table(title, rows):
        w_name = max(len(r["name"]) for r in rows)
        w_x = max(len(r["x_shape"]) for r in rows)
        w_y = max(len(r["y_shape"]) for r in rows)

        print(title)
        for r in rows:
            s = (
                f"  {r['name']:<{w_name}} | "
                f"x {r['x_shape']:<{w_x}} | "
                f"y {r['y_shape']:<{w_y}} "
                f"|| "
                f"q {r['q']:7d} | g {r['g']:7d} | W {r['w']:7d} | Z {r['z']:7d} | t {r['t']:7d}"
            )
            #if r["has_top"]:
            #    s += f" || top {r['top']:7d} | (q+g) {r['qg']:7d}"
            print(s)
        print()

    def _split_counts(mode_local, n_total):
        if mode_local == "5class":
            base = n_total // 5
            counts = {c: base for c in classes}
            rem = n_total - 5 * base
            for c in classes[:rem]:
                counts[c] += 1
            return counts
        if mode_local == "tqg":
            nq = int(round(0.25 * n_total))
            ng = int(round(0.25 * n_total))
            nt = n_total - nq - ng
            return {"q": nq, "g": ng, "w": 0, "z": 0, "t": nt}
        raise ValueError("counts only used for pooled modes")

    def _sample_disjoint_from_pool(x_pool, y_pool, counts, rng, avail_idx=None):
        if avail_idx is None:
            avail_idx = np.arange(x_pool.shape[0])

        yi_pool = np.argmax(y_pool, axis=1)

        selected = []
        for c in classes:
            need = int(counts.get(c, 0))
            if need == 0:
                continue
            cls_i = name_to_i[c]
            candidates = avail_idx[yi_pool[avail_idx] == cls_i]
            if need > candidates.shape[0]:
                raise ValueError(f"Not enough '{c}' in pool for split: need {need}, have {candidates.shape[0]}")
            pick = rng.choice(candidates, size=need, replace=False)
            selected.append(pick)

        if len(selected) == 0:
            sel_idx = np.array([], dtype=np.int64)
        else:
            sel_idx = np.concatenate(selected, axis=0)
            sel_idx = sel_idx[rng.permutation(sel_idx.shape[0])]

        if sel_idx.shape[0] == 0:
            new_avail = avail_idx
        else:
            sel_set = set(map(int, sel_idx))
            mask_keep = np.array([int(a) not in sel_set for a in avail_idx], dtype=bool)
            new_avail = avail_idx[mask_keep]

        return x_pool[sel_idx], y_pool[sel_idx], new_avail

    def _top_binary(y5):
        if y5 is None:
            return None
        if y5.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32)
        yi = np.argmax(y5, axis=1)
        return (yi == name_to_i["t"]).astype(np.int32)

    # load data files
    X, Y, raw_counts = {}, {}, {}
    for c in classes:
        xc, yc = _load_class(c)
        X[c], Y[c] = xc, yc
        raw_counts[c] = xc.shape[0]

    print("Data files:")
    print(f"  q:{raw_counts['q']} | g:{raw_counts['g']} | w:{raw_counts['w']} | z:{raw_counts['z']} | t:{raw_counts['t']}")
    print(f"  total:{sum(raw_counts.values())}\n")

    if mode == "qg":
        rng = np.random.default_rng(seed)
        perm = {c: rng.permutation(X[c].shape[0]) for c in classes}

        n_q_tr = n_train // 2
        n_g_tr = n_train // 2
        if n_q_tr > X["q"].shape[0] or n_g_tr > X["g"].shape[0]:
            raise ValueError(f"Not enough q/g: need q={n_q_tr}, g={n_g_tr}.")

        q_tr = perm["q"][:n_q_tr]; q_te = perm["q"][n_q_tr:]
        g_tr = perm["g"][:n_g_tr]; g_te = perm["g"][n_g_tr:]

        n_qg_te = min(len(q_te), len(g_te))
        q_te = q_te[:n_qg_te]
        g_te = g_te[:n_qg_te]

        x_train = np.concatenate([X["q"][q_tr], X["g"][g_tr]], axis=0)
        y_train = np.concatenate([Y["q"][q_tr], Y["g"][g_tr]], axis=0)

        x_test_parts = [X["q"][q_te], X["g"][g_te]]
        y_test_parts = [Y["q"][q_te], Y["g"][g_te]]

        w_idx = perm["w"]; z_idx = perm["z"]; t_idx = perm["t"]
        if qg_balance_anoms:
            n_target = min(len(q_te), X["w"].shape[0], X["z"].shape[0], X["t"].shape[0])
            w_sel = w_idx[:n_target]; z_sel = z_idx[:n_target]; t_sel = t_idx[:n_target]
        else:
            w_sel, z_sel, t_sel = w_idx, z_idx, t_idx
        x_test_parts += [X["w"][w_sel], X["z"][z_sel], X["t"][t_sel]]
        y_test_parts += [Y["w"][w_sel], Y["z"][z_sel], Y["t"][t_sel]]

        x_test = np.concatenate(x_test_parts, axis=0)
        y_test = np.concatenate(y_test_parts, axis=0)

        p = rng.permutation(x_train.shape[0])
        x_train, y_train = x_train[p], y_train[p]
        p = rng.permutation(x_test.shape[0])
        x_test, y_test = x_test[p], y_test[p]

        rows = [
            _make_row("train", x_train, y_train),
            _make_row("test",  x_test,  y_test),
        ]
        _print_table("Splits:", rows)

        return x_train, y_train, None, None, x_test, y_test, None, None, None

    # 5class and tqg
    x_all = np.concatenate([X[c] for c in classes], axis=0)
    y_all = np.concatenate([Y[c] for c in classes], axis=0)

    total_needed = n_train_pool + n_test_pool
    if total_needed > x_all.shape[0]:
        raise ValueError(f"n_train_pool+n_test_pool={total_needed} exceeds total available {x_all.shape[0]}.")

    rng = np.random.default_rng(seed)
    perm_all = rng.permutation(x_all.shape[0])
    x_all = x_all[perm_all]
    y_all = y_all[perm_all]

    x_train_pool = x_all[:n_train_pool]
    y_train_pool = y_all[:n_train_pool]
    x_test_pool = x_all[n_train_pool:n_train_pool + n_test_pool]
    y_test_pool = y_all[n_train_pool:n_train_pool + n_test_pool]

    _print_table(
        "Global pools:",
        [
            _make_row("train_pool", x_train_pool, y_train_pool),
            _make_row("test_pool",  x_test_pool,  y_test_pool),
        ],
    )

    rng_split = np.random.default_rng(seed)

    x_train, y_train, avail = _sample_disjoint_from_pool(
        x_train_pool, y_train_pool, _split_counts(mode, n_train), rng_split, avail_idx=None
    )

    if n_val > 0:
        x_val, y_val, avail = _sample_disjoint_from_pool(
            x_train_pool, y_train_pool, _split_counts(mode, n_val), rng_split, avail_idx=avail
        )
    else:
        x_val = np.zeros((0, 30, 4), dtype=np.float32)
        y_val = np.zeros((0, 5), dtype=np.float32)

    x_test, y_test, _ = _sample_disjoint_from_pool(
        x_test_pool, y_test_pool, _split_counts(mode, n_test), rng_split, avail_idx=None
    )

    y_train_top = y_val_top = y_test_top = None
    if mode == "tqg":
        y_train_top = _top_binary(y_train)
        y_val_top = _top_binary(y_val)
        y_test_top = _top_binary(y_test)

    _print_table(
        "Splits:",
        [
            _make_row("train", x_train, y_train, y_train_top),
            _make_row("val", x_val, y_val, y_val_top),
            _make_row("test", x_test, y_test, y_test_top),
        ],
    )

    return x_train, y_train, x_val, y_val, x_test, y_test, y_train_top, y_val_top, y_test_top