from imports import *

#-------------------------------------- plotting --------------------------------------

def plot_jets(x, y, idx):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    class_names = ['q', 'g', 'W', 'Z', 't']
    
    for i, class_name in enumerate(class_names):
        class_idx = np.where(y.argmax(axis=1)==i)[0][idx]
        jet = x[class_idx]
        eta = jet[:,0]
        phi = jet[:,1]
        pt = jet[:,2]
        mask = jet[:,3]
        
        eta = eta[mask==1]
        phi = phi[mask==1]
        pt = pt[mask==1]
    
        size = pt*10000
    
        axs[i].scatter(eta, phi, s=size, facecolors="none", edgecolors="C0", linewidths=0.8, alpha=0.5)
        axs[i].set_xlabel("Eta")
        axs[i].set_ylabel("Phi")
        axs[i].set_title(class_name)
        axs[i].set_xlim(-0.4, 0.4)
        axs[i].set_ylim(-0.4, 0.4)
    
    plt.tight_layout()
    plt.show()

def plot_jets_aug(x, y, mask_ratio_range, idx, save_path=None):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    class_names = ['q', 'g', 'W', 'Z', 't']
    
    lw = 0.8
    alpha = 0.5
    for col, cname in enumerate(class_names):
        jet_orig = x[np.where(y.argmax(axis=1)==col)[0][idx]]
    
        # row 0: original
        eta, phi, pt, valid = jet_orig.T
        m0 = (valid == 1)
        axes[0, col].scatter(eta[m0], phi[m0], s=pt[m0]*1e4, facecolors="none", edgecolors="C0", linewidths=lw, alpha=alpha)
        axes[0, col].set_title(f"{cname}", fontsize=14)
    
        # row 1: augmented
        jet_aug = augment_jet(jet_orig)
        eta, phi, pt, valid = jet_aug.T
        m1 = (valid == 1)
        #axes[1, col].scatter(eta[m1], phi[m1], s=pt[m1]*1e4, facecolors="none", edgecolors="C0", linewidths=lw, alpha=alpha)
        #axes[1, col].set_title(f"{cname} [augmented]")
    
        # row 2: augmented + masks
        r_target = float(RNG.uniform(mask_ratio_range[0], mask_ratio_range[1]))
        jet_tf = tf.convert_to_tensor(jet_aug[None, ...], dtype=tf.float32)
        mask_bool = tf_make_masks_pt_ratio(jet_tf, p_mask=1, ratio_range=(r_target, r_target)).numpy()[0].astype(bool)
    
        total_pt = float(pt[m1].sum())
        masked_pt = float(pt[mask_bool].sum())
        r_ach = masked_pt / total_pt
    
        unmasked = (valid == 1) & (~mask_bool)
        masked = (valid == 1) & (mask_bool)

        axes[1, col].scatter(eta[unmasked], phi[unmasked], s=pt[unmasked]*1e4, facecolors="none", edgecolors="C0", linewidths=lw, alpha=alpha)
        axes[1, col].scatter(eta[masked], phi[masked], s=pt[masked]*1e4, facecolors="none", edgecolors="red", linewidths=lw, alpha=alpha)
    
        axes[1, col].set_title(f"Aug. {cname} (masked in red, ~30% jet $p_\\mathrm{{T}}$)", fontsize=14)
    
        print(f"{cname}: r_target={r_target:.3f}, r_achieved={r_ach:.3f}, n_mask={masked.sum()} / n_valid={m1.sum()}")
    
        for row in range(2):
            axes[row, col].set_xlim(-0.4, 0.4)
            axes[row, col].set_ylim(-0.4, 0.4)
            axes[row, col].set_xlabel("Eta", fontsize=14)
            axes[row, col].set_ylabel("Phi", fontsize=14)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

def history_json(history, path, mode):
    def _to_jsonable(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.astype(float).tolist()
        return x

    if mode == "save":
        hist = {}
        for k, v in history.items():
            if isinstance(v, (list, tuple)):
                hist[k] = [_to_jsonable(t) for t in v]
            else:
                hist[k] = _to_jsonable(v)
        with open(path, "w") as f:
            json.dump(hist, f, indent=2)

    if mode == "load":
        with open(path, "r") as f:
            hist = json.load(f)
        return hist
        
def plot_jbot_pretraining(history, save_path=None):
    fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=False)
    lw=4
    title_size=12
    label_size=10
    ax[0,0].plot(history["loss_total"], linewidth=lw)
    ax[0,0].set_title("Loss (total)", fontsize=title_size)
    ax[0,0].set_xlabel("Epoch", fontsize=label_size)
    ax[0,0].grid(alpha=0.5)

    ax[0,1].plot(history["loss_cls"], linewidth=lw)
    ax[0,1].set_title("Loss ([CLS])", fontsize=title_size)
    ax[0,1].set_xlabel("Epoch", fontsize=label_size)
    ax[0,1].grid(alpha=0.5)

    ax[0,2].plot(history["loss_patch"], linewidth=lw)
    ax[0,2].set_title("Loss (particle)", fontsize=title_size)
    ax[0,2].set_xlabel("Epoch", fontsize=label_size)
    ax[0,2].grid(alpha=0.5)

    ax[1,0].plot(history["loss_koleo"], linewidth=lw)
    ax[1,0].set_title("Loss (Koleo)", fontsize=title_size)
    ax[1,0].set_xlabel("Epoch", fontsize=label_size)
    ax[1,0].grid(alpha=0.5)

    ax[1,1].plot(history["entropy_teacher"], label="Teacher", linewidth=lw)
    ax[1,1].plot(history["entropy_student"], label="Student", linewidth=lw)
    ax[1,1].set_title("Entropy", fontsize=title_size)
    ax[1,1].set_xlabel("Epoch", fontsize=label_size)
    ax[1,1].legend(fontsize=10)
    ax[1,1].grid(alpha=0.5)

    ax[1,2].plot(history["center_cls"], label="[CLS]", linewidth=lw)
    ax[1,2].plot(history["center_patch"], label="Particle", linewidth=lw)
    ax[1,2].set_title("Centering norm", fontsize=title_size)
    ax[1,2].set_xlabel("Epoch", fontsize=label_size)
    ax[1,2].legend(fontsize=10)
    ax[1,2].grid(alpha=0.5)

    #ax[2,0].plot(history["mask_token"], linewidth=lw)
    #ax[2,0].set_title("Learnable mask token")
    #ax[2,0].set_xlabel("Epoch")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()
    
def plot_attention(transformer, n_heads, n_layers, x_sample, y_sample, lw=0.8, save_path=None):
    last_block = transformer.get_layer(f"block_{n_layers-1}")
    mha_last = last_block.mha

    # rebuild model with attention scores as output
    x_in = transformer.input
    x = transformer.get_layer("token_embedding")(x_in)
    x = transformer.get_layer("prepend_cls")(x)
    for i in range(n_layers-1):
        x = transformer.get_layer(f"block_{i}")(x)
    _, attn_scores = mha_last(x, x, return_attention_scores=True)
    model_re = tf.keras.Model(x_in, attn_scores)

    class_names_all = ['q', 'g', 'W', 'Z', 't']
    y_arg = y_sample.argmax(axis=1)

    # keep only classes that actually exist in y_sample (preserve original order)
    present_cols = [i for i, _ in enumerate(class_names_all) if np.any(y_arg == i)]
    present_names = [class_names_all[i] for i in present_cols]
    n_cols = len(present_cols)

    rows = 1 + n_heads

    fig = plt.figure(figsize=(3*n_cols, 3*rows))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, n_heads], hspace=0.04)

    gs_top = gs[0].subgridspec(1, n_cols, wspace=0, hspace=0)
    gs_bottom = gs[1].subgridspec(n_heads, n_cols, wspace=0, hspace=0)

    axes = np.empty((rows, n_cols), dtype=object)

    # first row (input)
    for c in range(n_cols):
        if c == 0:
            axes[0, 0] = fig.add_subplot(gs_top[0, 0])
            ref_top = axes[0, 0]
        else:
            axes[0, c] = fig.add_subplot(gs_top[0, c], sharex=ref_top, sharey=ref_top)

    # remaining rows (attention heads)
    for r in range(n_heads):
        for c in range(n_cols):
            if r == 0 and c == 0:
                axes[1, 0] = fig.add_subplot(gs_bottom[0, 0])
                ref_bottom = axes[1, 0]
            else:
                axes[1+r, c] = fig.add_subplot(gs_bottom[r, c], sharex=ref_bottom, sharey=ref_bottom)

    for col, (orig_class_idx, cname) in enumerate(zip(present_cols, present_names)):
        idxs = np.where(y_arg == orig_class_idx)[0]
        if idxs.size == 0:
            continue  # extra safety; should not happen due to present_cols filter
        idx = idxs[0]

        jet = x_sample[idx][None, ...].astype("float32") # (1,30,4)
        scores = model_re(jet).numpy()[0] # (n_heads, 31, 31)
        QUERYcls_KEYpatch = scores[:, 0, 1:] # (n_heads, 30)

        eta, phi, pt, valid = jet[0].T
        m = (valid == 1)
        eta, phi, pt = eta[m], phi[m], pt[m]
        QUERYcls_KEYpatch = QUERYcls_KEYpatch[:, m]

        # row 0: input (hollow C0)
        ax0 = axes[0, col]
        ax0.scatter(eta, phi, s=pt*1e4, facecolors="none", edgecolors="C0", linewidths=lw, alpha=0.5)
        ax0.set_title(f"{cname}", fontsize=14)
        ax0.set_xlim(-0.4, 0.4)
        ax0.set_ylim(-0.4, 0.4)
        if col == 0:
            ax0.set_ylabel("input", fontsize=14)

        # attention heads (attention encoded as alpha)
        for h in range(n_heads):
            w = QUERYcls_KEYpatch[h]
            w = (w - w.min())/(w.ptp() + 1e-8)

            base_rgba = mcolors.to_rgba(f"C{h+1}")
            edge_rgba = np.zeros((len(w), 4), dtype=np.float32)
            edge_rgba[:, 0] = base_rgba[0]
            edge_rgba[:, 1] = base_rgba[1]
            edge_rgba[:, 2] = base_rgba[2]
            edge_rgba[:, 3] = w.astype(np.float32)

            ax = axes[1+h, col]
            ax.scatter(eta, phi, s=pt*1e4, facecolors="none", edgecolors=edge_rgba, linewidths=lw)
            if col == 0:
                ax.set_ylabel(f"head {h}", fontsize=14)
            ax.set_xlim(-0.4, 0.4)
            ax.set_ylim(-0.4, 0.4)

    for ax in axes.ravel():
        ax.tick_params(
            axis='both', which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False
        )
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

def plot_softmax_prob_cls(n_samples, x, student, teacher, proj_head_s, proj_head_t, center_cls, temp_t, temp_s):
    x_sample = x[:n_samples].astype("float32")

    cls_t, _ = teacher(x_sample, training=False)
    z_teacher = proj_head_t(tf.expand_dims(cls_t, 1))[:, 0, :]
    prob_t = tf.nn.softmax((z_teacher - center_cls) / temp_t, axis=-1)
    mean_prob_t = tf.reduce_mean(prob_t, axis=0).numpy()

    cls_s, _ = student(x_sample, training=False)
    z_student = proj_head_s(tf.expand_dims(cls_s, 1))[:, 0, :]
    prob_s = tf.nn.softmax(z_student / temp_s, axis=-1)
    mean_prob_s = tf.reduce_mean(prob_s, axis=0).numpy()

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(z_teacher.shape[-1]), mean_prob_t, label="teacher", alpha=0.5)
    plt.bar(np.arange(z_student.shape[-1]), mean_prob_s, label="student", alpha=0.5)
    plt.yscale("log")
    plt.ylim(1e-10, 10)
    plt.legend()
    plt.title("mean softmax prob. of projected [CLS]")
    plt.xlabel("K-dim")
    
def plot_tSNE_cls(n_samples, backbone, x, y, alpha, marker_size, save_path=None):
    x_sample = x[:n_samples].astype("float32")
    y_sample = y[:n_samples].argmax(1)
    class_names = ['q', 'g', 'W', 'Z', 't']
    class_colors = {'q': 'C3', 'g': 'C1', 'W': 'C2', 'Z': 'C0', 't': 'C4'}

    present = [class_names[k] for k in range(len(class_names)) if np.any(y_sample == k)]

    cls_s, _ = backbone(x_sample, training=False)
    z_backbone = cls_s.numpy()
    tsne_backbone = TSNE(n_components=2, random_state=42).fit_transform(z_backbone)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
    ax0 = ax[0] if isinstance(ax, np.ndarray) else ax

    for c in present:
        k = class_names.index(c)
        ax0.scatter(
            tsne_backbone[y_sample == k, 0],
            tsne_backbone[y_sample == k, 1],
            s=marker_size, label=c, color=class_colors[c], alpha=alpha
        )

    #ax0.set_title("t-SNE of [CLS] embedding")
    ax0.legend(markerscale=2.5, loc='upper right')
    ax0.tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()
    
def plot_pca_corner_cls_embeddings(backbone, x, y, n_samples, n_components, alpha, marker_size):
    n_samples = min(n_samples, x.shape[0])
    x_sample = x[:n_samples].astype("float32")
    y_sample = y[:n_samples].argmax(1)

    class_names = ['q', 'g', 'W', 'Z', 't']
    class_colors = {'q': 'C3', 'g': 'C1', 'W': 'C2', 'Z': 'C0', 't': 'C4'}

    present = [class_names[k] for k in range(len(class_names)) if np.any(y_sample == k)]

    cls_s, _ = backbone(x_sample, training=False)
    z = cls_s.numpy()

    pca = PCA(n_components=n_components, random_state=42)
    z_pca = pca.fit_transform(z)

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    df = pd.DataFrame(z_pca, columns=pc_names)
    df["class"] = [class_names[i] for i in y_sample]

    g = sns.pairplot(
        df,
        vars=pc_names,
        hue="class",
        hue_order=present,
        palette=class_colors,
        diag_kind="kde",
        plot_kws=dict(alpha=alpha, s=marker_size, edgecolor="none"),
        diag_kws=dict(fill=False, alpha=1, linewidth=3),
        corner=True,
        height=3,
        aspect=1,
    )

    n = len(pc_names)
    for i in range(n):
        for j in range(n):
            ax = g.axes[i, j]
            if ax is None:
                continue
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(bottom=False, left=False)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_visible(True)

    axis_label_size = 16
    for j, pc in enumerate(pc_names):
        ax = g.axes[n-1, j]
        if ax is not None:
            ax.set_xlabel(pc, fontsize=axis_label_size)

    for i, pc in enumerate(pc_names):
        ax = g.axes[i, 0]
        if ax is not None:
            ax.set_ylabel(pc, fontsize=axis_label_size)

    leg = g._legend
    if leg is not None:
        leg.set_title("")
        handles = getattr(leg, "legend_handles", None)
        if handles is None:
            handles = getattr(leg, "legendHandles", [])

        for h in handles:
            if hasattr(h, "set_sizes"):
                h.set_sizes([60.0])
            if hasattr(h, "set_markersize"):
                h.set_markersize(12)

        for txt in leg.texts:
            txt.set_fontsize(20)

        leg.set_frame_on(True)
        frame = leg.get_frame()
        frame.set_linewidth(1)
        frame.set_edgecolor("black")
        frame.set_facecolor("white")
        frame.set_alpha(0.9)

        leg.set_bbox_to_anchor((0.9, 1.0), transform=g.fig.transFigure)
        leg._loc = 1

    g.fig.suptitle("PCA corner plot of [CLS] embedding", y=1.02, fontsize=20)
    plt.show()

def plot_sne_snapshots(snapshot_dir, x, y, save=False):
    d_model, n_heads, n_layers = map(int, snapshot_dir.split("_")[-3:])
    snapshots = sorted(
        glob.glob(os.path.join(snapshot_dir, "student_epoch*.weights.h5")),
        key=lambda p: int(re.search(r"epoch(\d+)", os.path.basename(p)).group(1))
    )
    for snapshot in snapshots:
        print(snapshot)
        student = build_backbone(d_model=d_model, n_heads=n_heads, n_layers=n_layers, name="backbone_student")
        student.load_weights(snapshot)
        save_path = None
        if save:
            epoch_str = re.search(r"epoch(\d+)", os.path.basename(snapshot)).group(1)
            save_path = os.path.join(snapshot_dir, f"student_epoch{epoch_str}.png")
        plot_tSNE_cls(n_samples=5000, backbone=student, x=x, y=y, alpha=0.5, marker_size=10, save_path=save_path)

def plot_training_histories(hist_standalone, history_ft, labels):
    (label_standalone, label_ft) = labels

    hs = hist_standalone.history

    #epochs_frozen = np.arange(1, len(hf["loss"]) + 1)
    epochs_standalone = np.arange(1, len(hs["loss"]) + 1)
    epochs_ft = np.arange(1, len(history_ft["train_loss"]) + 1)

    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(1, 2, 1)
    lw=2
    #ax1.plot(epochs_frozen, hf["loss"], label=f"{label_frozen} (train)", color="C0", linestyle="-", linewidth=lw)
    ax1.plot(epochs_standalone, hs["loss"], label=f"{label_standalone} (train)", color="C1", linestyle="-", linewidth=lw)
    ax1.plot(epochs_ft, history_ft["train_loss"], label=f"{label_ft} (train)", color="C2", linestyle="-", linewidth=lw)

    #ax1.plot(epochs_frozen, hf["val_loss"], label=f"{label_frozen} (val)", color="C0", linestyle="--", linewidth=lw)
    ax1.plot(epochs_standalone, hs["val_loss"], label=f"{label_standalone} (val)", color="C1", linestyle="--", linewidth=lw)
    ax1.plot(epochs_ft, history_ft["val_loss"], label=f"{label_ft} (val)", color="C2", linestyle="--", linewidth=lw)

    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.legend(fontsize=10)

    ax2 = plt.subplot(1, 2, 2)

    #ax2.plot(epochs_frozen, hf["accuracy"], label=f"{label_frozen} (train)", color="C0", linestyle="-", linewidth=lw)
    ax2.plot(epochs_standalone, hs["accuracy"], label=f"{label_standalone} (train)", color="C1", linestyle="-", linewidth=lw)
    ax2.plot(epochs_ft, history_ft["train_acc"], label=f"{label_ft} (train)", color="C2", linestyle="-", linewidth=lw)

    #ax2.plot(epochs_frozen, hf["val_accuracy"], label=f"{label_frozen} (val)", color="C0", linestyle="--", linewidth=lw)
    ax2.plot(epochs_standalone, hs["val_accuracy"], label=f"{label_standalone} (val)", color="C1", linestyle="--", linewidth=lw)
    ax2.plot(epochs_ft, history_ft["val_acc"], label=f"{label_ft} (val)", color="C2", linestyle="--", linewidth=lw)

    ax2.set_xlabel("Epoch", fontsize=14)
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_roc_ft_5class(y_test, y_pred_standalone, y_pred_ft, save_path=None):
    plt.figure(figsize=(7,6))
    class_names = ['q','g','W','Z','t']
    class_colors = ['C3','C1','C2','C0','C4']
    
    for k,c in enumerate(class_names):
        fpr_standalone, tpr_standalone, _ = roc_curve(y_test[:,k], y_pred_standalone[:,k])
        auc_standalone = auc(fpr_standalone, tpr_standalone)
        plt.plot(tpr_standalone, fpr_standalone, color=class_colors[k], lw=1.5, linestyle='dashed', label=f"{c} [Supervised] ({auc_standalone:.4f})")
    
        fpr_ft, tpr_ft, _ = roc_curve(y_test[:,k], y_pred_ft[:,k])
        auc_ft = auc(fpr_ft, tpr_ft)
        plt.plot(tpr_ft, fpr_ft, color=class_colors[k], lw=1.5, linestyle='-', label=f"{c} [jBOT] ({auc_ft:.4f})")
    
    plt.xlabel("TPR", size=16)
    plt.ylabel("FPR", size=16)
    plt.yscale("log")
    plt.ylim(1e-4,1)
    plt.title("MLP head + [CLS] embedding", size=16)
    plt.legend(fontsize=9, loc='lower right')
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

def plot_roc_ft_5class_kfold(y_test, y_pred_ft, k_folds, save_path=None):
    plt.figure(figsize=(5.5,5))
    class_names = ['q','g','W','Z','t']
    class_colors = ['C3','C1','C2','C0','C4']

    if k_folds is None:
        for k, c in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test[:, k], y_pred_ft[:, k])
            plt.plot(tpr, fpr, color=class_colors[k], lw=1.5, linestyle='-',
                     label=f"{c} ({auc(fpr,tpr):.4f})")
    else:
        y_idx = y_test.argmax(1)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fpr_grid = np.logspace(-4, 0, 400)

        for k, c in enumerate(class_names):
            tprs = []
            aucs = []
            for _, te_idx in skf.split(np.zeros_like(y_idx), y_idx):
                y_te = y_test[te_idx, k]
                p_te = y_pred_ft[te_idx, k]
                fpr, tpr, _ = roc_curve(y_te, p_te)

                tpr_i = np.interp(fpr_grid, fpr, tpr, left=0.0, right=1.0)
                tprs.append(tpr_i)
                aucs.append(auc(fpr, tpr))

            tprs = np.asarray(tprs)
            tpr_mu = tprs.mean(axis=0)
            tpr_sd = tprs.std(axis=0)

            plt.plot(tpr_mu, fpr_grid, color=class_colors[k], lw=1.5, linestyle='-',
                     label=f"{c} ({np.mean(aucs):.4f}$\pm${np.std(aucs):.4f})")

            tpr_lo = np.clip(tpr_mu - tpr_sd, 0.0, 1.0)
            tpr_hi = np.clip(tpr_mu + tpr_sd, 0.0, 1.0)
            plt.fill_betweenx(fpr_grid, tpr_lo, tpr_hi, color=class_colors[k], alpha=0.2)

    plt.xlabel("TPR", size=18)
    plt.ylabel("FPR", size=18)
    plt.yscale("log")
    plt.ylim(1e-4, 1)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=10.5, loc='lower right')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

def plot_roc_ft_tqg(y_test_2, y_pred_standalone, y_pred_ft, save_path=None):
    plt.figure(figsize=(7,6))
    pos_k = 1
    
    fpr, tpr, _ = roc_curve(y_test_2[:,pos_k], y_pred_standalone[:,pos_k])
    plt.plot(tpr, fpr, lw=1.8, linestyle='-', label=f"top [Supervised] ({auc(fpr,tpr):.4f})")
    
    fpr, tpr, _ = roc_curve(y_test_2[:,pos_k], y_pred_ft[:,pos_k])
    plt.plot(tpr, fpr, lw=1.8, linestyle='--', label=f"top [jBOT] ({auc(fpr,tpr):.4f})")
    
    plt.xlabel("TPR", size=16)
    plt.ylabel("FPR", size=16)
    plt.yscale("log")
    plt.ylim(1e-4,1)
    plt.title("MLP head + CLS embedding", size=16)
    plt.legend(fontsize=10, loc="lower right")
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

def plot_roc_tqg_kfold(
    backbone,
    x_train,
    y_train_top,
    x_test,
    y_test_2,
    y_pred_standalone,
    y_pred_ft,
    knn_n,
    k_folds,
    seed,
    save_path=None,
):
    z_train = backbone.predict(x_train)[0]
    z_test = backbone.predict(x_test)[0]

    knn = KNeighborsClassifier(n_neighbors=knn_n).fit(z_train, y_train_top)
    knn_p = knn.predict_proba(z_test)

    scaler = StandardScaler().fit(z_train)
    Z_train = scaler.transform(z_train)
    Z_test = scaler.transform(z_test)
    logreg = LogisticRegression(max_iter=2000).fit(Z_train, y_train_top)
    linear_p = logreg.predict_proba(Z_test)

    y_pos = y_test_2[:, 1].astype(int)
    y_idx = y_pos

    fpr_grid = np.logspace(-4, 0, 400)

    prob_dict = {
        "jBOT [$k$-NN]": knn_p,
        "jBOT [linear]": linear_p,
        "Supervised": y_pred_standalone,
        "jBOT [FT]": y_pred_ft,
    }

    plt.figure(figsize=(5.5,5))
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for name, P in prob_dict.items():
        s = P[:, 1]

        tprs = []
        aucs = []
        for _, te_idx in skf.split(np.zeros_like(y_idx), y_idx):
            y_te = y_pos[te_idx]
            s_te = s[te_idx]
            fpr, tpr, _ = roc_curve(y_te, s_te)

            tpr_i = np.interp(fpr_grid, fpr, tpr, left=0.0, right=1.0)
            tprs.append(tpr_i)
            aucs.append(auc(fpr, tpr))

        tprs = np.asarray(tprs)
        mu = tprs.mean(axis=0)
        sd = tprs.std(axis=0)

        plt.plot(mu, fpr_grid, lw=1.5, label=f"{name} ({np.mean(aucs):.4f}$\pm${np.std(aucs):.4f})")
        plt.fill_betweenx(
            fpr_grid,
            np.clip(mu - sd, 0.0, 1.0),
            np.clip(mu + sd, 0.0, 1.0),
            alpha=0.2
        )

    plt.xlabel("TPR", size=18)
    plt.ylabel("FPR", size=18)
    plt.yscale("log")
    plt.ylim(1e-4, 1)
    plt.xlim(0, 1)
    #plt.title("Pretrained [CLS] embedding", size=15)
    plt.legend(fontsize=10.5, loc="lower right")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

def plot_ad_score_hist(score, y5, title, xlim=None, save_path=None):    
    yi = np.argmax(y5, axis=1)
    m_q = (yi == 0); m_g = (yi == 1); m_w = (yi == 2); m_z = (yi == 3); m_t = (yi == 4)
    groups = {"QCD": score[m_q | m_g], "W": score[m_w], "Z": score[m_z], "t": score[m_t]}

    #colors = {"QCD": "black", "W": 'C2', "Z": 'C0', "t": 'C4'}
    
    plt.figure(figsize=(5, 4))
    for label in ["QCD", "W", "Z", "t"]:
        plt.hist(groups[label], bins=100, density=True, histtype="step", linewidth=2, label=f"{label}")

    plt.xlabel("Anomaly score", fontsize=14)
    plt.ylabel("Density", fontsize=16)
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    plt.legend(loc="upper right", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

def plot_ad_roc_one_score_three_signals(score, y5, title):
    class_colors = {'q': 'C3', 'g': 'C1', 'W': 'C2', 'Z': 'C0', 't': 'C4'}
    yi = np.argmax(y5, axis=1)
    bkg = (yi == 0) | (yi == 1)

    plt.figure(figsize=(7, 6))
    for sig_name, sig_idx in [("W", 2), ("Z", 3), ("t", 4)]:
        sig = (yi == sig_idx)
        m = bkg | sig
        y_bin = sig[m].astype(int)
        s = np.asarray(score)[m]

        fpr, tpr, _ = roc_curve(y_bin, s)
        plt.plot(tpr, fpr, label=f"{sig_name} (AUC={auc(fpr,tpr):.4f})", lw=1.5)

    plt.yscale("log")
    plt.ylim(1e-3, 1)
    plt.xlim(0, 1)
    plt.xlabel("TPR", fontsize=16)
    plt.ylabel("FPR", fontsize=16)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_ad_roc_mult_scores_one_signal(scores_dict, y5, signal, kfold=None, save_path=None):
    name_to_idx = {"W": 2, "Z": 3, "t": 4}
    yi = np.argmax(y5, axis=1)
    bkg = (yi == 0) | (yi == 1)

    if signal in name_to_idx:
        sig = (yi == name_to_idx[signal])
        title_sig = signal
    if signal == "combined":
        sig = (yi == 2) | (yi == 3) | (yi == 4)
        title_sig = "Combined signal"

    m = bkg | sig
    y_bin = sig[m]

    plt.figure(figsize=(5.5, 5))

    if kfold is None:
        for name, score in scores_dict.items():
            s = np.asarray(score)[m]
            fpr, tpr, _ = roc_curve(y_bin, s)
            plt.plot(tpr, fpr, label=f"{name} ({auc(fpr, tpr):.4f})", lw=1.5)

    else:
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        tpr_grid = np.linspace(0.0, 1.0, 500)

        for name, score in scores_dict.items():
            s_all = np.asarray(score)[m]

            fprs_interp = []
            aucs = []
            for _, te_idx in skf.split(np.zeros_like(y_bin), y_bin):
                y_te = y_bin[te_idx]
                s_te = s_all[te_idx]

                fpr, tpr, _ = roc_curve(y_te, s_te)
                aucs.append(auc(fpr, tpr))

                order = np.argsort(tpr)
                tpr_s = tpr[order]
                fpr_s = fpr[order]

                if tpr_s[0] > 0:
                    tpr_s = np.r_[0.0, tpr_s]
                    fpr_s = np.r_[fpr_s[0], fpr_s]
                if tpr_s[-1] < 1:
                    tpr_s = np.r_[tpr_s, 1.0]
                    fpr_s = np.r_[fpr_s, fpr_s[-1]]

                fprs_interp.append(np.interp(tpr_grid, tpr_s, fpr_s))

            fprs_interp = np.asarray(fprs_interp)
            mu = fprs_interp.mean(axis=0)
            sd = fprs_interp.std(axis=0)

            line, = plt.plot(
                tpr_grid, mu,
                label=f"{name} ({np.mean(aucs):.4f}$\pm${np.std(aucs):.4f})",
                lw=1.5, alpha=0.7
            )
            c = line.get_color()
            low = np.clip(mu - sd, 1e-6, None)
            high = np.clip(mu + sd, 1e-6, None)
            plt.fill_between(tpr_grid, low, high, color=c, alpha=0.2, linewidth=0)

    plt.yscale("log")
    plt.ylim(1e-3, 1)
    plt.xlim(0, 1)
    plt.xlabel("TPR", fontsize=18)
    plt.ylabel("FPR", fontsize=18)
    plt.title(f"{title_sig} vs QCD", fontsize=18)
    plt.legend(loc="lower right", fontsize=13)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

#-------------------------------------- evaluation --------------------------------------

def maha_classifier_prob(z_train,
                         y_train_idx,
                         z_test,
                         cov_tied,
                         reg,
                         l2norm,
                         temp):
    C = np.max(y_train_idx) + 1

    def _l2_normalize(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(n, 1e-12)

    if l2norm:
        z_train = _l2_normalize(z_train)
        z_test = _l2_normalize(z_test)

    # class means
    mus = np.zeros((C, z_train.shape[1]), dtype=np.float32)
    counts = np.zeros((C,), dtype=np.int32)
    for c in range(C):
        m = (y_train_idx == c)
        counts[c] = m.sum()
        mus[c] = z_train[m].mean(axis=0)

    def _chol(cov):
        cov = cov + reg * np.eye(cov.shape[0], dtype=cov.dtype)
        return np.linalg.cholesky(cov)

    def _md2_from_chol(z, mu, L):
        Yc = (z - mu[None, :]).T 
        v = np.linalg.solve(L, Yc)
        return np.sum(v*v, axis=0)

    # maha distances
    N, d = z_test.shape
    md2 = np.zeros((N, C), dtype=np.float32)

    if cov_tied:
        # shared covariance across classes
        Xc = np.empty_like(z_train)
        for c in range(C):
            m = (y_train_idx == c)
            Xc[m] = z_train[m] - mus[c][None, :]
        denom = max(1, z_train.shape[0] - C)
        cov = (Xc.T @ Xc) / denom
        L = _chol(cov)

        for c in range(C):
            md2[:, c] = _md2_from_chol(z_test, mus[c], L)

    else:
        # per-class covariance
        for c in range(C):
            m = (y_train_idx == c)
            X = z_train[m] - mus[c][None, :]
            denom = max(1, X.shape[0] - 1)
            cov = (X.T @ X) / denom
            L = _chol(cov)
            md2[:, c] = _md2_from_chol(z_test, mus[c], L)

    # softmax over -0.5*md2
    logits = (-0.5 * md2) / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    prob = expv / expv.sum(axis=1, keepdims=True)

    return prob
    
def pretrain_probe_5class(backbone, x_train, y_train, x_test, y_test, knn_n, k_folds=10):
    def _acc_auc_kfold(y_onehot, prob, k_folds=10, seed=42):
        y_idx = y_onehot.argmax(1)
        C = y_onehot.shape[1]
    
        y_pred = prob.argmax(1)
        acc_full = accuracy_score(y_idx, y_pred)
        auc_full = np.zeros((C,), dtype=np.float32)
        for k in range(C):
            fpr, tpr, _ = roc_curve(y_onehot[:, k], prob[:, k])
            auc_full[k] = auc(fpr, tpr)
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
        accs = []
        aucs = []
        for _, te_idx in skf.split(np.zeros_like(y_idx), y_idx):
            y_te = y_onehot[te_idx]
            p_te = prob[te_idx]
    
            accs.append(accuracy_score(y_te.argmax(1), p_te.argmax(1)))
            fold_auc = np.full((C,), np.nan, dtype=np.float32)
            for k in range(C):
                fpr, tpr, _ = roc_curve(y_te[:, k], p_te[:, k])
                fold_auc[k] = auc(fpr, tpr)
            aucs.append(fold_auc)
    
        accs = np.asarray(accs, dtype=np.float32)
        aucs = np.asarray(aucs, dtype=np.float32)
    
        acc_kf_mean = np.nanmean(accs)
        acc_kf_std = np.nanstd(accs)
        auc_kf_mean = np.nanmean(aucs, axis=0)
        auc_kf_std = np.nanstd(aucs, axis=0)
    
        return acc_full, acc_kf_mean, acc_kf_std, auc_full, auc_kf_mean, auc_kf_std
        
    z_cls_train = backbone.predict(x_train)[0]
    z_cls_test = backbone.predict(x_test)[0]

    y_train_idx = y_train.argmax(1)
    y_test_idx = y_test.argmax(1)

    knn = KNeighborsClassifier(n_neighbors=knn_n).fit(z_cls_train, y_train_idx)
    knn_p = knn.predict_proba(z_cls_test)

    scaler = StandardScaler().fit(z_cls_train)
    Z_cls_train = scaler.transform(z_cls_train)
    Z_cls_test  = scaler.transform(z_cls_test)
    logreg = LogisticRegression(max_iter=1000).fit(Z_cls_train, y_train_idx)
    linear_p = logreg.predict_proba(Z_cls_test)

    maha_p = maha_classifier_prob(z_cls_train, y_train_idx, z_cls_test, cov_tied=True, reg=1e-8, l2norm=True, temp=1.0)

    def _print_block(name, prob):
        acc_full, acc_mu, acc_sd, auc_full, auc_mu, auc_sd = _acc_auc_kfold(
            y_test, prob, k_folds=k_folds, seed=42
        )
        print(f"{name:7s} acc: {acc_full:.4f} ({acc_mu:.4f} +/- {acc_sd:.4f})")

        class_names = ['q', 'g', 'W', 'Z', 't']
        auc_str = " | ".join([f"{c}:{auc_full[i]:.4f} ({auc_mu[i]:.4f} +/- {auc_sd[i]:.4f})" for i, c in enumerate(class_names)])
        print(f"{name:7s} AUC: {auc_str}")

    _print_block("maha", maha_p); print()
    _print_block("k-NN", knn_p); print()
    _print_block("linear",linear_p); print()

    print_effsig_table(y_test, {"kNN": knn_p, "linear": linear_p, "maha": maha_p}, (0.1, 0.01, 0.001))

    plt.figure(figsize=(7,6))
    class_names = ['q', 'g', 'W', 'Z', 't']
    class_colors = {'q': 'C3', 'g': 'C1', 'W': 'C2', 'Z': 'C0', 't': 'C4'}

    for k, c in enumerate(class_names):
        fpr_knn, tpr_knn, _ = roc_curve(y_test[:, k], knn_p[:, k])
        plt.plot(tpr_knn, fpr_knn, color=class_colors[c],
                 label=f"{c} [k-NN] ({auc(fpr_knn, tpr_knn):.4f})", linestyle="-", lw=1.5)

        fpr_linear, tpr_linear, _ = roc_curve(y_test[:, k], linear_p[:, k])
        plt.plot(tpr_linear, fpr_linear, color=class_colors[c],
                 label=f"{c} [linear] ({auc(fpr_linear, tpr_linear):.4f})", linestyle="--", lw=1.5)

        fpr_maha, tpr_maha, _ = roc_curve(y_test[:, k], maha_p[:, k])
        plt.plot(tpr_maha, fpr_maha, color=class_colors[c],
                 label=f"{c} [maha] ({auc(fpr_maha, tpr_maha):.4f})", linestyle="dotted", lw=1.5)

    plt.xlabel("TPR", size=16)
    plt.ylabel("FPR", size=16)
    plt.yscale("log")
    plt.ylim(1e-4, 1)
    plt.title("Pretrained [CLS] embedding", size=15)
    plt.legend(fontsize=10, loc='lower right')
    plt.show()

def pretrain_probe_tqg(backbone, x_train, y_train_top, x_test, y_test_top, y_test_2, knn_n):
    z_cls_train = backbone.predict(x_train)[0]
    z_cls_test = backbone.predict(x_test)[0]
    
    knn = KNeighborsClassifier(n_neighbors=knn_n).fit(z_cls_train, y_train_top)
    acc_knn = accuracy_score(y_test_top, knn.predict(z_cls_test))
    knn_p = knn.predict_proba(z_cls_test)
    
    scaler = StandardScaler().fit(z_cls_train)
    Z_train = scaler.transform(z_cls_train)
    Z_test = scaler.transform(z_cls_test)
    logreg = LogisticRegression(max_iter=2000).fit(Z_train, y_train_top)
    acc_linear = accuracy_score(y_test_top, logreg.predict(Z_test))
    linear_p = logreg.predict_proba(Z_test)
    
    maha_p = maha_classifier_prob(z_cls_train, y_train_top, z_cls_test, cov_tied=True, reg=1e-8, l2norm=True, temp=1.0)
    acc_maha = accuracy_score(y_test_top, maha_p.argmax(1))
    
    #print(f"maha acc:   {acc_maha:.4f}")
    #print(f"linear acc: {acc_linear:.4f}")
    #print(f"k-NN acc:   {acc_knn:.4f}")
    
    report_acc_eff(y_test_2, {"maha": maha_p, "linear": linear_p, "k-NN": knn_p}, ("QCD","t"), (0.1, 0.01, 0.001), 10)

    pos_k = 1
    plt.figure(figsize=(7,6))
    
    fpr, tpr, _ = roc_curve(y_test_2[:, pos_k], knn_p[:, pos_k])
    plt.plot(tpr, fpr, lw=1.5, linestyle="-", label=f"top [k-NN] ({auc(fpr,tpr):.4f})")
    
    fpr, tpr, _ = roc_curve(y_test_2[:, pos_k], linear_p[:, pos_k])
    plt.plot(tpr, fpr, lw=1.5, linestyle="--", label=f"top [linear] ({auc(fpr,tpr):.4f})")
    
    fpr, tpr, _ = roc_curve(y_test_2[:, pos_k], maha_p[:, pos_k])
    plt.plot(tpr, fpr, lw=1.5, linestyle="dotted", label=f"top [maha] ({auc(fpr,tpr):.4f})")

    plt.xlabel("TPR", size=16)
    plt.ylabel("FPR", size=16)
    plt.yscale("log")
    plt.ylim(1e-4, 1)
    plt.title("Pretrained [CLS] embedding", size=15)
    plt.legend(fontsize=10, loc="lower right")
    plt.show()

def report_acc_eff(y_true, prob_dict, class_names, eff_bkg_targets, k_folds):
    col_w=22; method_w=14
    y_true_cls = y_true.argmax(axis=1)
    C = y_true.shape[1]

    model_order = list(prob_dict.keys())
    model_name_map = {}

    def _display_name(key):
        return model_name_map.get(key, key)

    def _eff_signal_triplet(y_true_bin, score, targets):
        fpr, tpr, _ = roc_curve(y_true_bin, score)
        out = []
        for eb in targets:
            ok = np.where(fpr <= eb)[0]
            out.append(np.max(tpr[ok]) if ok.size else 0)
        return out

    def _targets_str(targets):
        return " ".join([f"{t:0.4f}" for t in targets])

    print("Accuracy:")
    max_name = max(len(_display_name(k)) for k in model_order)
    name_w = max(12, max_name) + 1

    for key in model_order:
        P = np.asarray(prob_dict[key])
        y_pred_cls = P.argmax(axis=1)
        acc = accuracy_score(y_true_cls, y_pred_cls)

        if k_folds > 1:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            accs = []
            for _, te_idx in skf.split(np.zeros_like(y_true_cls), y_true_cls):
                accs.append(accuracy_score(y_true_cls[te_idx], y_pred_cls[te_idx]))
            accs = np.asarray(accs)
            print(f"{_display_name(key):<{name_w}s} {acc:.4f} ({accs.mean():.4f} +/- {accs.std():.4f})")
        else:
            print(f"{_display_name(key):<{name_w}s} {acc:.4f}")
    print()

    print("AUC:")
    if C == 2:
        sel_idx = [1]
        sel_names = [class_names[1]]
    else:
        sel_idx = list(range(C))
        sel_names = list(class_names)
    for key in model_order:
        P = np.asarray(prob_dict[key])

        aucs = []
        for k in sel_idx:
            y_true_bin = (y_true[:, k] == 1).astype(np.int32)
            aucs.append(roc_auc_score(y_true_bin, P[:, k]))
        aucs = np.asarray(aucs, dtype=np.float32)

        if k_folds > 1:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_aucs = []
            for _, te_idx in skf.split(np.zeros_like(y_true_cls), y_true_cls):
                y_te = y_true[te_idx]
                P_te = P[te_idx]
                a = []
                for k in sel_idx:
                    y_bin = (y_te[:, k] == 1).astype(np.int32)
                    a.append(roc_auc_score(y_bin, P_te[:, k]))
                fold_aucs.append(a)
            fold_aucs = np.asarray(fold_aucs, dtype=np.float32)

            parts = []
            for i, cname in enumerate(sel_names):
                parts.append(f"{cname} {aucs[i]:.4f} ({fold_aucs[:, i].mean():.4f} +/- {fold_aucs[:, i].std():.4f})")
            print(f"{_display_name(key):<{name_w}s} " + " | ".join(parts))
        else:
            parts = [f"{cname} {aucs[i]:.4f}" for i, cname in enumerate(sel_names)]
            print(f"{_display_name(key):<{name_w}s} " + " | ".join(parts))
    print()

    if C == 2:
        sel_idx = [1] # for t-vs-qg, select t as positive class
        sel_names = [class_names[1]]
    else:
        sel_idx = list(range(C))
        sel_names = list(class_names)

    # eff without k-fold
    per_model_cells = {}
    for key in model_order:
        P = np.asarray(prob_dict[key])
        cells = []
        for k in sel_idx:
            y_true_bin = (y_true[:, k] == 1)
            vals = _eff_signal_triplet(y_true_bin, P[:, k], eff_bkg_targets)
            cells.append(" ".join([f"{v:.4f}" for v in vals]))
        per_model_cells[key] = cells

    max_label = 0
    for key in model_order:
        max_label = max(max_label, len(_display_name(key)))
    method_w_local = max(method_w, max_label + 1)

    indent = " " * method_w_local
    hdr1 = indent + "".join([f"{c:^{col_w}s}" for c in sel_names])
    hdr2 = indent + "".join([f"{_targets_str(eff_bkg_targets):>{col_w}s}" for _ in sel_names])
    print(hdr1)
    print(hdr2)
    print("-" * len(hdr1))

    for key in model_order:
        name = _display_name(key)
        line = f"{name:<{method_w_local}s}" + "".join([f"{cell:>{col_w}s}" for cell in per_model_cells[key]])
        print(line)
    print()

    # eff with k-fold
    if k_folds > 1:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        model_fold_eff = {}
        for key in model_order:
            P = np.asarray(prob_dict[key])
            fold_vals = []
            for _, te_idx in skf.split(np.zeros_like(y_true_cls), y_true_cls):
                y_te = y_true[te_idx]
                P_te = P[te_idx]
                fold_mat = np.zeros((len(sel_idx), len(eff_bkg_targets)))
                for ii, k in enumerate(sel_idx):
                    y_true_bin = (y_te[:, k] == 1)
                    fold_mat[ii, :] = _eff_signal_triplet(y_true_bin, P_te[:, k], eff_bkg_targets)
                fold_vals.append(fold_mat)
            model_fold_eff[key] = np.stack(fold_vals, axis=0)

        max_label = 0
        for key in model_order:
            disp = _display_name(key)
            max_label = max(max_label, len(f"{disp} mean"), len(f"{disp} std"))
        method_w_local = max(method_w, max_label + 1)

        indent = " " * method_w_local
        hdr1 = indent + "".join([f"{c:^{col_w}s}" for c in sel_names])
        hdr2 = indent + "".join([f"{_targets_str(eff_bkg_targets):>{col_w}s}" for _ in sel_names])
        print(hdr1)
        print(hdr2)
        print("-" * len(hdr1))

        for key in model_order:
            disp = _display_name(key)
            X = model_fold_eff[key]
            mu = X.mean(axis=0)
            std = X.std(axis=0)

            mu_cells = [" ".join([f"{mu[i, j]:.4f}" for j in range(mu.shape[1])]) for i in range(mu.shape[0])]
            std_cells = [" ".join([f"{std[i, j]:.4f}" for j in range(std.shape[1])]) for i in range(std.shape[0])]

            line_mu = f"{(disp + ' mean'):<{method_w_local}s}" + "".join([f"{cell:>{col_w}s}" for cell in mu_cells])
            line_std = f"{(disp + ' std'):<{method_w_local}s}"  + "".join([f"{cell:>{col_w}s}" for cell in std_cells])

            print(line_mu)
            print(line_std)

def print_effsig_table(y_test, prob_dict, effbkg_targets):
    def eff_signal_at_eff_bkg(y_true_bin, score, targets):
        fpr, tpr, _ = roc_curve(y_true_bin, score)
        out = {}
        for eb in targets:
            ok = np.where(fpr <= eb)[0]
            out[eb] = np.max(tpr[ok]) if ok.size else 0
        return out
        
    class_names=("q","g","W","Z","t"); col_w=26; method_w=12; val_w=6; prec=4
    fmt = f"{{:>{val_w}.{prec}f}}"
    eff_hdr_triplet = " ".join([fmt.format(t) for t in effbkg_targets])

    header1 = " " * method_w + "".join([f"{c:^{col_w}s}" for c in class_names])
    header2 = " " * method_w + "".join([f"{eff_hdr_triplet:>{col_w}s}" for _ in class_names])
    print(header1)
    print(header2)
    print("-" * len(header1))

    for method, p in prob_dict.items():
        row_cells = []
        for k in range(len(class_names)):
            y_true = y_test[:, k] == 1
            d = eff_signal_at_eff_bkg(y_true, p[:, k], effbkg_targets)
            vals = [d[t] for t in effbkg_targets]
            cell = " ".join([fmt.format(v) for v in vals])
            row_cells.append(cell)

        line = f"{method:<{method_w}s}" + "".join([f"{cell:>{col_w}s}" for cell in row_cells])
        print(line)