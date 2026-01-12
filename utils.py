import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
import sys
import re, glob
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import h5py
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import keras
from keras.layers import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize, StandardScaler
from keras.utils import Progbar

#-------------------------------------- data preparation --------------------------------------

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

#-------------------------------------- augmentations --------------------------------------

RNG = np.random.default_rng(42)

def rotate_eta_phi(jet):
    theta = RNG.uniform(-np.pi, np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    eta, phi, pt, valid = jet.T
    eta_r = cos_t*eta - sin_t*phi
    phi_r = sin_t*eta + cos_t*phi
    jet_r = np.vstack([eta_r, phi_r, pt, valid]).T
    return jet_r

def gaussian_smear_eta_phi(jet):
    eta, phi, pt, valid = jet.T
    sigma = 3e-4 / np.clip(pt, 1e-6, None) # 1e-4 = (QCD_scale 100 MeV) / (1 TeV); pt is particle pt / jet pt
    eta_s = eta + RNG.normal(0., sigma)
    phi_s = phi + RNG.normal(0., sigma)
    phi_s = np.where(phi_s > np.pi, phi_s - 2*np.pi,
            np.where(phi_s <= -np.pi, phi_s + 2*np.pi, phi_s))
    jet_s = np.vstack([eta_s, phi_s, pt, valid]).T
    return jet_s

def split_particles(jet):
    eta, phi, pt, valid = jet.T
    valid_idx = np.where(valid==1)[0]
    n_valid = valid_idx.size
    n_empty = 30 - n_valid
    if n_empty==0:
        return jet.copy().astype(np.float32)

    n_split  = min(n_empty, n_valid)
    to_split = RNG.choice(valid_idx, n_split, replace=False)

    extra = []
    for idx in to_split:
        a = RNG.uniform(0.3, 0.7)
        pt1 = a*pt[idx]
        pt2 = (1-a)*pt[idx]
        pt[idx] = pt1
        extra.append([eta[idx], phi[idx], pt2, 1.])

    jet_new = np.concatenate([jet, np.array(extra, dtype=np.float32)], axis=0)

    if jet_new.shape[0]<30:
        pad = np.zeros((30 - jet_new.shape[0], 4), dtype=np.float32)
        jet_new = np.concatenate([jet_new, pad], axis=0)
    return jet_new[:30].astype(np.float32)

def augment_jet(jet):
    jet = rotate_eta_phi(jet)
    jet = gaussian_smear_eta_phi(jet)
    jet = split_particles(jet)
    return jet.astype(np.float32)

#-------------------------------------- patch masking --------------------------------------

@tf.function
def tf_make_masks_pt_ratio(jets, p_mask, ratio_range):
    jets = tf.cast(jets, tf.float32)
    pt = jets[..., 2]
    valid = jets[..., 3] > 0.5
    pt = tf.where(valid, pt, 0.)

    B = tf.shape(jets)[0]

    # p_mask: r~Uniform[rmin,rmax]; (1-p_mask): r=0
    p_mask = tf.cast(p_mask, tf.float32)
    rmin = tf.cast(ratio_range[0], tf.float32)
    rmax = tf.cast(ratio_range[1], tf.float32)

    do_mask = tf.random.uniform([B]) < p_mask
    r = tf.where(do_mask, tf.random.uniform([B], rmin, rmax), 0.0) # (B,)

    total_pt = tf.reduce_sum(pt, axis=-1) # (B,)
    target = r * total_pt # (B,)

    # random permutation among valid, invalid pushed to end
    keys = tf.random.uniform([B, 30])
    keys = tf.where(valid, keys, 2.0) # set invalid keys to a large number like 2 so sorted last
    order = tf.argsort(keys, axis=-1) # (B,30)

    pt_perm = tf.gather(pt, order, batch_dims=1) # (B,30)
    csum = tf.cumsum(pt_perm, axis=-1) # (B,30)

    # find boundary index k where cumulative sum crosses target
    hit = csum >= target[:, None] # (B,30)
    any_hit = tf.reduce_any(hit, axis=-1) # (B,)
    k_first = tf.argmax(tf.cast(hit, tf.int32), axis=-1, output_type=tf.int32) # (B,)

    # if it never crosses, fallback to last_valid
    n_valid = tf.reduce_sum(tf.cast(valid, tf.int32), axis=-1) # (B,)
    last_valid = n_valid - 1
    k = tf.where(any_hit, k_first, last_valid)

    # compare taking k tokens vs (k+1) tokens (closest-to-target choice)
    b = tf.range(B)
    curr_sum = tf.gather_nd(csum, tf.stack([b, k], axis=1)) # sum of first (k+1) perm tokens

    k_prev = tf.maximum(k-1, 0)
    prev_sum_raw = tf.gather_nd(csum, tf.stack([b, k_prev], axis=1)) # sum of first k perm tokens
    prev_sum = tf.where(k > 0, prev_sum_raw, 0.0)

    use_curr = tf.abs(curr_sum - target) <= tf.abs(prev_sum - target)
    n_take = tf.where(use_curr, k+1, k) # (B,)

    # enforce at least 1 masked and at least 1 unmasked; if r=0 then mask none
    n_take = tf.maximum(n_take, 1)
    n_take = tf.minimum(n_take, n_valid - 1)
    n_take = tf.where(r > 0.0, n_take, 0)

    # build mask in perm space: first n_take are masked
    mask_sorted = tf.sequence_mask(n_take, maxlen=30) # (B,30) bool, perm order

    # map back to original indices
    inv = tf.argsort(order, axis=-1) # inverse permutation
    mask = tf.gather(mask_sorted, inv, batch_dims=1) # (B,30) original order

    return mask

#-------------------------------------- model building --------------------------------------

class AddClsToken(Layer):
    # add [CLS] token: (batch, n_tokens, d_model) -> (batch, n_tokens+1, d_model)
    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.cls = self.add_weight(name="cls_token", shape=(1, 1, d_model), initializer="zeros", trainable=True)

    def call(self, x):
        # tile along the batch dimension
        cls_token = tf.tile(self.cls, [tf.shape(x)[0], 1, 1])
        return tf.concat([cls_token, x], axis=1)

class TransformerBlock(Layer):
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(n_heads, d_model//n_heads)
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([Dense(d_model*4, activation='gelu'), Dense(d_model)])
        self.drop1 = Dropout(0.2)
        self.drop2 = Dropout(0.2)

    def call(self, x):
        attn_out = self.mha(x, x)
        attn_out = self.ln1(x + self.drop1(attn_out))
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ln2(attn_out + self.drop2(ffn_out))
        return ffn_out

def build_backbone(d_model, n_heads, n_layers, name="backbone"):
    x_in = keras.Input((30, 4), name='particles_in') 
    x = Dense(d_model, name="token_embedding")(x_in)
    x = AddClsToken(d_model, name="prepend_cls")(x)
    for i in range(n_layers):
        x = TransformerBlock(d_model, n_heads, name=f"block_{i}")(x)

    cls_out = x[:, 0]
    particle_out = x[:, 1:]
    return keras.Model(x_in, [cls_out, particle_out], name=name)

def build_proj_head(d_in, d_proj, name):
    head = keras.Sequential([
        Dense(d_proj*8, activation='gelu'),
        Dense(d_proj),
        Lambda(lambda t: tf.math.l2_normalize(t, -1))
    ], name=name)
    head.build((None, None, d_in))
    return head

class MaskTokens(keras.layers.Layer):
    def __init__(self, d_model, init_std=0.02, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.init_std = init_std

    def build(self, input_shape):
        self.mask_token = self.add_weight(
            name="mask_token",
            shape=(self.d_model,),
            initializer=keras.initializers.RandomNormal(stddev=self.init_std),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, embeddings, mask_bool):
        float_mask = tf.cast(mask_bool, embeddings.dtype)[..., None]
        return embeddings * (1.0 - float_mask) + self.mask_token[None, None, :] * float_mask

def build_mlp(dim_in, n_classes, name="mlp"):
    x_in = Input(shape=(dim_in,))
    x = Dense(dim_in*2, activation='gelu')(x_in)
    x = Dropout(0.2)(x)
    x = Dense(dim_in*1, activation='gelu')(x)
    x = Dropout(0.2)(x)
    x = Dense(n_classes, activation='softmax')(x)
    return tf.keras.models.Model(x_in, x, name=name)

def load_finetune(backbone_file, mlp_file, d_model, n_heads, n_layers, n_classes):
    backbone = build_backbone(d_model, n_heads, n_layers, name="backbone")
    backbone.load_weights(f"models/{backbone_file}.weights.h5")
    mlp = build_mlp(d_model, n_classes=n_classes, name="mlp")
    mlp.load_weights(f"models/{mlp_file}.weights.h5")
    x_in = tf.keras.Input((30,4))
    cls, _ = backbone(x_in)
    x_out = mlp(cls)
    model = tf.keras.models.Model(x_in, x_out, name="model")
    return backbone, mlp, model

#-------------------------------------- training --------------------------------------

def tf_augment_batch(jets):
    def aug_(j):
        out = tf.numpy_function(augment_jet, [j], tf.float32)
        out.set_shape((30, 4))
        return out
    return tf.map_fn(aug_, jets, fn_output_signature=tf.TensorSpec((30,4), tf.float32))

def update_center(tok_cls, tok_patch, center_cls, center_patch, center_beta):
    center_cls.assign(center_beta * center_cls + (1 - center_beta) * tf.reduce_mean(tok_cls, axis=0))
    center_patch.assign(center_beta * center_patch + (1 - center_beta) * tf.reduce_mean(tok_patch, axis=[0, 1]))

def softmax_temp(x, t):
    return tf.nn.softmax(x/t, axis=-1)

def ce_teacher_student(t_logits, s_logits, temp_t=0.04, temp_s=0.1):
    p_t = tf.stop_gradient(softmax_temp(t_logits, temp_t))
    log_p_s = tf.nn.log_softmax(s_logits / temp_s, axis=-1)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(p_t * log_p_s, axis=-1))
    return cross_entropy

def jbot_loss(cls_s, cls_t, patch_s, patch_t, mask_bool, center_cls, center_patch, d_proj):
    # CLS
    loss_cls = ce_teacher_student(cls_t - center_cls, cls_s)

    # masked tokens
    float_mask = tf.cast(mask_bool, tf.float32)[..., None]
    patch_s = patch_s * float_mask
    patch_t = (patch_t - center_patch) * float_mask

    # flatten everything
    patch_s_flat = tf.reshape(patch_s, [-1, d_proj])
    patch_t_flat = tf.reshape(patch_t, [-1, d_proj])
    bool_flat  = tf.reshape(mask_bool, [-1])

    # keep only masked positions
    patch_s_sel = tf.boolean_mask(patch_s_flat, bool_flat)
    patch_t_sel = tf.boolean_mask(patch_t_flat, bool_flat)

    loss_tok = ce_teacher_student(patch_t_sel, patch_s_sel)
    return loss_cls + loss_tok, loss_cls, loss_tok

class EtaPhiHint(keras.layers.Layer):
    def __init__(self, d_model, hidden=0, **kwargs):
        super().__init__(**kwargs)
        if hidden and hidden > 0:
            self.net = keras.Sequential([
                Dense(hidden, activation="gelu"),
                Dense(d_model),
            ])
        else:
            self.net = Dense(d_model)

    def call(self, eta_phi): # eta_phi: (B,30,2)
        return self.net(eta_phi) # (B,30,d_model)

@tf.function
def koleo_loss_cls(cls, eps=1e-8):
    cls = tf.cast(cls, tf.float32)
    z = tf.math.l2_normalize(cls, axis=-1)

    sim = tf.matmul(z, z, transpose_b=True)
    n = tf.shape(sim)[0]

    # mask diagonal so a vector cant pick itself as nearest neighbor
    neg_big = tf.constant(-1e9, dtype=sim.dtype)
    sim = tf.where(tf.eye(n, dtype=tf.bool), neg_big, sim)

    max_sim = tf.reduce_max(sim, axis=1)
    max_sim = tf.clip_by_value(max_sim, -1.0, 1.0)

    # nn_dist^2 = 2 - 2*cos
    nn_dist = tf.sqrt(tf.maximum(2.0 - 2.0 * max_sim, eps))

    return -tf.reduce_mean(tf.math.log(nn_dist + eps))
    
def train_jbot(x_train, epochs, batch_size, optimizer, base_lr, warmup_epochs, ema_tau,
               student, teacher, proj_head_s, proj_head_t, n_layers, d_proj,
               mask_prob, mask_ratio_range, masker, center_cls, center_patch,
               center_beta, temp_t, temp_s,
               use_hint, hint_hidden,
               lambda_koleo,
               save_student_snapshots,
               snapshot_dir):

    token_emb_layer = student.get_layer("token_embedding")
    prepend_cls_layer = student.get_layer("prepend_cls")
    blocks = [student.get_layer(f"block_{i}") for i in range(n_layers)]

    hint_enc = None
    if use_hint:
        d_model_local = int(token_emb_layer.units) # matches token embedding dim
        hint_enc = EtaPhiHint(d_model_local, hidden=hint_hidden, name="eta_phi_hint")

    def ema_update(ema_tau=ema_tau):
        for s, t in zip(student.weights, teacher.weights):
            t.assign(ema_tau * t + (1 - ema_tau) * s)
        for s, t in zip(proj_head_s.weights, proj_head_t.weights):
            t.assign(ema_tau * t + (1 - ema_tau) * s)

    @tf.function
    def train_step(ds_jets, optimizer, student, teacher, proj_head_s, proj_head_t, masker):
        # two augmented views
        jet_u = tf_augment_batch(ds_jets)
        jet_v = tf_augment_batch(ds_jets)

        # tokens to mask
        mask_u = tf_make_masks_pt_ratio(jet_u, p_mask=mask_prob, ratio_range=mask_ratio_range)
        mask_v = tf_make_masks_pt_ratio(jet_v, p_mask=mask_prob, ratio_range=mask_ratio_range)

        with tf.GradientTape() as tape:
            # student: replace masked positions with the learnable mask token
            emb_u = token_emb_layer(jet_u)
            emb_v = token_emb_layer(jet_v)
            emb_u_masked = masker(emb_u, mask_u)
            emb_v_masked = masker(emb_v, mask_v)

            # add eta/phi hints only on masked tokens (no backbone change)
            if use_hint:
                hint_u = hint_enc(jet_u[..., :2]) # (B,30,d_model)
                hint_v = hint_enc(jet_v[..., :2])
                mu = tf.cast(mask_u, emb_u_masked.dtype)[..., None]
                mv = tf.cast(mask_v, emb_v_masked.dtype)[..., None]
                emb_u_masked = emb_u_masked + hint_u * mu
                emb_v_masked = emb_v_masked + hint_v * mv

            xu = prepend_cls_layer(emb_u_masked)
            xv = prepend_cls_layer(emb_v_masked)
            for blk in blocks:
                xu = blk(xu)
                xv = blk(xv)

            # raw backbone outputs (pre-proj) for koleo
            cls_u_s_raw, patch_u_s = xu[:, 0], xu[:, 1:]
            cls_v_s_raw, patch_v_s = xv[:, 0], xv[:, 1:]

            # projections
            cls_u_s = proj_head_s(tf.expand_dims(cls_u_s_raw, 1))[:, 0, :]
            cls_v_s = proj_head_s(tf.expand_dims(cls_v_s_raw, 1))[:, 0, :]
            patch_u_s = proj_head_s(patch_u_s)
            patch_v_s = proj_head_s(patch_v_s)

            # teacher
            cls_u_t, patch_u_t = teacher(jet_u, training=False)
            cls_v_t, patch_v_t = teacher(jet_v, training=False)
            cls_u_t = proj_head_t(tf.expand_dims(cls_u_t, 1))[:, 0, :]
            cls_v_t = proj_head_t(tf.expand_dims(cls_v_t, 1))[:, 0, :]
            patch_u_t = proj_head_t(patch_u_t)
            patch_v_t = proj_head_t(patch_v_t)

            # loss
            loss_uv, loss_cls_uv, loss_patch_uv = jbot_loss(
                cls_s=cls_u_s, cls_t=cls_v_t,
                patch_s=patch_u_s, patch_t=patch_u_t,
                mask_bool=mask_u, center_cls=center_cls, center_patch=center_patch, d_proj=d_proj
            )
            
            loss_vu, loss_cls_vu, loss_patch_vu = jbot_loss(
                cls_s=cls_v_s, cls_t=cls_u_t,
                patch_s=patch_v_s, patch_t=patch_v_t,
                mask_bool=mask_v, center_cls=center_cls, center_patch=center_patch, d_proj=d_proj
            )
            loss = 0.5 * (loss_uv + loss_vu)
            loss_cls = 0.5 * (loss_cls_uv + loss_cls_vu)
            loss_patch = 0.5 * (loss_patch_uv + loss_patch_vu)

            loss_koleo = 0
            if lambda_koleo > 0:
                loss_koleo = koleo_loss_cls(cls_u_s_raw) # only on one view (if on both views then you would push positives apart)
                loss = loss + tf.cast(lambda_koleo, loss.dtype) * loss_koleo

        vars_ = student.trainable_weights + proj_head_s.trainable_weights + masker.trainable_weights
        if use_hint:
            vars_ = vars_ + hint_enc.trainable_weights
        grads = tape.gradient(loss, vars_)
        grads_vars = [(g, v) for g, v in zip(grads, vars_) if g is not None]
        if grads_vars:
            optimizer.apply_gradients(grads_vars)

        # entropy
        z_t = tf.concat([cls_u_t, cls_v_t], axis=0)
        prob_t = tf.nn.softmax((z_t - center_cls)[...] / temp_t, axis=-1)
        entropy_teacher = -tf.reduce_mean(tf.reduce_sum(prob_t * tf.math.log(prob_t + 1e-9), axis=-1))
        prob_s = tf.nn.softmax((cls_u_s + cls_v_s) / 2 / temp_s, axis=-1)
        entropy_student = -tf.reduce_mean(tf.reduce_sum(prob_s * tf.math.log(prob_s + 1e-9), axis=-1))

        # centers
        update_center(
            tok_cls=tf.stop_gradient(tf.concat([cls_u_t, cls_v_t], 0)),
            tok_patch=tf.stop_gradient(tf.concat([patch_u_t, patch_v_t], 0)),
            center_cls=center_cls, center_patch=center_patch, center_beta=center_beta
        )
        return loss, loss_cls, loss_patch, loss_koleo, entropy_teacher, entropy_student

    history = {
        "loss_total": [], "loss_cls": [], "loss_patch": [], "loss_koleo": [],
        "entropy_teacher": [], "entropy_student": [],
        "center_cls": [], "center_patch": [],
        "mask_token": []
    }

    if save_student_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)
        student.save_weights(f"{snapshot_dir}/student_epoch000.weights.h5")

    ds = (tf.data.Dataset.from_tensor_slices(x_train.astype("float32"))
          .shuffle(x_train.shape[0], seed=42)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

    steps_per_epoch = math.ceil(x_train.shape[0] / batch_size)
    
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            current_lr = base_lr * float(epoch + 1) / float(warmup_epochs)
        else:
            current_lr = base_lr
        optimizer.learning_rate.assign(current_lr)
        print(f"epoch {epoch+1}/{epochs} | lr = {current_lr:.6f}")
        prog = Progbar(target=steps_per_epoch, interval=0.1, unit_name="step")
        
        ema_tau_waved = 1 - (1 - ema_tau) * (1 + math.cos(math.pi * epoch / epochs)) / 2
        L_total, L_cls, L_patch, L_koleo, E_teacher, E_student = [], [], [], [], [], []
        
        step = 0
        for ds_jets in ds:
            l_total, l_cls, l_patch, l_koleo, e_teacher, e_student = train_step(
                ds_jets=ds_jets, optimizer=optimizer,
                student=student, teacher=teacher,
                proj_head_s=proj_head_s, proj_head_t=proj_head_t,
                masker=masker
            )
            ema_update(ema_tau_waved)

            ll_total = float(l_total.numpy())
            ll_cls = float(l_cls.numpy())
            ll_patch = float(l_patch.numpy())
            ll_koleo = float(l_koleo.numpy())
            
            L_total.append(l_total.numpy())
            L_cls.append(l_cls.numpy())
            L_patch.append(l_patch.numpy())
            L_koleo.append(l_koleo.numpy())
            E_teacher.append(e_teacher.numpy())
            E_student.append(e_student.numpy())

            step += 1
            prog.update(step, values=[("loss", ll_total), ("cls", ll_cls), ("patch", ll_patch), ("koleo", ll_koleo)])

        history["loss_total"].append(np.mean(L_total))
        history["loss_cls"].append(np.mean(L_cls))
        history["loss_patch"].append(np.mean(L_patch))
        history["loss_koleo"].append(np.mean(L_koleo))
        history["entropy_teacher"].append(np.mean(E_teacher))
        history["entropy_student"].append(np.mean(E_student))
        history["center_cls"].append(np.linalg.norm(center_cls.numpy()))
        history["center_patch"].append(np.linalg.norm(center_patch.numpy()))
        history["mask_token"].append(np.linalg.norm(masker.mask_token.numpy()))

        print(f"epoch {epoch+1}/{epochs} | "
              f"loss(total)={history['loss_total'][-1]:.3f}; "
              f"loss(cls)={history['loss_cls'][-1]:.3f}; "
              f"loss(patch)={history['loss_patch'][-1]:.3f}; "
              f"loss(koleo)={history['loss_koleo'][-1]:.3f}; "
              f"entropy(teacher)={history['entropy_teacher'][-1]:.3f}; "
              f"entropy(student)={history['entropy_student'][-1]:.3f}; "
              f"norm(center_cls)={history['center_cls'][-1]:.3f}; "
              f"norm(center_patch)={history['center_patch'][-1]:.3f}; "
              f"norm(mask_token)={history['mask_token'][-1]:.3f}")

        if save_student_snapshots:
            student.save_weights(f"{snapshot_dir}/student_epoch{epoch+1:03d}.weights.h5")
    return history

def train_standalone(
    x_train,
    y_train,
    x_val,
    y_val,
    d_model,
    n_heads,
    n_layers,
    n_classes,
    lr,
    epochs,
    batch_size,
    tolerance,
    patience
):
    backbone_standalone = build_backbone(d_model, n_heads, n_layers, name="backbone_standalone")
    mlp_standalone = build_mlp(d_model, n_classes=n_classes, name="mlp_standalone")
    x_in = tf.keras.Input((30,4))
    cls, _ = backbone_standalone(x_in)
    x_out = mlp_standalone(cls)
    model_standalone = tf.keras.models.Model(x_in, x_out, name="model_standalone")
    model_standalone.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    history_standalone = model_standalone.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=0,
        callbacks=[EarlyStoppingLogger("supervised",epochs,patience,tolerance)]
    )
    return backbone_standalone, mlp_standalone, model_standalone, history_standalone
    
def finetune(
    x_train,
    y_train,
    x_val,
    y_val,
    d_model,
    n_heads,
    n_layers,
    n_classes,
    backbone_pretrain,
    base_lr,
    decay,
    epochs,
    batch_size,
    tolerance,
    patience
):
    backbone_ft = build_backbone(d_model, n_heads, n_layers, name="backbone_ft")
    backbone_ft.set_weights(backbone_pretrain.get_weights())
    mlp_ft = build_mlp(d_model, n_classes=n_classes, name="mlp_ft")
    x_in = tf.keras.Input((30, 4))
    cls, _ = backbone_ft(x_in)
    x_out = mlp_ft(cls)
    model = tf.keras.models.Model(x_in, x_out, name="model_finetune")

    optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # LLRD multipliers
    layer_groups = [["token_embedding", "prepend_cls"]] + [[f"block_{i}"] for i in range(n_layers)]
    n_groups = len(layer_groups)

    lr_multipliers_by_name = {}

    # backbone groups decaying
    for depth, group_names in enumerate(layer_groups):
        depth_from_last = n_groups - 1 - depth
        group_lr = base_lr * (decay ** depth_from_last)
        mult = group_lr / base_lr
        for layer_name in group_names:
            layer = backbone_ft.get_layer(layer_name)
            for v in layer.trainable_weights:
                lr_multipliers_by_name[v.name] = mult

    # all other trainable vars get multiplier 1.0
    var_list = model.trainable_weights
    for v in var_list:
        if v.name not in lr_multipliers_by_name:
            lr_multipliers_by_name[v.name] = 1.0

    lr_multipliers = [float(lr_multipliers_by_name[v.name]) for v in var_list]

    # convert data to tensors to avoid per-step numpy copies
    x_train_tf = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_val_tf = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val_tf = tf.convert_to_tensor(y_val, dtype=tf.float32)

    n_train = x_train.shape[0]
    n_val = x_val.shape[0]
    steps_per_epoch = math.ceil(n_train / batch_size)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # early stopping state
    best_val_acc = -np.inf
    best_weights = None
    epochs_no_improve = 0
    last_msg_len = 0

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb, logits)

        grads = tape.gradient(loss, var_list)
        scaled_grads_vars = []
        for g, v, m in zip(grads, var_list, lr_multipliers):
            if g is not None:
                scaled_grads_vars.append((g * m, v))
        optimizer.apply_gradients(scaled_grads_vars)

        preds = tf.argmax(logits, axis=-1)
        y_true = tf.argmax(yb, axis=-1)
        batch_correct = tf.reduce_sum(tf.cast(preds == y_true, tf.int32))
        batch_total = tf.shape(xb)[0]

        return loss, batch_correct, batch_total

    @tf.function
    def val_step(xb, yb):
        logits = model(xb, training=False)
        loss = loss_fn(yb, logits)
        preds = tf.argmax(logits, axis=-1)
        y_true = tf.argmax(yb, axis=-1)
        batch_correct = tf.reduce_sum(tf.cast(preds == y_true, tf.int32))
        batch_total = tf.shape(xb)[0]
        return loss, batch_correct, batch_total

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        train_losses = []
        train_correct = 0
        train_total = 0

        # training loop
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min((step + 1) * batch_size, n_train)
            idx_np = perm[start:end]
            idx = tf.convert_to_tensor(idx_np, dtype=tf.int32)

            xb = tf.gather(x_train_tf, idx)
            yb = tf.gather(y_train_tf, idx)

            loss, batch_correct, batch_total = train_step(xb, yb)
            loss_val = float(loss.numpy())
            train_losses.append(loss_val)
            train_correct += int(batch_correct.numpy())
            train_total += int(batch_total.numpy())

        train_loss = float(np.mean(train_losses))
        train_acc = train_correct / train_total

        # validation
        val_losses = []
        val_correct = 0
        val_total = 0
        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            xb = x_val_tf[start:end]
            yb = y_val_tf[start:end]
            v_loss, batch_correct, batch_total = val_step(xb, yb)
            v_loss_val = float(v_loss.numpy())
            val_losses.append(v_loss_val)
            val_correct += int(batch_correct.numpy())
            val_total += int(batch_total.numpy())

        val_loss = float(np.mean(val_losses))
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # early stopping bookkeeping
        if val_acc > best_val_acc + tolerance:
            best_val_acc = val_acc
            best_weights = model.get_weights()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        msg = (f"[jBOT] Epoch {epoch+1}/{epochs} "
               f"| acc: {train_acc:.4f} | loss: {train_loss:.4f} "
               f"| val_acc: {val_acc:.4f} | val_loss: {val_loss:.4f} "
               f"| best_val_acc: {best_val_acc:.4f} "
               f"| no_improve: {epochs_no_improve}/{patience}")

        # clean overwrite of previous line
        erase = " " * max(0, last_msg_len - len(msg))
        sys.stdout.write("\r" + msg + erase)
        sys.stdout.flush()
        last_msg_len = len(msg)

        if epochs_no_improve >= patience:
            sys.stdout.write(
                f"\n[finetuning] Early stopping at epoch {epoch+1} "
                f"(best val_acc = {best_val_acc:.4f})\n"
            )
            break

        if epoch+1 == epochs:
            sys.stdout.write("\n")

    # restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

    return backbone_ft, mlp_ft, model, history

class EarlyStoppingLogger(keras.callbacks.Callback):
    def __init__(self, name, epochs, patience, min_delta):
        super().__init__()
        self.name = name
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = -np.inf
        self.no_improve = 0
        self.best_weights = None
        self._last_msg_len = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = float(logs.get("accuracy", 0.0))
        loss = float(logs.get("loss", 0.0))
        val_acc = float(logs.get("val_accuracy", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))

        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.no_improve = 0
            self.best_weights = self.model.get_weights()
        else:
            self.no_improve += 1

        msg = (f"[{self.name}] Epoch {epoch+1}/{self.epochs} "
               f"| acc: {acc:.4f} | loss: {loss:.4f} "
               f"| val_acc: {val_acc:.4f} | val_loss: {val_loss:.4f} "
               f"| best_val_acc: {self.best_val_acc:.4f} "
               f"| no_improve: {self.no_improve}/{self.patience}")

        # clean overwrite
        erase = " " * max(0, self._last_msg_len - len(msg))
        sys.stdout.write("\r" + msg + erase)
        sys.stdout.flush()
        self._last_msg_len = len(msg)

        if self.no_improve >= self.patience:
            sys.stdout.write(
                f"\n[{self.name}] Early stopping at epoch {epoch+1} "
                f"(best val_acc = {self.best_val_acc:.4f})\n"
            ) 
            self.model.stop_training = True

        elif self.epochs is not None and (epoch+1 == self.epochs):
            sys.stdout.write("\n")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

def scan_finetune_lr_decay(
    base_lrs, decays, train_frac,
    x_train, y_train, x_val, y_val, x_test, y_test,
    d_model, n_heads, n_layers, n_classes,
    backbone_pretrain,
    epochs, batch_size, tolerance, patience
):
    n = int(len(x_train) * train_frac)
    x_train_f, y_train_f = x_train[:n], y_train[:n]
    
    init_w = backbone_pretrain.get_weights()

    for base_lr in base_lrs:
        for decay in decays:
            
            backbone_pretrain.set_weights(init_w)
            lr = base_lr * (batch_size / 256)
            
            _, _, model_ft, _ = finetune(
                x_train=x_train_f,
                y_train=y_train_f,
                x_val=x_val,
                y_val=y_val,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                n_classes=n_classes,
                backbone_pretrain=backbone_pretrain,
                base_lr=lr,
                decay=decay,
                epochs=epochs,
                batch_size=batch_size,
                tolerance=tolerance,
                patience=patience
            )
            p = model_ft.predict(x_test, verbose=0)
            acc = (p.argmax(1) == y_test.argmax(1)).mean()
            print(f"base_lr={base_lr:.0e}  decay={decay:.2f}  test_acc={acc:.4f}\n")
            
    backbone_pretrain.set_weights(init_w)

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
    
#-------------------------------------- plotting / evaluation --------------------------------------

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
    plt.figure(figsize=(7,6))
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
                     label=f"{c} ({np.mean(aucs):.4f} $\pm$ {np.std(aucs):.4f})")

            tpr_lo = np.clip(tpr_mu - tpr_sd, 0.0, 1.0)
            tpr_hi = np.clip(tpr_mu + tpr_sd, 0.0, 1.0)
            plt.fill_betweenx(fpr_grid, tpr_lo, tpr_hi, color=class_colors[k], alpha=0.2)

    plt.xlabel("TPR", size=16)
    plt.ylabel("FPR", size=16)
    plt.yscale("log")
    plt.ylim(1e-4, 1)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=12, loc='lower right')
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

    plt.figure(figsize=(7,6))
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

        plt.plot(mu, fpr_grid, lw=1.5, label=f"{name} ({np.mean(aucs):.4f} $\pm$ {np.std(aucs):.4f})")
        plt.fill_betweenx(
            fpr_grid,
            np.clip(mu - sd, 0.0, 1.0),
            np.clip(mu + sd, 0.0, 1.0),
            alpha=0.2
        )

    plt.xlabel("TPR", size=16)
    plt.ylabel("FPR", size=16)
    plt.yscale("log")
    plt.ylim(1e-4, 1)
    plt.xlim(0, 1)
    #plt.title("Pretrained [CLS] embedding", size=15)
    plt.legend(fontsize=12, loc="lower right")
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

    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    if xlim is not None:
        plt.xlim(xlim[0],xlim[1])
    plt.legend(loc="upper right")
    plt.title(title)
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

    plt.figure(figsize=(7, 6))

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
                label=f"{name} ({np.mean(aucs):.4f}±{np.std(aucs):.4f})",
                lw=1.5, alpha=0.7
            )
            c = line.get_color()
            low = np.clip(mu - sd, 1e-6, None)
            high = np.clip(mu + sd, 1e-6, None)
            plt.fill_between(tpr_grid, low, high, color=c, alpha=0.2, linewidth=0)

    plt.yscale("log")
    plt.ylim(1e-3, 1)
    plt.xlim(0, 1)
    plt.xlabel("TPR", fontsize=16)
    plt.ylabel("FPR", fontsize=16)
    plt.title(f"{title_sig} vs QCD", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}")
    plt.show()

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

#-------------------------------------- AD score metrics --------------------------------------

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
            


