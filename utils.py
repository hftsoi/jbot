from imports import *
from utils_data import *
from utils_model import *
from utils_eval import *
from utils_ad import *

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


