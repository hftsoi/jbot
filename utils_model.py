from imports import *

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