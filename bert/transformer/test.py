import math

import keras
import keras_transformer
import keras_bert
from keras_bert.backend import backend as K

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

def get_inputs(seq_len):
    """Get input layers.
    See: https://arxiv.org/pdf/1810.04805.pdf
    :param seq_len: Length of the sequence or None.
    """
    names = ['Token', 'Segment', 'Masked']
    return [keras.layers.Input(
        shape=(seq_len,),
        name='Input-%s' % name,
    ) for name in names]

def gelu(x):
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return 0.5 * x * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))

def get_model(
        token_num,
        pos_num=512,
        seq_len=512,
        embed_dim=768,
        transformer_num=12,
        head_num=12,
        feed_forward_dim=3072,
        dropout_rate=0.1,
        weight_decay=0.01,
        attention_activation=None,
        feed_forward_activation=gelu,
        custom_layers=None,
        output_layer_num=1,
        decay_steps=100000,
        warmup_steps=10000,
        lr=1e-4
):
    inputs, bert_output_layer = keras_bert.get_model(
        token_num=token_num,
        pos_num=pos_num,
        seq_len=seq_len,
        embed_dim=embed_dim,
        transformer_num=transformer_num,
        head_num=head_num,
        feed_forward_dim=feed_forward_dim,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        custom_layers=custom_layers,
        training=False,
        trainable=False,
        output_layer_num=output_layer_num,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        lr=lr
    )
