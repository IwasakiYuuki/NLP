import math

import keras
import keras_transformer
import keras_bert
from keras_bert.backend import backend as K
from keras_embed_sim import EmbeddingRet, EmbeddingSim

config_file_path = ''
checkpoint_file_path = ''


def get_model(
        token_num,
        embed_dim,
        encoder_num,
        decoder_num,
        head_num,
        hidden_dim,
        attention_activation=None,
        feed_forward_activation='relu',
        dropout_rate=0.0,
        use_same_embed=True,
        embed_weights=None,
        embed_trainable=None,
        trainable=True
):
    encoder_model = keras_bert.load_trained_model_from_checkpoint(
        config_file=config_file_path,
        checkpoint_file=checkpoint_file_path,
    )

    decoder_model = keras_bert.load_trained_model_from_checkpoint(
        config_file=config_file_path,
        checkpoint_file=checkpoint_file_path,
    )

    encoder_inputs, encoder_embed = encoder_model.inputs, encoder_model.output
    decoder_inputs, decoder_embed = decoder_model.inputs, decoder_model.output
    encoded_layer = keras_transformer.get_encoders(
        encoder_num=encoder_num,
        input_layer=encoder_embed,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    decoded_layer = keras_transformer.get_decoders(
        decoder_num=decoder_num,
        input_layer=decoder_embed,
        encoded_layer=encoded_layer,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    dense_layer = keras.layers.Dense(
        token_num,
        name='Output',
    )(decoded_layer)

    return keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_layer)