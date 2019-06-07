import pickle
import numpy as np

import keras
import keras_transformer
import keras_bert


config_file_path = 'data/config/config.json'
checkpoint_file_path = 'data/checkpoint/bert_model.ckpt'

with open('data/nagoya_corpus/nagoya_encoder_inputs_L20.pickle', 'rb') as f:
    encoder_inputs = pickle.load(f)
with open('data/nagoya_corpus/nagoya_decoder_inputs_L20.pickle', 'rb') as f:
    decoder_inputs = pickle.load(f)
with open('data/nagoya_corpus/nagoya_decoder_outputs_L20.pickle', 'rb') as f:
    decoder_outputs = pickle.load(f)
with open('data/nagoya_corpus/token2text.pickle', 'rb') as f:
    token2text = pickle.load(f)


encoder_inputs = encoder_inputs[:32]
decoder_inputs = decoder_inputs[:32]
decoder_outputs = decoder_outputs[:32]


def get_transformer_on_bert_model(
        token_num: int,
        embed_dim: int,
        encoder_num: int,
        decoder_num: int,
        head_num: int,
        hidden_dim: int,
        embed_weights,
        attention_activation=None,
        feed_forward_activation: str = 'relu',
        dropout_rate: float = 0.0,
        use_same_embed: bool = True,
        embed_trainable=True,
        trainable: bool = True
) -> keras.engine.training.Model:
    """
    Transformerのモデルのinputsを特徴ベクトルにしたモデル．それ以外は特に変わらない．
    inputsのshapeは (None, seq_len, embed_dim) となっている，

    Parameters
    ----------
    token_num
        トークンのサイズ．（vocab_sizeと同じ）
    embed_dim
        特徴ベクトルの次元．inputsの次元数と同じにする．
    encoder_num
        エンコーダの層の数．
    decoder_num
        デコーダの層の数．
    head_num
        Multi-Head Attentionレイヤの分割ヘッド数．
    hidden_dim
        隠し層の次元数．
    embed_weights
        特徴ベクトルの初期化．
    attention_activation
        Attentionレイヤの活性化関数．
    feed_forward_activation
        FFNレイヤの活性化関数．
    dropout_rate
        Dropoutのレート．
    use_same_embed
        エンコーダとデコーダで同じweightsを使用するか．
    embed_trainable
        特徴ベクトルがトレーニング可能かどうか．
    trainable
        モデルがトレーニング可能かどうか．

    Returns
    -------
    model
        日本語学習済みのBERTの特徴ベクトルを用いたTransformerモデル
    """
    return keras_transformer.get_model(
        token_num=token_num,
        embed_dim=embed_dim,
        encoder_num=encoder_num,
        decoder_num=decoder_num,
        head_num=head_num,
        hidden_dim=hidden_dim,
        embed_weights=embed_weights,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        use_same_embed=use_same_embed,
        embed_trainable=embed_trainable,
        trainable=trainable
    )


def train(
        use_checkpoint=True,
        initial_epoch=0,
):
    if use_checkpoint:
        transformer_model = keras_transformer.get_model(
            token_num=32006,
            embed_dim=768,
            encoder_num=4,
            decoder_num=4,
            head_num=8,
            hidden_dim=256,
            dropout_rate=0.1,
        )
        transformer_model.load_weights('data/checkpoint/transformer_model.ckpt')
    else:
        bert_model = keras_bert.load_trained_model_from_checkpoint(
            checkpoint_file=checkpoint_file_path,
            config_file=config_file_path
        )
        bert_weights = bert_model.get_layer(name='Embedding-Token').get_weights()[0]
        transformer_model = get_transformer_on_bert_model(
            token_num=32006,
            embed_dim=768,
            encoder_num=4,
            decoder_num=4,
            head_num=8,
            hidden_dim=256,
            dropout_rate=0.1,
            embed_weights=bert_weights,
        )
    transformer_model.compile(
        optimizer=keras.optimizers.Adam(beta_2=0.98),
#        optimizer=keras.optimizers.SGD(),
#        optimizer='adam',
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[keras.metrics.mae, keras.metrics.sparse_categorical_accuracy],
    )
    transformer_model.summary()
    history = transformer_model.fit_generator(
        generator=_generator(),
        steps_per_epoch=100,
        epochs=200,
        validation_data=_generator(),
        validation_steps=20,
        callbacks=[
            keras.callbacks.ModelCheckpoint('./data/checkpoint/transformer_model.ckpt', monitor='val_loss'),
            keras.callbacks.TensorBoard(log_dir='./data/log-adam-4000-D32/'),
            keras.callbacks.LearningRateScheduler(_decay),
#            keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto'),
            PredictionCallback(encoder_inputs[0], 20),
        ],
        initial_epoch=initial_epoch,
    )


def prediction(
        model,
        inputs,
        max_len,
):
#    predicted = np.asarray([2]+[0]*(max_len-1))
#    for i in range(max_len-1):
#        predicted[i+1] = model.predict(x=[inputs, predicted]).argmax(axis=2).flatten()[i+1]
    predicted = decode(
        model,
        inputs.tolist(),
        start_token=2,
        end_token=3,
        pad_token=0,
        max_len=max_len,
    )
    print('=================predict result=================')
    print('input:')
    for i in inputs:
        print(token2text[i], end='')
    print('\n------------------------------------------------')
    print('output:')
    for i in predicted:
        print(token2text[i], end='')
    print('\n================================================')


def beam_search(
        model,
        inputs,
        max_len,
):
    result = np.asarray([2]+[0]*(max_len-1))
    for i in range(max_len-1):
        current_pre = model.predict(x=[inputs, result])
        current_num = current_pre[i+1][0].argsort()[::-1][:2]
        current_pro = current_pre[i+1][0][current_num]
        next_pro = []
        for c in current_num:
            cache = result.copy()
            cache[i+1] = c
            next_pre = model.predict(x=[inputs, cache])
            next_pro.append(next_pre[i+1][0][next_pre[i+1][0].argsort()[::-1][:2]])
        next_pro = next_pro * current_pro[:, None]
        result[i+1] = current_num[np.unravel_index(np.argmax(next_pro), next_pro.shape)[1]]

    return result


def decode(model, tokens, start_token, end_token, pad_token, max_len=10000, max_repeat=10, max_repeat_block=10):
    """Decode with the given model and input tokens.
    :param model: The trained model.
    :param tokens: The input tokens of encoder.
    :param start_token: The token that represents the start of a sentence.
    :param end_token: The token that represents the end of a sentence.
    :param pad_token: The token that represents padding.
    :param max_len: Maximum length of decoded list.
    :param max_repeat: Maximum number of repeating blocks.
    :param max_repeat_block: Maximum length of the repeating block.
    :return: Decoded tokens.
    """
    is_single = not isinstance(tokens[0], list)
    if is_single:
        tokens = [tokens]
    batch_size = len(tokens)
    decoder_inputs = [[start_token] for _ in range(batch_size)]
    outputs = [None for _ in range(batch_size)]
    output_len = 1
    while len(list(filter(lambda x: x is None, outputs))) > 0:
        output_len += 1
        batch_inputs, batch_outputs = [], []
        max_input_len = 0
        index_map = {}
        for i in range(batch_size):
            if outputs[i] is None:
                index_map[len(batch_inputs)] = i
                batch_inputs.append(tokens[i][:])
                batch_outputs.append(decoder_inputs[i])
                max_input_len = max(max_input_len, len(tokens[i]))
        for i in range(len(batch_inputs)):
            batch_inputs[i] += [pad_token] * (max_input_len - len(batch_inputs[i]))
        predicts = model.predict([np.asarray(batch_inputs), np.asarray(batch_outputs)])
        for i in range(len(predicts)):
            last_token = np.argsort(predicts[i][-1])[::-1][:2]
            if last_token[0] == 1:
                last_token = last_token[1]
            else:
                last_token = last_token[0]
            decoder_inputs[index_map[i]].append(last_token)
            if last_token == end_token or\
                    (max_len is not None and output_len >= max_len) or\
                    _get_max_suffix_repeat_times(decoder_inputs, max_repeat * max_repeat_block) >= max_repeat:
                outputs[index_map[i]] = decoder_inputs[index_map[i]]
    if is_single:
        outputs = outputs[0]
    return outputs


def main():
    train(
        use_checkpoint=False,
    )


class PredictionCallback(keras.callbacks.Callback):
    def __init__(self, input, max_len, **kwargs):
        super().__init__(**kwargs)
        self.input = input
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        prediction(
            self.model,
            self.input,
            self.max_len,
        )


def _decay(epochs):
    if epochs == 0:
        step_num = 1
    else:
        step_num = epochs*25
    warmup_steps = 4000
    d = 768
    return 0.0001 * (min(step_num**-0.5, step_num*(warmup_steps**-1.5)) / warmup_steps ** -0.5)


def _generator():
    i = 0
    data_len = len(encoder_inputs)
    batch_size = 32
    while True:
        if (i + batch_size) >= data_len:
            i = 0
        else:
            i += 1
        yield [encoder_inputs[i:i+batch_size], decoder_inputs[i:i+batch_size]], decoder_outputs[i:i+batch_size]


def _get_max_suffix_repeat_times(tokens, max_len):
    detect_len = min(max_len, len(tokens))
    next = [-1] * detect_len
    k = -1
    for i in range(1, detect_len):
        while k >= 0 and tokens[len(tokens) - i - 1] != tokens[len(tokens) - k - 2]:
            k = next[k]
        if tokens[len(tokens) - i - 1] == tokens[len(tokens) - k - 2]:
            k += 1
        next[i] = k
    max_repeat = 1
    for i in range(2, detect_len):
        if next[i] >= 0 and (i + 1) % (i - next[i]) == 0:
            max_repeat = max(max_repeat, (i + 1) // (i - next[i]))
    return max_repeat


if __name__ == '__main__':
    print(encoder_inputs[0:128].shape)
    main()
