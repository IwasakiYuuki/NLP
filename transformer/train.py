import pickle
import numpy as np
import keras
import keras_transformer
import keras_bert
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('./data/wiki-ja.model')

config_file_path = './data/config.json'
checkpoint_file_path = './data/checkpoint/model.ckpt-1400000'

with open('./data/generator_data.pickle', 'rb') as f:
    generator_data = pickle.load(f)


def train(
        use_checkpoint=True,
        initial_epoch=0,
):
    if use_checkpoint:
        transformer_model = keras_transformer.get_model(
            token_num=32000,
            embed_dim=768,
            encoder_num=4,
            decoder_num=4,
            head_num=8,
            hidden_dim=512,
            attention_activation='relu',
            feed_forward_activation='relu',
            dropout_rate=0.1,
        )
        transformer_model.load_weights('./data/checkpoint/transformer_onbert_model-Adam4000-Dall.ckpt')
    else:
        bert_model = keras_bert.load_trained_model_from_checkpoint(
            checkpoint_file=checkpoint_file_path,
            config_file=config_file_path
        )
        bert_weights = bert_model.get_layer(name='Embedding-Token').get_weights()[0]
        transformer_model = keras_transformer.get_model(
            token_num=32000,
            embed_dim=768,
            encoder_num=4,
            decoder_num=4,
            head_num=8,
            hidden_dim=512,
            attention_activation='relu',
            feed_forward_activation='relu',
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
    tb = keras.callbacks.TensorBoard(log_dir='./data/log-adam-4000-Dall-onbert/')
    try:
        history = transformer_model.fit_generator(
            generator=_generator(),
            steps_per_epoch=100,
            epochs=1000,
            validation_data=_generator(),
            validation_steps=20,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    './data/checkpoint/transformer_onbert_model-Adam4000-Dall.ckpt',
                    monitor='val_loss'),
                tb,
                keras.callbacks.LearningRateScheduler(_decay),
                PredictionCallback(generator_data[:2, 0], 30),
            ],
            initial_epoch=initial_epoch,
        )
    except KeyboardInterrupt:
        tb.writer.close()


def prediction(
        model,
        inputs,
        max_len,
):
    predicted = keras_transformer.decode(
        model,
        inputs.tolist(),
        start_token=2,
        end_token=3,
        pad_token=0,
        max_len=max_len,
        top_k=10,
        temperature=1.0,
    )
    print('=================predict result=================')
    print('input:', sp.decode_ids(inputs.tolist()[0]))
    print('\n------------------------------------------------')
    print('output:', sp.decode_ids(list(map(int, predicted[0]))))
    print('\n================================================')
    print('input:', sp.decode_ids(inputs.tolist()[1]))
    print('\n------------------------------------------------')
    print('output:', sp.decode_ids(list(map(int, predicted[1]))))
    print('\n================================================')


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
        step_num = epochs * 100
    warmup_steps = 4000
    d = 512
    return 0.0001 * (min(step_num ** -0.5, step_num * (warmup_steps ** -1.5)) / warmup_steps ** -0.5)


def _generator():
    i = 0
    data_len = len(generator_data)
    batch_size = 128
    while True:
        yield [generator_data[i * batch_size:(i + 1) * batch_size, 0],
               generator_data[i * batch_size:(i + 1) * batch_size, 1]], \
              generator_data[i * batch_size:(i + 1) * batch_size, 2, :, None]
        if ((i + 2) * batch_size) >= data_len:
            i = 0
        else:
            i += 1


if __name__ == '__main__':
    main()