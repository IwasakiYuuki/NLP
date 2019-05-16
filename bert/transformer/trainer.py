import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
print(sys.path)
import tensorflow as tf
import numpy as np
from bert.transformer.transformer import Transformer
from bert.transformer.preprocess.batch_generator import BatchGenerator
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('data/natsume_model.model')

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

data_path = './data/natsume.txt'

batch_generator = BatchGenerator()
batch_generator.load(data_path)

vocab_size = batch_generator.vocab_size

graph = tf.Graph()
with graph.as_default():
    transformer = Transformer(
        vocab_size=vocab_size,
    )
    transformer.build_graph()

log_dir = './log/'
ckpt_path = './checkpoint/model.ckpt'

with graph.as_default():
    generated_text = tf.placeholder(tf.string, (None, None), name='generated_text')
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta2=0.98,
    )
    optimize_op = optimizer.minimize(transformer.loss, global_step=global_step)

    summary_op = tf.summary.merge([
        tf.summary.scalar('train/loss', transformer.loss),
        tf.summary.scalar('train/acc', transformer.acc),
        tf.summary.scalar('train/learning_rate', learning_rate),
        tf.summary.text('train/generated_text', generated_text),
    ], name='train_summary')
    summary_writer = tf.summary.FileWriter(log_dir, graph)
    saver = tf.train.Saver()

max_step = 100000
batch_size = 64
max_learning_rate = 0.0001
warmup_step = 4000


def get_learning_rate(step: int) -> float:
    rate = min(step ** -0.5, step * warmup_step ** -1.5) / warmup_step ** -0.5
    return max_learning_rate * rate


with graph.as_default():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0

#with graph.as_default():
#    for batch in batch_generator.get_batch(batch_size=batch_size):
#        feed = {
#            **batch,
#            learning_rate: get_learning_rate(step + 1),
#        }
#        _, loss, acc, prediction, step, summary = sess.run(
#            [optimize_op, transformer.loss, transformer.acc, transformer.prediction, global_step, summary_op], feed_dict=feed)
#        summary_writer.add_summary(summary, step)
#
#        if step % 100 == 0:
#            print(f'{step}: loss: {loss},\t acc: {acc}')
#            saver.save(sess, ckpt_path, global_step=step)
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))

    print(transformer.predict(np.array([[6, 49, 9, 172, 853, 500]])))
