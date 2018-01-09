# encoding=utf8  
import os
import numpy as np
import pprint
import pickle
import tensorflow as tf

from data import read_data, read_our_data
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 6, "number of hops [6]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "ptb", "data set name [ptb]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")
flags.DEFINE_boolean("inference", False, "inference ")

flags.DEFINE_integer("mem_size", 800, "memory size (the word number of context to use) [800]")

FLAGS = flags.FLAGS

def main(_):
    count = []
    with open('./processed/word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    
    train_data = read_our_data('./data/CBData/cbtest_CN_train.txt', count, word2idx)
    valid_data = read_our_data('./data/CBData/cbtest_CN_valid_2000ex.txt', count, word2idx)
    test_data = read_our_data('./data/CBData/cbtest_CN_test_2500ex.txt', count, word2idx)
    # Some statistics
    lens = [np.sum([len(sentence) for sentence in context]) for context in train_data['contexts']]
    print('The vocabulary size is now: %d' % len(word2idx))
    print('The distribution of word number of contexts(training data):')
    print(np.histogram(lens))

#     for key in train_data.keys():
#         print(key)
#         print(data[key][0])
#         print()
    
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.nwords = len(word2idx)
    pp.pprint(flags.FLAGS.__flags)

#     train_data = read_data('%s/%s.train.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
#     valid_data = read_data('%s/%s.valid.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
#     test_data = read_data('%s/%s.test.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
#     exit()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()

        if FLAGS.is_test:
            model.run(valid_data, test_data, word2idx)
        else:
            model.run(train_data, valid_data, word2idx)

if __name__ == '__main__':
    tf.app.run()
