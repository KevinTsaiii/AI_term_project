# encoding=utf8  
import os
import numpy as np
import pprint
import pickle
import tensorflow as tf

from glob import glob
from data import read_data, read_our_data, read_test_data
from model import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [3]")
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
flags.DEFINE_boolean("inference", False, "to inference the data/Test_Set/test_set.txt to csv")
flags.DEFINE_boolean("restore", True, "resotre or not")

flags.DEFINE_integer("mem_size", 800, "memory size (the word number of context to use) [800]")

FLAGS = flags.FLAGS

def main(_):
    count = []
    with open('./processed/word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.nwords = len(word2idx)
    pp.pprint(flags.FLAGS.__flags)

#     train_data = read_data('%s/%s.train.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
#     valid_data = read_data('%s/%s.valid.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
#     test_data = read_data('%s/%s.test.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
#     exit()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3 if FLAGS.inference else 0.6
    with tf.Session(config=config) as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()
        
        if FLAGS.inference:
            test_set_data = read_test_data('./data/Test_Set/test_set.txt', word2idx)
            model.load()
            answer = model.inference(test_set_data, word2idx)
            import pandas as pd
            answer = pd.DataFrame(answer, columns=['answer'])
            answer.index += 1
            answer.to_csv('./guess/guess.csv', index_label='id')
        else:
            if FLAGS.restore:
                model.load()
            with open('./processed/all_train.pkl', 'rb') as f:
                train_data = pickle.load(f)
            with open('./processed/all_valid.pkl', 'rb') as f:
                valid_data = pickle.load(f)
            test_data = read_our_data('./data/CBData/cbtest_CN_test_2500ex.txt', count, word2idx)
            
            if FLAGS.is_test:
                model.run(valid_data, test_data, word2idx)
            else:
                model.run(train_data, valid_data, word2idx)
            # Some statistics
#             lens = [
#                 np.sum([len(sentence) for sentence in context])
#                 for context in train_data['contexts']
#             ]
#             print('The vocabulary size is now: %d' % len(word2idx))
#             print('The distribution of word number of contexts(training data):')
#             print(np.histogram(lens))

if __name__ == '__main__':
    tf.app.run()
