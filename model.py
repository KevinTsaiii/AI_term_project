import os
import math
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from past.builtins import xrange
from data import read_test_data

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.nwords
        self.init_hid = config.init_hid
        self.init_std = config.init_std
        self.batch_size = config.batch_size
        self.nepoch = config.nepoch
        self.nhop = config.nhop
        self.edim = config.edim
        self.mem_size = config.mem_size
        self.lindim = config.lindim
        self.max_grad_norm = config.max_grad_norm

        self.show = config.show
        self.is_test = config.is_test
        self.checkpoint_dir = config.checkpoint_dir

        if not os.path.isdir(self.checkpoint_dir):
            raise Exception(" [!] Directory %s not found" % self.checkpoint_dir)

#         self.input = tf.placeholder(tf.float32, [None, self.edim], name="input")
        self.input = tf.placeholder(tf.float32, [None, self.nwords], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [None, self.nwords], name="target")
        self.context = tf.placeholder(tf.int32, [None, self.mem_size], name="context")

        self.hid = []
#         self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.current_lr = config.init_lr
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.A = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.B = tf.Variable(tf.random_normal([self.nwords, self.edim], stddev=self.init_std))
        self.C = tf.Variable(tf.random_normal([self.edim, self.edim], stddev=self.init_std))
        
        self.query_bow_emb = tf.Variable(
            tf.random_normal([self.nwords, self.edim], stddev=self.init_std)
        )

        # Temporal Encoding
        self.T_A = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))
        self.T_B = tf.Variable(tf.random_normal([self.mem_size, self.edim], stddev=self.init_std))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        Bin = tf.add(Bin_c, Bin_t)

        self.hid.append(tf.matmul(self.input, self.query_bow_emb))
        for h in xrange(self.nhop):
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.matmul(self.hid3dim, Ain, adjoint_b=True)
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], self.C)
            Dout = tf.add(Cout, Bout2dim)

            self.share_list[0].append(Cout)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [tf.shape(self.input)[0], self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [tf.shape(self.input)[0], self.edim-self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat(axis=1, values=[F, K]))

    def build_model(self):
        self.build_memory()

        self.W = tf.Variable(tf.random_normal([self.edim, self.nwords], stddev=self.init_std))
        self.z = tf.matmul(self.hid[-1], self.W)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.z, labels=self.target)

        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(max_to_keep=20)

    def train(self, data):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)

        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                m = random.randrange(self.mem_size, len(data))
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size :m]

            _, loss, self.step = self.sess.run(
                [self.optim, self.loss, self.global_step],
                feed_dict={
                    self.input: x,
                    self.time: time,
                    self.target: target,
                    self.context: context
                }
            )
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def our_train(self, data, word2idx):
        N = int(math.floor(len(data['answers']) / self.batch_size))
        cost = 0

        context = np.ndarray([self.batch_size, self.mem_size])
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        
        x = np.zeros([self.batch_size, self.nwords], dtype=np.float32) # bag-of-word to encode a query
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded

        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Train', max=N)
            
        random_perm = np.random.permutation(len(data['answers']))
        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            x.fill(0)
            # constructing training examples for this batch
            for b in xrange(self.batch_size):
                # find which training example to use
                i = random_perm[idx * self.batch_size + b]
                
                # one-hot of target
                target[b][data['answers'][i]] = 1
                
                # context(only pick last part if len(context) is too long)
                raw_context = data['contexts'][i]
                raw_context = [word for sent in raw_context for word in sent]
                n_pick = min(self.mem_size, len(raw_context))
                context.fill(word2idx[''])
                context[b][:n_pick] = raw_context[-n_pick:]

                # x (bag-of-word of query)
                for word_id in data['querys'][i]:
                    x[b][word_id] += 1
                
            _, loss, self.step = self.sess.run(
                [self.optim, self.loss, self.global_step],
                feed_dict={
                    self.input: x,
                    self.time: time,
                    self.target: target,
                    self.context: context
                }
            )
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def our_test(self, data, word2idx, label='Test'):
        N = int(math.floor(len(data['answers']) / self.batch_size))
        cost = 0

        context = np.ndarray([self.batch_size, self.mem_size])
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        
        x = np.zeros([self.batch_size, self.nwords], dtype=np.float32) # bag-of-word to encode a query
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded

        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)
            
        random_perm = np.random.permutation(len(data['answers']))
        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            x.fill(0)
            # constructing training examples for this batch
            for b in xrange(self.batch_size):
                # find which training example to use
                i = random_perm[idx * self.batch_size + b]
                
                # one-hot of target
                target[b][data['answers'][i]] = 1
                
                # context(only pick last part if len(context) is too long)
                raw_context = data['contexts'][i]
                raw_context = [word for sent in raw_context for word in sent]
                n_pick = min(self.mem_size, len(raw_context))
                context.fill(word2idx[''])
                context[b][:n_pick] = raw_context[-n_pick:]

                # x (bag-of-word of query)
                for word_id in data['querys'][i]:
                    x[b][word_id] += 1
                
            loss = self.sess.run(
                [self.loss],
                feed_dict={
                    self.input: x,
                    self.time: time,
                    self.target: target,
                    self.context: context
                }
            )
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size
    
    def test(self, data, label='Test'):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)

        m = self.mem_size
        for idx in xrange(N):
            if self.show: bar.next()
            target.fill(0)
            for b in xrange(self.batch_size):
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size:m]
                m += 1

                if m >= len(data):
                    m = self.mem_size

            loss = self.sess.run([self.loss], feed_dict={self.input: x,
                                                         self.time: time,
                                                         self.target: target,
                                                         self.context: context})
            cost += np.sum(loss)

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def run(self, train_data, test_data, word2idx, test_set_data):
        if not self.is_test:
            start = self.sess.run(self.global_step)
            for idx in xrange(self.nepoch):
                train_loss = np.sum(self.our_train(train_data, word2idx))
                test_loss = np.sum(self.our_test(test_data, word2idx, label='Validation'))

                # Logging
                self.log_loss.append([train_loss, test_loss])
                self.log_perp.append([math.exp(train_loss), math.exp(test_loss)])

                state = {
                    'perplexity': math.exp(train_loss),
                    'epoch': idx,
                    'learning_rate': self.current_lr,
                    'valid_perplexity': math.exp(test_loss)
                }
                print(state)
                
                answer = self.inference(test_set_data, word2idx)
                answer = pd.DataFrame(answer, columns=['answer'])
                answer.index += 1
                answer.to_csv('./guess/train_all_%d.csv' % start + idx, index_label='id')
                
                # Learning rate annealing
                if len(self.log_loss) > 1 and self.log_loss[idx][1] > self.log_loss[idx-1][1] * 0.9999:
                    self.current_lr = self.current_lr / 1.5
                    self.lr.assign(self.current_lr).eval()
                if self.current_lr < 1e-5: break

                if idx % 2 == 0:
                    self.saver.save(self.sess,
                                    os.path.join(self.checkpoint_dir, "MemN2N.model"),
                                    global_step=idx)
        else:
            self.load()

            valid_loss = np.sum(self.our_test(train_data, word2idx, label='Validation'))
            test_loss = np.sum(self.our_test(test_data, word2idx, label='Test'))

            state = {
                'valid_perplexity': math.exp(valid_loss),
                'test_perplexity': math.exp(test_loss)
            }
            print(state)
            
    def inference(self, data, word2idx):
        N = len(data['contexts'])

        context = np.ndarray([N, self.mem_size])
        time = np.ndarray([N, self.mem_size], dtype=np.int32)
        x = np.zeros([N, self.nwords], dtype=np.float32) # bag-of-word to encode a query

        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar(label, max=N)
            
        x.fill(0)
        for i in xrange(N):
            raw_context = data['contexts'][i]
            raw_context = [word for sent in raw_context for word in sent]
            n_pick = min(self.mem_size, len(raw_context))
            context.fill(word2idx[''])
            context[i][:n_pick] = raw_context[-n_pick:]

            # x (bag-of-word of query)
            for word_id in data['querys'][i]:
                x[i][word_id] += 1
                
        logits = self.sess.run(
            self.z,
            feed_dict={
                self.input: x,
                self.time: time,
                self.context: context
            }
        )
        print(logits.shape)
        guess = []
        for i in range(N):
            cands = np.array(data['candidates'][i])
            guess.append(np.argmax(logits[i, cands]))
        return guess

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
#         print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            restore_epoch = int(str(ckpt.model_checkpoint_path).split('-')[1])
            self.sess.run(tf.assign(self.global_step, restore_epoch + 1))
            print(' [v] Success to restore %s' % ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Trest mode but no checkpoint found")