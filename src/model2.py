'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from multiG import multiG
import pickle
from utils import circular_correlation, np_ccorr

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(object):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels1, num_ents1, num_rels2, num_ents2, method='distmult', bridge='CG', dim1=300, dim2=100, batch_sizeK1=512, batch_sizeK2=512, batch_sizeA=256, L1=False):
        self._num_relsA = num_rels1
        self._num_entsA = num_ents1
        self._num_relsB = num_rels2
        self._num_entsB = num_ents2
        self.method=method
        self.bridge=bridge
        self._dim1 = dim1
        self._dim2 = dim2
        self._hidden_dim = hid_dim = 50
        self._batch_sizeK1 = batch_sizeK1
        self._batch_sizeK2 = batch_sizeK2
        self._batch_sizeA = batch_sizeA
        self._epoch_loss = 0
        # margins
        self._m1 = 0.5
        self._m2 = 1.0
        self._mA = 0.5
        self.L1 = L1
        self.build()
        print("TFparts build up! Embedding method: ["+self.method+"]. Bridge method:["+self.bridge+"]")
        print("Margin Paramter: [m1] "+str(self._m1)+ " [m2] " +str(self._m2))

    @property
    def dim(self):
        return self._dim1, self._dim2  

    def build(self):
        tf.reset_default_graph()

        with tf.variable_scope("graph"):
            # Variables (matrix of embeddings/transformations)
            # KG1
            self._ht1 = ht1 = tf.get_variable(
                name='ht1',  # for t AND h
                shape=[self._num_entsA, self._dim1],
                dtype=tf.float32)
            self._r1 = r1 = tf.get_variable(
                name='r1',
                shape=[self._num_relsA, self._dim1],
                dtype=tf.float32)
            # KG2
            self._ht2 = ht2 = tf.get_variable(
                name='ht2',  # for t AND h
                shape=[self._num_entsB, self._dim2],
                dtype=tf.float32)
            self._r2 = r2 = tf.get_variable(
                name='r2',
                shape=[self._num_relsB, self._dim2],
                dtype=tf.float32)

            tf.summary.histogram("ht1", ht1)
            tf.summary.histogram("ht2", ht2)
            tf.summary.histogram("r1", r1)
            tf.summary.histogram("r2", r2)

            self._ht1_norm = tf.nn.l2_normalize(ht1, 1)
            self._ht2_norm = tf.nn.l2_normalize(ht2, 1)

            ######################## Graph A Loss #######################
            # Language A KM loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_t_index')
            self._A_hn_index = A_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_hn_index')
            self._A_tn_index = A_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK1],
                name='A_tn_index')
            '''
            A_loss_matrix = tf.subtract(
                tf.add(
                    tf.batch_matmul(A_h_ent_batch, tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim])),
                    A_rel_batch),
                tf.batch_matmul(A_t_ent_batch, tf.reshape(A_mat_h_batch, [-1, self.dim, self.dim]))
            )'''
            
            A_h_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_h_index), 1)
            A_t_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, A_t_index), 1)
            A_rel_batch = tf.nn.embedding_lookup(r1, A_r_index)
           
            A_hn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1,A_hn_index), 1)
            A_tn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1,A_tn_index), 1)

            if self.method == 'transe':
                ##### TransE score
                # This stores h + r - t
                A_loss_matrix = tf.subtract(tf.add(A_h_ent_batch, A_rel_batch), A_t_ent_batch)
                # This stores h' + r - t' for negative samples
                A_neg_matrix = tf.subtract(tf.add(A_hn_ent_batch, A_rel_batch), A_tn_ent_batch)
                if self.L1:
                    self._A_loss = A_loss = tf.reduce_sum(
                        tf.maximum(
                        tf.subtract(tf.add(tf.reduce_sum(tf.abs(A_loss_matrix), 1), self._m1),
                        tf.reduce_sum(tf.abs(A_neg_matrix), 1)), 
                        0.)
                    ) / self._batch_sizeK1
                else:
                    self._A_loss = A_loss = tf.reduce_sum(
                        tf.maximum(
                        tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(A_loss_matrix), 1)), self._m1),
                        tf.sqrt(tf.reduce_sum(tf.square(A_neg_matrix), 1))), 
                        0.)
                    ) / self._batch_sizeK1

            elif self.method == 'distmult':
                ##### DistMult score
                A_loss_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_h_ent_batch, A_t_ent_batch)), 1)
                A_neg_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, tf.multiply(A_hn_ent_batch, A_tn_ent_batch)), 1)

                self._A_loss = A_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(A_neg_matrix, A_loss_matrix), self._m1), 0.)) / self._batch_sizeK1
    
            elif self.method == 'hole':
                ##### HolE score
                A_loss_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, circular_correlation(A_h_ent_batch, A_t_ent_batch)), 1)
                A_neg_matrix = tf.reduce_sum(tf.multiply(A_rel_batch, circular_correlation(A_hn_ent_batch, A_tn_ent_batch)), 1)

                self._A_loss = A_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(A_neg_matrix, A_loss_matrix), self._m1), 0.)) / self._batch_sizeK1

            else:
                raise ValueError('Embedding method not valid!')


            ######################## Graph B Loss #######################
            # Language B KM loss : [|| h + r - t ||_2 + m1 - || h + r - t ||_2]+    here [.]+ means max (. , 0)
            self._B_h_index = B_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_h_index')
            self._B_r_index = B_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_r_index')
            self._B_t_index = B_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_t_index')
            self._B_hn_index = B_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_hn_index')
            self._B_tn_index = B_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeK2],
                name='B_tn_index')
            
            B_h_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_h_index), 1)
            B_t_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, B_t_index), 1)
            B_rel_batch = tf.nn.embedding_lookup(r2, B_r_index)
           
            B_hn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2,B_hn_index), 1)
            B_tn_ent_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2,B_tn_index), 1)


            if self.method == 'transe':
                #### TransE Score
                # This stores h + r - t
                B_loss_matrix = tf.subtract(tf.add(B_h_ent_batch, B_rel_batch), B_t_ent_batch)
                # This stores h' + r - t' for negative samples
                B_neg_matrix = tf.subtract(tf.add(B_hn_ent_batch, B_rel_batch), B_tn_ent_batch)
                if self.L1:
                    self._B_loss = B_loss = tf.reduce_sum(
                        tf.maximum(
                        tf.subtract(tf.add(tf.reduce_sum(tf.abs(B_loss_matrix), 1), self._m2),
                        tf.reduce_sum(tf.abs(B_neg_matrix), 1)), 
                        0.)
                    ) / self._batch_sizeK2
                else:
                    self._B_loss = B_loss = tf.reduce_sum(
                        tf.maximum(
                        tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(B_loss_matrix), 1)), self._m2),
                        tf.sqrt(tf.reduce_sum(tf.square(B_neg_matrix), 1))), 
                        0.)
                    ) / self._batch_sizeK2

            elif self.method == 'distmult':
                ##### DistMult score
                B_loss_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, tf.multiply(B_h_ent_batch, B_t_ent_batch)), 1)
                B_neg_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, tf.multiply(B_hn_ent_batch, B_tn_ent_batch)), 1)

                self._B_loss = B_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(B_neg_matrix, B_loss_matrix), self._m2), 0.)) / self._batch_sizeK2
            elif self.method == 'hole':
                ##### HolE score
                B_loss_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, circular_correlation(B_h_ent_batch, B_t_ent_batch)), 1)
                B_neg_matrix = tf.reduce_sum(tf.multiply(B_rel_batch, circular_correlation(B_hn_ent_batch, B_tn_ent_batch)), 1)

                self._B_loss = B_loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(B_neg_matrix, B_loss_matrix), self._m2), 0.)) / self._batch_sizeK2
            else:
                raise ValueError('Embedding method not valid!')


            ######################## Type Loss #######################
            self._AM_index1 = AM_index1 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_index1')
            self._AM_index2 = AM_index2 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_index2')
            
            self._AM_nindex1 = AM_nindex1 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_nindex1')
            self._AM_nindex2 = AM_nindex2 = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_sizeA],
                name='AM_nindex2')
            
            AM_ent1_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, AM_index1), 1)
            AM_ent2_batch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, AM_index2), 1)
            AM_ent1_nbatch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht1, AM_nindex1), 1)
            AM_ent2_nbatch = tf.nn.l2_normalize(tf.nn.embedding_lookup(ht2, AM_nindex2), 1)

            # Affine map
            self._M = M = tf.get_variable(name='M', shape=[self._dim1, self._dim2],initializer=orthogonal_initializer(),dtype=tf.float32)
            self._b = bias = tf.get_variable(name='b', shape=[self._dim2],initializer=tf.truncated_normal_initializer,dtype=tf.float32)
            self._Mc = Mc = tf.get_variable(name='Mc', shape=[self._dim2, self._hidden_dim],initializer=orthogonal_initializer(),dtype=tf.float32)
            self._bc = b_c = tf.get_variable(name='bc', shape=[self._hidden_dim],initializer=tf.truncated_normal_initializer,dtype=tf.float32)
            self._Me = Me = tf.get_variable(name='Me', shape=[self._dim1, self._hidden_dim],initializer=orthogonal_initializer(),dtype=tf.float32)
            self._be = b_e = tf.get_variable(name='be', shape=[self._hidden_dim],initializer=tf.truncated_normal_initializer,dtype=tf.float32)
                
            if self.bridge == 'CG':
                AM_pos_loss_matrix = tf.subtract( AM_ent1_batch, AM_ent2_batch )
                AM_neg_loss_matrix = tf.subtract( AM_ent1_nbatch, AM_ent2_nbatch )
            elif self.bridge == 'CMP-linear':
                # c - (W * e + b)
                #AM_pos_loss_matrix = tf.subtract( tf.add(tf.matmul(AM_ent1_batch, M),bias), AM_ent2_batch )
                AM_pos_loss_matrix = tf.subtract( tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent1_batch, M),bias), 1), AM_ent2_batch )
                AM_neg_loss_matrix = tf.subtract( tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent1_nbatch, M),bias), 1), AM_ent2_nbatch )
            elif self.bridge == 'CMP-single':
                # c - \sigma( W * e + b )
                #AM_pos_loss_matrix = tf.subtract( tf.tanh(tf.add(tf.matmul(AM_ent1_batch, M),bias)), AM_ent2_batch )
                AM_pos_loss_matrix = tf.subtract( tf.nn.l2_normalize( tf.tanh(tf.add(tf.matmul(AM_ent1_batch, M),bias)),1), AM_ent2_batch )
                AM_neg_loss_matrix = tf.subtract( tf.nn.l2_normalize( tf.tanh(tf.add(tf.matmul(AM_ent1_nbatch, M),bias)),1), AM_ent2_nbatch )
            elif self.bridge == 'CMP-double':
                # \sigma (W1 * c + bias1) - \sigma(W2 * c + bias1) --> More parameters to be defined
                #AM_pos_loss_matrix = tf.subtract( tf.add(tf.matmul(AM_ent1_batch, Me), b_e), tf.add(tf.matmul(AM_ent2_batch, Mc), b_c))
                #AM_pos_loss_matrix = tf.subtract( tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent1_batch, Me), b_e),1), tf.nn.l2_normalize(tf.add(tf.matmul(AM_ent2_batch, Mc), b_c),1))
                AM_pos_loss_matrix = tf.subtract( tf.nn.l2_normalize( tf.tanh(tf.add(tf.matmul(AM_ent1_batch, Me), b_e)),1), tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent2_batch, Mc), b_c)),1))
                AM_neg_loss_matrix = tf.subtract( tf.nn.l2_normalize( tf.tanh(tf.add(tf.matmul(AM_ent1_nbatch, Me), b_e)),1), tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent2_nbatch, Mc), b_c)),1))
            else:
                raise ValueError('Bridge method not valid!')

            '''
            if self.L1:
                self._AM_loss = AM_loss = tf.reduce_sum(
                tf.reduce_sum(tf.abs(AM_loss_matrix),1)
                ) / self._batch_sizeA
            else:
                self._AM_loss = AM_loss = tf.reduce_sum(
                tf.sqrt(tf.reduce_sum(tf.square(AM_loss_matrix), 1)
                )
                ) / self._batch_sizeA
            '''
            # hinge loss for AM pos and neg batch
            
            # Hinge Loss for AM
            if self.L1:
                self._AM_loss = AM_loss = tf.reduce_sum(
                    tf.maximum(
                        tf.subtract(tf.add(tf.reduce_sum(tf.abs(AM_pos_loss_matrix), 1), self._mA),
                            tf.reduce_sum(tf.abs(AM_neg_loss_matrix), 1)), 
                        0.)) / self._batch_sizeA
            else:
                self._AM_loss = AM_loss = tf.reduce_sum(
                    tf.maximum(
                        tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(AM_pos_loss_matrix), 1)), self._mA),
                            tf.sqrt(tf.reduce_sum(tf.square(AM_neg_loss_matrix), 1))), 
                        0.)) / self._batch_sizeA

            tf.summary.scalar("A_loss", A_loss)
            tf.summary.scalar("B_loss", B_loss)
            tf.summary.scalar("AM_loss", AM_loss)
            
            # Optimizer
            self._lr = lr = tf.placeholder(tf.float32)
            self._opt = opt = tf.train.AdamOptimizer(lr) #AdagradOptimizer(lr)#GradientDescentOptimizer(lr)
            self._train_op_A = train_op_A = opt.minimize(A_loss)
            self._train_op_B = train_op_B = opt.minimize(B_loss)
            self._train_op_AM = train_op_AM = opt.minimize(AM_loss)

            # Saver
            self.summary_op = tf.summary.merge_all()
            self._saver = tf.train.Saver()