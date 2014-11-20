__author__ = 'OscarInn'

import theano
import theano.tensor as T
import time
import numpy
import pandas as pn
from numpy.random import RandomState
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function


class dA(object):
   """Denoising Auto-Encoder class (dA)

   A denoising autoencoders tries to reconstruct the input from a corrupted
   version of it by projecting it first in a latent space and reprojecting
   it afterwards back in the input space. Please refer to Vincent et al.,2008
   for more details. If x is the input then equation (1) computes a partially
   destroyed version of x by means of a stochastic mapping q_D. Equation (2)
   computes the projection of the input into the latent space. Equation (3)
   computes the reconstruction of the input, while equation (4) computes the
   reconstruction error.

   .. math::

       \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

       y = s(W \tilde{x} + b)                                           (2)

       x = s(W' y  + b')                                                (3)

       L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

   """

   def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500,
              W=None, bhid=None, bvis=None):
       """
       Initialize the dA class by specifying the number of visible units (the
       dimension d of the input ), the number of hidden units ( the dimension
       d' of the latent or hidden space ) and the corruption level. The
       constructor also receives symbolic variables for the input, weights and
       bias. Such a symbolic variables are useful when, for example the input is
       the result of some computations, or when weights are shared between the
       dA and an MLP layer. When dealing with SdAs this always happens,
       the dA on layer 2 gets as input the output of the dA on layer 1,
       and the weights of the dA are used in the second stage of training
       to construct an MLP.

       :type numpy_rng: numpy.random.RandomState
       :param numpy_rng: number random generator used to generate weights

       :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
       :param theano_rng: Theano random generator; if None is given one is generated
                    based on a seed drawn from `rng`

       :type input: theano.tensor.TensorType
       :param input: a symbolic description of the input or None for standalone
                     dA

       :type n_visible: int
       :param n_visible: number of visible units

       :type n_hidden: int
       :param n_hidden:  number of hidden units

       :type W: theano.tensor.TensorType
       :param W: Theano variable pointing to a set of weights that should be
                 shared belong the dA and another architecture; if dA should
                 be standalone set this to None

       :type bhid: theano.tensor.TensorType
       :param bhid: Theano variable pointing to a set of biases values (for
                    hidden units) that should be shared belong dA and another
                    architecture; if dA should be standalone set this to None

       :type bvis: theano.tensor.TensorType
       :param bvis: Theano variable pointing to a set of biases values (for
                    visible units) that should be shared belong dA and another
                    architecture; if dA should be standalone set this to None


       """
       self.n_visible = n_visible
       self.n_hidden = n_hidden

       # create a Theano random generator that gives symbolic random values
       if not theano_rng :
           theano_rng = RandomStreams(rng.randint(2 ** 30))

       # note : W' was written as `W_prime` and b' as `b_prime`
       if not W:
           # W is initialized with `initial_W` which is uniformely sampled
           # from -4.*sqrt(6./(n_visible+n_hidden)) and 4.*sqrt(6./(n_hidden+n_visible))
           # the output of uniform if converted using asarray to dtype
           # theano.config.floatX so that the code is runable on GPU
           initial_W = numpy.asarray(numpy_rng.uniform(
                     low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                     high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                     size=(n_visible, n_hidden)), dtype=theano.config.floatX)
           W = theano.shared(value=initial_W, name='W')

       if not bvis:
           bvis = theano.shared(value = numpy.zeros(n_visible,
                                        dtype=theano.config.floatX), name='bvis')

       if not bhid:
           bhid = theano.shared(value=numpy.zeros(n_hidden,
                                             dtype=theano.config.floatX), name='bhid')

       self.W = W
       # b corresponds to the bias of the hidden
       self.b = bhid
       # b_prime corresponds to the bias of the visible
       self.b_prime = bvis
       # tied weights, therefore W_prime is W transpose
       self.W_prime = self.W.T
       self.theano_rng = theano_rng
       # if no input is given, generate a variable representing the input
       if input == None:
           # we use a matrix because we expect a minibatch of several examples,
           # each example being a row
           self.x = T.dmatrix(name='input')
       else:
           self.x = input

       self.params = [self.W, self.b, self.b_prime]

   def get_corrupted_input(self, input, corruption_level):
       """ This function keeps ``1-corruption_level`` entries of the inputs the same
       and zero-out randomly selected subset of size ``coruption_level``
       Note : first argument of theano.rng.binomial is the shape(size) of
              random numbers that it should produce
              second argument is the number of trials
              third argument is the probability of success of any trial

               this will produce an array of 0s and 1s where 1 has a probability of
               1 - ``corruption_level`` and 0 with ``corruption_level``
       """
       return  self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input


   def get_hidden_values(self, input):
       """ Computes the values of the hidden layer """
       return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

   def get_reconstructed_input(self, hidden ):
       """ Computes the reconstructed input given the values of the hidden layer """
       return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

   def get_cost_updates(self, corruption_level, learning_rate):
       """ This function computes the cost and the updates for one trainng
       step of the dA """

       tilde_x = self.get_corrupted_input(self.x, corruption_level)
       y = self.get_hidden_values( tilde_x)
       z = self.get_reconstructed_input(y)
       # note : we sum over the size of a datapoint; if we are using minibatches,
       #        L will  be a vector, with one entry per example in minibatch
       L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1 )
       # note : L is now a vector, where each element is the cross-entropy cost
       #        of the reconstruction of the corresponding example of the
       #        minibatch. We need to compute the average of all these to get
       #        the cost of the minibatch
       cost = T.mean(L)

       # compute the gradients of the cost of the `dA` with respect
       # to its parameters
       gparams = T.grad(cost, self.params)
       # generate the list of updates
       updates = []
       for param, gparam in zip(self.params, gparams):
           updates.append((param, T.cast(param - learning_rate * gparam, theano.config.floatX)))

       return cost, updates



if __name__ == '__main__':

    def shared_dataset(data_x, borrow=True):

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x

    train_set_x = shared_dataset(pn.read_csv('word_embeddings').as_matrix()[:,1:51])

    # ========== hyper-parameters ==========
    learning_rate = .1
    batch_size = 500
    training_epochs = 200
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    # ========== ==================== ==========

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x', dtype=theano.config.floatX)  # the data is presented as rasterized images

    ######################
    # BUILDING THE MODEL #
    ######################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=50, n_hidden=10)

    cost, updates = da.get_cost_updates(corruption_level=0.2,
                                        learning_rate=learning_rate)


    train_da = theano.function([index], cost, updates=updates,
                               givens = {x: train_set_x[(index * batch_size):((index + 1) * batch_size)]})

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print ('Training took %f minutes' % (training_time / 60.))




    inp = T.dvector('inp')
    encoded = da.get_hidden_values(inp)

    #encoded = T.mean(inp)
    encode_da = theano.function([inp], encoded)


    v = vecOf('two', wrds, vecs).tolist()
    encode_da(np.array(v, dtype=theano.config.floatX))





