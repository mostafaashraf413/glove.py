#!/usr/bin/env python

from argparse import ArgumentParser
import codecs
from collections import Counter
import itertools
from functools import partial
import logging
from math import log
import os.path
import cPickle as pickle
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse

#from util import listify


logger = logging.getLogger("glove")


def parse_args():
    parser = ArgumentParser(
        description=('Build a GloVe vector-space model from the '
                     'provided corpus'))

    parser.add_argument('corpus', metavar='corpus_path',
                        type=partial(codecs.open, encoding='utf-8'))

    g_vocab = parser.add_argument_group('Vocabulary options')
    g_vocab.add_argument('--vocab-path',
                         help=('Path to vocabulary file. If this path '
                               'exists, the vocabulary will be loaded '
                               'from the file. If it does not exist, '
                               'the vocabulary will be written to this '
                               'file.'))

    g_cooccur = parser.add_argument_group('Cooccurrence tracking options')
    g_cooccur.add_argument('--cooccur-path',
                           help=('Path to cooccurrence matrix file. If '
                                 'this path exists, the matrix will be '
                                 'loaded from the file. If it does not '
                                 'exist, the matrix will be written to '
                                 'this file.'))
    g_cooccur.add_argument('-w', '--window-size', type=int, default=10,
                           help=('Number of context words to track to '
                                 'left and right of each word'))
    g_cooccur.add_argument('--min-count', type=int, default=10,
                           help=('Discard cooccurrence pairs where at '
                                 'least one of the words occurs fewer '
                                 'than this many times in the training '
                                 'corpus'))

    g_glove = parser.add_argument_group('GloVe options')
    g_glove.add_argument('--vector-path',
                         help=('Path to which to save computed word '
                               'vectors'))
    g_glove.add_argument('-s', '--vector-size', type=int, default=100,
                         help=('Dimensionality of output word vectors'))
    g_glove.add_argument('--iterations', type=int, default=25,
                         help='Number of training iterations')
    g_glove.add_argument('--learning-rate', type=float, default=0.05,
                         help='Initial learning rate')
    g_glove.add_argument('--save-often', action='store_true', default=False,
                         help=('Save vectors after every training '
                               'iteration'))

    return parser.parse_args()


def get_or_build(path, build_fn, *args, **kwargs):
    """
    Load from serialized form or build an object, saving the built
    object.

    Remaining arguments are provided to `build_fn`.
    """

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        with open(path, 'rb') as obj_f:
            obj = msgpack.load(obj_f, use_list=False, encoding='utf-8')
    else:
        save = True

    if obj is None:
        obj = build_fn(*args, **kwargs)

        if save and path is not None:
            with open(path, 'wb') as obj_f:
                msgpack.dump(obj, obj_f)

    return obj


def build_vocab(corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.

    Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
    word ID and word corpus frequency.
    """

    logger.info("Building vocab from corpus")

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    logger.info("Done building vocab from corpus.")

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.iteritems())}


#@listify
def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """
    Build a word co-occurrence list for the given corpus.

    This function is a tuple generator, where each element (representing
    a cooccurrence pair) is of the form

        (i_main, i_context, cooccurrence)

    where `i_main` is the ID of the main word in the cooccurrence and
    `i_context` is the ID of the context word, and `cooccurrence` is the
    `X_{ij}` cooccurrence value as described in Pennington et al.
    (2014).

    If `min_count` is not `None`, cooccurrence pairs where either word
    occurs in the corpus fewer than `min_count` times are ignored.
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # Collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size) : center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # Distance from center word
                distance = contexts_len - left_i

                # Weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # Build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    
    cooccur_result = sparse.lil_matrix((vocab_size, vocab_size), dtype=np.float64)
    
    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            cooccur_result[i, j] = data[data_idx]
            
    return cooccur_result   


def run_iter(W, cooccur, s_cooccur):  
    eps = 1e-20
    global_cost = 0
   
    SW1 = cooccur.dot(W)
    #SW2 = np.mat(np.zeros(SW1.shape))
    #for i in xrange(SW2.shape[0]):
    #    SW2[i] = np.add(s_cooccur[i].multiply(W[i].dot(W.T)).dot(W), eps)
    SW2 = np.add(s_cooccur.multiply(W.dot(W.T)).dot(W), eps)

    W[:] = np.multiply(W , SW1/SW2)
    W[:] = np.maximum(W, eps)
    
    return global_cost


def train_glove(vocab, cooccurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):

    
    vocab_size = len(vocab)

    W = np.mat((np.random.rand(vocab_size, vector_size)) / float(vector_size + 1))

    s_cooccur = cooccurrences.copy()
    s_cooccur[s_cooccur > 0] = 1
    
    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(W, cooccurrences, s_cooccur)

        logger.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return np.array(W)


def save_model(W, path):
    with open(path, 'wb') as vector_f:
        pickle.dump(W, vector_f, protocol=2)

    logger.info("Saved vectors to %s", path)


def main(arguments):
    corpus = arguments.corpus

    logger.info("Fetching vocab..")
    vocab = get_or_build(arguments.vocab_path, build_vocab, corpus)
    logger.info("Vocab has %i elements.\n", len(vocab))

    logger.info("Fetching cooccurrence list..")
    corpus.seek(0)
    cooccurrences = get_or_build(arguments.cooccur_path,
                                 build_cooccur, vocab, corpus,
                                 window_size=arguments.window_size,
                                 min_count=arguments.min_count)
    logger.info("Cooccurrence list fetch complete (%i pairs).\n",
                len(cooccurrences))

    if arguments.save_often:
        iter_callback = partial(save_model, path=arguments.vector_path)
    else:
        iter_callback = None

    logger.info("Beginning GloVe training..")
    W = train_glove(vocab, cooccurrences,
                    iter_callback=iter_callback,
                    vector_size=arguments.vector_size,
                    iterations=arguments.iterations,
                    learning_rate=arguments.learning_rate)

    # TODO shave off bias values, do something with context vectors
    save_model(W, arguments.vector_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s\t%(message)s")
    main(parse_args())
