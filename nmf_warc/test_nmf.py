# -*- coding: utf-8 -*-
import logging
import utils
from nose.tools import assert_equal, assert_true
import numpy as np
from numpy.testing import assert_allclose
import evaluate_nmf as evaluate
import nmf_we as nmf_we


#Mock corpus (shamelessly stolen from Gensim word2vec tests)
test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")

print 'reading corpus from zipped warc file'
#test_corpus = utils.extract_arabic_warc('../resources/0000.warc.gz')

nmf_we.logger.setLevel(logging.ERROR)
print 'building vocab'
vocab = nmf_we.build_vocab(test_corpus)
print 'building cooccurence matrix'
cooccur = nmf_we.build_cooccur(vocab, test_corpus, window_size=10, min_count=1)
id2word = evaluate.make_id2word(vocab)
print 'start training the model'
W = nmf_we.train_glove(vocab, cooccur, vector_size=10, iterations=50)
print 'training is finished'

# Merge and normalize word vectors
W = evaluate.merge_main_context(W)


def test_similarity():
    similar = evaluate.most_similar(W, vocab, id2word, 'trees')
    logging.debug(similar)
    
    print similar

    assert('graph' in similar)
    
test_similarity()
