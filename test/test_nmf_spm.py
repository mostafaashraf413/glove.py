import logging

from nose.tools import assert_equal, assert_true
import numpy as np
from numpy.testing import assert_allclose
import evaluate_nmf as evaluate
import nmf_we_spm as nmf_we
import util
import time


# Mock corpus (shamelessly stolen from Gensim word2vec tests)
#test_corpus = ("""human interface computer
#survey user computer system response time
#eps user interface system
#system human system eps
#user response time
#trees
#graph trees
#graph minors trees
#graph minors survey
#I like graph and stuff
#I like trees and stuff
#Sometimes I build a graph
#Sometimes I build trees""").split("\n")

start_time = time.time()

print 'reading corpus from zipped warc file'
test_corpus = util.read_txt_file('../resources/test_corpus.txt')

nmf_we.logger.setLevel(logging.ERROR)
print 'building vocab'
vocab = nmf_we.build_vocab(test_corpus)
print 'building cooccurence matrix'
cooccur = nmf_we.build_cooccur(vocab, test_corpus, window_size=5)
id2word = evaluate.make_id2word(vocab)
print 'start training the model'

W = nmf_we.train_glove(vocab, cooccur, vector_size=10, iterations=200)
# Merge and normalize word vectors
W = evaluate.merge_main_context(W)
elapsed_time = time.time() - start_time
print 'training is finished in %d'%(elapsed_time)

def test_similarity():
    similar = evaluate.most_similar(W, vocab, id2word, 'genetic')
    logging.debug(similar)
    
    print similar

    #assert_equal('trees', similar[0])
    
test_similarity()
