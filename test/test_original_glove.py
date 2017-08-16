# -*- coding: utf-8 -*-
import codecs
from glove import Glove
from glove import Corpus
import time

def read_corpus(filename):
    with codecs.open(filename, 'r', encoding='utf-8') as datafile:
        for line in datafile:
            yield line.lower().split(' ')
           
fileName = '../resources/test_corpus2.txt'
#co_matrix_name = '../resources/corpus.model'
#embed_model_name = '../resources/glove.model'

#######################################3 
print('Pre-processing corpus')

start_time = time.time()

corpus_model = Corpus()
corpus_model.fit(read_corpus(fileName), window=5)
#corpus_model.save(co_matrix_name)

print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)
#######################################


#######################################
print('Training the GloVe model')

start_time = time.time()

#corpus_model = Corpus.load(co_matrix_name)

glove = Glove(no_components=50, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
            no_threads=1, verbose=False)
glove.add_dictionary(corpus_model.dictionary)

elapsed_time = time.time() - start_time
print 'training is finished in %d'%(elapsed_time)

#glove.save(embed_model_name)
#######################################

query = 'income'
print('Loading pre-trained GloVe model')
#glove = Glove.load(embed_model_name)

print('Querying for %s' % query)
for i in glove.most_similar(query, number=10):
    print i[0],' ',i[1]