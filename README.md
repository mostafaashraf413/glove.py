This repository contains an implementation of the [GloVe][1] word vector learning algorithm in Python (NumPy + SciPy).

You can follow along with the [accompanying tutorial][2] on my blog.

The implementation is for educational purposes only; you should look elsewhere if you are looking for an efficient / robust solution.

[1]: http://www-nlp.stanford.edu/projects/glove/
[2]: http://www.foldl.me/2014/glove-python/

# Our Modification:
## You can find our work in "nmf_warc" folder.
## We have just modified the functions "train_glove", and "run_iter" in "glove.py".
## We replaced the original GloVe technique for matrix factorization with another one called "Mutiplicative Update Rule". This can be found in "nmf_warc/nmf_we_spm.py" script.
## For testing use "nmf_warc/test_nmf.py" script.

## For more information about "Mutiplicative Update Rule" read:

[3]: Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." Advances in neural information processing systems. 2001.

[4]: S. Zhang, W. Wang, J. Ford, and F. Makedon, “Learning from incomplete ratings using non-negative matrix-factorization,” in Proc. SIAM Int. Conf. Data Mining, Bethesda, MD, USA, 2006, pp. 549–553.

[5]: Luo, Xin, et al. "An efficient non-negative matrix-factorization-based approach to collaborative filtering for recommender systems." IEEE Transactions on Industrial Informatics 10.2 (2014): 1273-1284.

