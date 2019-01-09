# Sentiment analysis of reviews

The goal of this task is to classify chunks of text into “positive” or “negative” sentiments (or a
rating scale).
The stanford movie review corpus with reviews from IMDB offers a good start for this task [1].
This corpus has 50k labeled movie reviews. This dataset was part of a Kaggle challenge last year, so
you can research a lot of approaches at the challenge website [2]. There's also a short tutorial on
using word2vec for this task.

If you're interested in using neural networks, you can find multiple examples for the IMDB task in
Keras [3]. Please note that the data in these examples is already preprocessed. You should also note
that these examples use only half of the corpus data (25k samples) so it might be for the better to
reimplement the preprocessing part yourself on the complete corpus (or you can use the script
provided at [4]). We recommend starting with simple 1-layer neural networks (MLP) before trying
the more advanced architectures presented in the examples. Keep in mind that simple models might
train faster and perform similarly well, provided you chose good features beforehand.
If you want to focus on other approaches, you can use scikit-learn to compare the performance of
different algorithms. An example can be found at [5].

[1]: http://ai.stanford.edu/~amaas/data/sentiment/
[2]: https://www.kaggle.com/c/word2vec-nlp-tutorial
[3]: https://github.com/fchollet/keras/blob/master/examples
[4]: http://deeplearning.net/tutorial/lstm.html
[5]: https://github.com/Poyuli/sentiment.analysis

