# Relaxed Conformal Prediction Cascades
Code for [Relaxed Conformal Prediction Cascades for Efficient Inference Over Many Labels](https://people.csail.mit.edu/fisch/assets/pdf/conformal_cascades.pdf).

### Abstract
Providing a small set of promising candidates in place of a single prediction is well-suited for many open-ended classification tasks. <ins>Conformal Prediction</ins> (CP) is a technique for creating classifiers that produce a valid set of predictions that contains the true answer with arbitrarily high probability. In practice, however, standard CP can suffer from both low *predictive* and *computational* efficiency during inference—i.e., the predicted set is both unusably large and costly to obtain. This is particularly pervasive in the considered setting, where the correct answer is not unique and the number of total possible answers is high. In this work, we develop two simple and complementary techniques for improving both types of efficiencies. First, we relax CP validity to arbitrary criterions of success—allowing our framework to make more efficient predictions while remaining "equivalently correct." Second, we amortize cost by conformalizing prediction cascades, in which we aggressively prune implausible labels early on by using progressively stronger classifiers—while still guaranteeing marginal coverage. We demonstrate the empirical effectiveness of our approach for multiple applications in natural language processing and computational chemistry for drug discovery.

### Conformal Prediction

All of the code for analyzing cascaded conformal predictions is in the [cpcascades](cpcascades) directory. Examples for how to call into it are given in the tasks subdirectories. Note that in this repository, the functions exposed in `cpcascades` are for analysis only, i.e. they are not implemented as a real-time predictors.

Efficient implementations of online predictors for the tasks considered here might be included later.

### Tasks

Code for training models for extractive question answering ([qa](qa)), information retrieval for fact verification ([ir](ir)), and in-silico screening for drug discovery ([hiv](hiv)), can be found in their respective sub-directories (which also contain further instuctions).

The outputs we used in our experiments are available in the `data/predictions` directory after downloading and untarring the data:

```
wget https://people.csail.mit.edu/fisch/assets/data/cpcascade/data.tar.gz && tar -xvf data.tar.gz
```
