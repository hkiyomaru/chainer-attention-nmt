# Chainer Implementation of Attentional SEQ2SEQ Model

Chainer-based implementation of Attention-based seq2seq model.

See "[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)", Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, arxiv 2014.

This repository is partly derived from [this repository](https://github.com/nakario/segnmt) and Chainer's official [seq2seq example](https://github.com/chainer/chainer/tree/master/examples/seq2seq).

# Development Environment

* Ubuntu 16.04
* Python 3.5.2
* Chainer 3.1.0
* numpy 1.13.3
* cupy 2.1.0
* nltk
* progressbar
* and their dependencies

# How to Run

```
$ python train.py <path/to/training-source> <path/to/training-target> <path/to/source-vocabulary> <path/to/target-vocabulary> --validation-source <path/to/validation-source> --validation-target <path/to/validation-target> -g <gpu-id>
```
