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

First you need to prepare parallel corpus.
Download 10^9 French-English corpus from WMT15 website.

[http://www.statmt.org/wmt15/translation-task.html](http://www.statmt.org/wmt15/translation-task.html)

```
$ sh download_wmt.sh
```

Now you can get six files:

* Source sentence file: `giga-fren.preprocess.en`
* Source vocabulary file: `vocab.en`
* Target sentence file: `giga-fren.preprocess.fr`
* Source vocabulary file: `vocab.fr`
* Source sentence file (validation): `newstest2013.preprocess.en`
* Target sentence file (validation): `newstest2013.preprocess.fr`

Then, let's start training.

```
$ python train.py giga-fren.preprocess.en giga-fren.preprocess.fr vocab.en vocab.fr --validation-source newstest2013.preprocess.en --validation-target newstest2013.preprocess.fr
```

See command line help for other options.
