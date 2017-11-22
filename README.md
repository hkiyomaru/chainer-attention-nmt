# Chainer implementation of Attention-based SEQ2SEQ model

Chainer-based implementation of Attention-based seq2seq model.

See "[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)", Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, arxiv 2014.

This repository is partly derived from [this repository](https://github.com/nakario/segnmt) and Chainer's official [seq2seq example](https://github.com/chainer/chainer/tree/master/examples/seq2seq).

# How to Run

```
$ python train.py <path/to/training-source> <path/to/training-target> <path/to/source-vocabulary> <path/to/target-vocabulary> --validation-source <path/to/validation-source> --validation-target <path/to/validation-target> -g <gpu-id>
```
