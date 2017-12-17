"""Metrics to evaluate translation performance."""
from nltk.translate import bleu_score

import chainer

from utils import seq2seq_pad_concat_convert
from utils import get_subsequence_before_eos


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, key,
                 batch_size=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch_size):
                sources, targets = seq2seq_pad_concat_convert(
                    self.test_data[i:i + self.batch_size],
                    self.device
                )
                references.extend(
                    [[get_subsequence_before_eos(t).tolist()] for t in targets]
                )
                ys = [y.tolist() for y in self.model.translate(
                      sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})
