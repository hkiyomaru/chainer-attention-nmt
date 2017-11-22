"""Attention-based seq2seq model."""
import numpy as np

import chainer
from chainer import Variable
from chainer import Parameter
import chainer.functions as F
import chainer.links as L

PAD = -1
UNK = 0
EOS = 1


class Seq2seq(chainer.Chain):

    def __init__(self, n_source_vocab, n_target_vocab,
                 n_encoder_layers, n_encoder_units, n_encoder_dropout,
                 n_decoder_units, n_attention_units, n_maxout_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(
                n_source_vocab,
                n_encoder_layers,
                n_encoder_units,
                n_encoder_dropout
            )
            self.decoder = Decoder(
                n_target_vocab,
                n_decoder_units,
                n_attention_units,
                n_encoder_units * 2,  # because of bi-directional lstm
                n_maxout_units,
            )

    def __call__(self, xs, ys):
        """Calculate loss between outputs and ys.

        Args:
            xs: Source sentences.
            ys: Target sentences.

        Returns:
            loss: Cross-entoropy loss between outputs and ys.

        """
        batch_size = len(xs)

        hxs = self.encoder(xs)
        os = self.decoder(ys, hxs)

        concatenated_os = F.concat(os, axis=0)
        concatenated_ys = F.flatten(ys)
        n_words = len(self.xp.where(concatenated_ys.data != EOS)[0])

        loss = F.sum(
            F.softmax_cross_entropy(
                concatenated_os, concatenated_ys, reduce='no', ignore_label=PAD
            )
        )
        loss = loss / n_words
        chainer.report({'loss': loss.data}, self)
        perp = self.xp.exp(loss.data * batch_size / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        """Generate sentences based on xs.

        Args:
            xs: Source sentences.

        Returns:
            ys: Generated target sentences.

        """
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            hxs = self.encoder(xs)
            ys = self.decoder.translate(hxs, max_length)
        return ys


class Encoder(chainer.Chain):

    def __init__(self, n_vocab, n_layers, n_units, dropout):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.bilstm = L.NStepBiLSTM(n_layers, n_units, n_units, dropout)

    def __call__(self, xs):
        """Encode source sequences into the representations.

        Args:
            xs: Source sequences.

        Returns:
            hxs: Hidden states for source sequences.

        """
        batch_size, max_length = xs.shape

        exs = self.embed_x(xs)
        exs = F.separate(exs, axis=0)
        masks = self.xp.vsplit(xs != -1, batch_size)
        masked_exs = [ex[mask.reshape((-1, ))] for ex, mask in zip(exs, masks)]

        _, _, hxs = self.bilstm(None, None, masked_exs)
        hxs = F.pad_sequence(hxs, length=max_length, padding=0.0)
        return hxs


class Decoder(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_attention_units,
                 n_encoder_output_units, n_maxout_units, n_maxout_pools=2):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed_y = L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.lstm = L.StatelessLSTM(
                n_units + n_encoder_output_units,
                n_units
            )
            self.maxout = L.Maxout(
                n_units + n_encoder_output_units + n_units,
                n_maxout_units,
                n_maxout_pools
            )
            self.w = L.Linear(n_units, n_vocab)
            self.attention = AttentionModule(
                n_encoder_output_units,
                n_attention_units,
                n_units
            )
            self.bos_state = Parameter(
                initializer=self.xp.random.randn(1, n_units).astype('f')
            )
        self.n_units = n_units

    def __call__(self, ys, hxs):
        """Calculate cross-entoropy loss between predictions and ys.

        Args:
            ys: Target sequences.
            hxs: Hidden states for source sequences.

        Returns:
            os: Probability density for output sequences.

        """
        batch_size, max_length, encoder_output_size = hxs.shape

        compute_context = self.attention(hxs)
        # initial cell state
        c = Variable(self.xp.zeros((batch_size, self.n_units), 'f'))
        # initial hidden state
        h = F.broadcast_to(self.bos_state, ((batch_size, self.n_units)))
        # initial character's embedding
        previous_embedding = self.embed_y(
            Variable(self.xp.full((batch_size, ), EOS, 'i'))
        )

        os = []
        for y in self.xp.hsplit(ys, ys.shape[1]):
            y = y.reshape((batch_size, ))
            context = compute_context(h)
            concatenated = F.concat((previous_embedding, context))

            c, h = self.lstm(c, h, concatenated)
            concatenated = F.concat((concatenated, h))
            o = self.w(self.maxout(concatenated))

            os.append(o)
            previous_embedding = self.embed_y(y)
        return os

    def translate(self, hxs, max_length):
        """Generate target sentences given hidden states of source sentences.

        Args:
            hxs: Hidden states for source sequences.

        Returns:
            ys: Generated sequences.

        """
        batch_size, _, _ = hxs.shape
        compute_context = self.attention(hxs)
        c = Variable(self.xp.zeros((batch_size, self.n_units), 'f'))
        h = F.broadcast_to(self.bos_state, ((batch_size, self.n_units)))
        # first character's embedding
        previous_embedding = self.embed_y(
            Variable(self.xp.full((batch_size, ), EOS, 'i'))
        )

        results = []
        for _ in range(max_length):
            context = compute_context(h)
            concatenated = F.concat((previous_embedding, context))

            c, h = self.lstm(c, h, concatenated)
            concatenated = F.concat((concatenated, h))

            logit = self.w(self.maxout(concatenated))
            y = F.reshape(F.argmax(logit, axis=1), (batch_size, ))

            results.append(y)
            previous_embedding = self.embed_y(y)
        else:
            results = F.separate(F.transpose(F.vstack(results)), axis=0)

        ys = []
        for result in results:
            index = np.argwhere(result.data == EOS)
            if len(index) > 0:
                result = result[:index[0, 0] + 1]
            ys.append(result.data)
        return ys


class AttentionModule(chainer.Chain):

    def __init__(self, n_encoder_output_units,
                 n_attention_units, n_decoder_units):
        super(AttentionModule, self).__init__()
        with self.init_scope():
            self.h = L.Linear(n_encoder_output_units, n_attention_units)
            self.s = L.Linear(n_decoder_units, n_attention_units)
            self.o = L.Linear(n_attention_units, 1)
        self.n_encoder_output_units = n_encoder_output_units
        self.n_attention_units = n_attention_units

    def __call__(self, hxs):
        """Returns a function that calculates context given decoder's state.

        Args:
            hxs: Encoder's hidden states.

        Returns:
            compute_context: A function to calculate attention.

        """
        batch_size, max_length, encoder_output_size = hxs.shape

        encoder_factor = F.reshape(
            self.h(
                F.reshape(
                    hxs,
                    (batch_size * max_length, self.n_encoder_output_units)
                )
            ),
            (batch_size, max_length, self.n_attention_units)
        )

        def compute_context(previous_state):
            decoder_factor = F.broadcast_to(
                F.reshape(
                    self.s(previous_state),
                    (batch_size, 1, self.n_attention_units)
                ),
                (batch_size, max_length, self.n_attention_units)
            )

            attention = F.softmax(
                F.reshape(
                    self.o(
                        F.reshape(
                            F.tanh(encoder_factor + decoder_factor),
                            (batch_size * max_length, self.n_attention_units)
                        )
                    ),
                    (batch_size, max_length)
                )
            )

            context = F.reshape(
                F.batch_matmul(attention, hxs, transa=True),
                (batch_size, encoder_output_size)
            )
            return context

        return compute_context
