import chainer
import chainer.functions as F
import chainer.links as L


class AttentionSeq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(AttentionSeq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units, ignore_label=-1)
            self.embed_y = L.EmbedID(n_target_vocab, n_units, ignore_label=-1)
            self.encoder = Encoder(n_units)
            self.decoder = Decoder(n_units)
            self.W = L.Linear(n_units, n_target_vocab)
        
        self.n_units = n_units

    def __call__(self, xs, ys_in, ys_out):
        # Both xs and ys_in are lists of arrays.
        exs = self.embed_x(xs)
        eys = self.embed_y(ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        hxs = self.encoder(exs)
        os = self.decoder(eys, hxs)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, 'i')
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


class Attention(chainer.Chain):
    
    def __init__(self, n_units):
        super(Attention, self).__init__()
        with self.init_scope():
            self.eh = L.Linear(n_units * 2, n_units)  # forward + backward
            self.dh = L.Linear(n_units, n_units)
            self.hw = L.Linear(n_units, 1)
        
        self.n_units = n_units
    
    def __call__(self, hxs, hy):
        """Calculate attention for source sequences based on decoder's context.
        
        Args:
            hxs: Encoder's hidden states.
            dh: Decoder's hidden state.
        
        Returns:
            attention: Attention for each word of source sequences.
        
        """
        attention = None
        # apply `eh` to context vector

        # apply `dh` to decoder state
        
        # calculate the sum
        
        # apply `hw` to calculate attention  

        return attention


class Encoder(chainer.Chain):
    
    def __init__(self, n_units):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.forward = L.LSTM(n_units, n_units)
            self.backward = L.LSTM(n_units, n_units)
        
        self.n_units = n_units

    def __call__(self, exs):
        """Encode source sequences into the hidden vectors.

        Args:
            xs: Mini-batch of source sequences.
        
        Returns:
            hxs: Hidden states for source sequences.

        """
        self.__reset_state()

        exs = F.separate(exs, axis=1)

        fhxs = self.__encode(exs, self.forward)
        bhxs = reversed(self.__encode(reversed(exs), self.backward))
        hxs = [F.concat([fh, bh]) for fh, bh in zip(fhxs, bhxs)]

        return hxs

    def __encode(self, exs, transformer):
        states = []
        for ex in exs:
            states.append(transformer(ex))
        return states

    def __reset_state(self):
        self.forward.reset_state()
        self.backward.reset_state()


class Decoder(chainer.Chain):
    
    def __init__(self, n_units):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.forward = L.LSTM(n_units, n_units)
            self.attention = Attention(n_units)

        self.n_units = n_units

    def __call__(self, eys, hxs):
        """Encode source sequences into the hidden vectors.

        Args:
            eys: Mini-batch of target sequences.
            hxs: Annotation-vector of source sentences.

        Returns:
          dhs: Hidden states for target sequences.

        """
        eys = F.separate(eys, axis=1)
        for ey in eys:
            attention = self.attention()

    def __decode(self, eys, hxs):
        os = []
        for 
