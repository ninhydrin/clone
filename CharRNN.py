import numpy as np
from chainer import Variable, FunctionSet
import chainer.functions as F
import chainer.links as L

class CharRNN(FunctionSet):
    def __init__(self, n_vocab, n_units):
        super(CharRNN, self).__init__(
            embed = F.EmbedID(n_vocab, n_units),
            l1_x = L.Linear(n_units, 4*n_units),
            l1_h = L.Linear(n_units, 4*n_units),
            l2_x = L.Linear(n_units, 4*n_units),
            l2_h = L.Linear(n_units, 4*n_units),
            l3   = L.Linear(n_units, n_vocab),
        )
        for param in self.parameters:
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)
        self.n_units = n_units

    def forward_one_step(self, x_data, y_data, state, train=True, dropout_ratio=0.5):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h0      = self.embed(x)
        h1_in   = self.l1_x(F.dropout(h0, ratio=dropout_ratio, train=train)) + self.l1_h(state['h1'])
        c1, h1  = F.lstm(state['c1'], h1_in)
        h2_in   = self.l2_x(F.dropout(h1, ratio=dropout_ratio, train=train)) + self.l2_h(state['h2'])
        c2, h2  = F.lstm(state['c2'], h2_in)
        y       = self.l3(F.dropout(h2, ratio=dropout_ratio, train=train))
        state   = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        if train:
            return state, F.softmax_cross_entropy(y, t)
        else:
            return state, F.softmax(y)

    def add_unit(self):
        add_layer =  self.embed
        add_layer.W.data=np.vstack((add_layer.W.data,np.array([0.1]*len(add_layer.W.data[0]),dtype=np.float32)))
        add_layer =  self.l3
        add_layer.W.data=np.vstack((add_layer.W.data,np.ones(add_layer.W.data.shape[1],dtype=np.float32)*np.average(add_layer.W.data)))
        add_layer.b.data=np.append(add_layer.b.data,np.average(add_layer.b.data))

    def make_initial_state(self, batchsize=50, train=True):
        return {name: Variable(np.zeros((batchsize, self.n_units), dtype=np.float32),
                               volatile=not train)
                for name in ('c1', 'h1', 'c2', 'h2')}
