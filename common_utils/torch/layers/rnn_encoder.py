import torch

from common_utils.torch.helpers import variable


class RNNEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, return_sequence=False, nb_layers=1):
        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.return_sequence = return_sequence
        self.nb_layers = nb_layers

        self.rnn = None

        if self.bidirectional:
            raise ValueError(f'Bidirectional encoder is not supported right now')

    def zero_state(self, batch_size):
        # The axes semantics are (num_layers, batch_size, hidden_dim)
        nb_layers = self.nb_layers if not self.bidirectional else self.nb_layers * 2
        state_shape = (nb_layers, batch_size, self.hidden_size)

        # will work on both GPU and CPU in contrast to just Variable(*state_shape)
        h = variable(torch.zeros(*state_shape))
        return h

    def get_hidden(self, cell_state):
        if self.nb_layers > 1:
            cell_state = cell_state[-1]
        else:
            cell_state = cell_state.squeeze(0)

        return cell_state

    def forward(self, inputs, lengths=None):
        batch_size = inputs.size(0)
        cell_state = self.zero_state(batch_size)

        if lengths is not None:
            # sort by length
            lengths_sorted, inputs_sorted_idx = lengths.sort(descending=True)
            inputs_sorted = inputs[inputs_sorted_idx]

            # pack sequences
            packed = torch.nn.utils.rnn.pack_padded_sequence(inputs_sorted, list(lengths_sorted.data), batch_first=True)

            outputs, cell_state = self.rnn(packed, cell_state)
            h = self.get_hidden(cell_state)

            # unpack sequences
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

            # un-sort
            _, inputs_unsorted_idx = inputs_sorted_idx.sort(descending=False)
            outputs = outputs[inputs_unsorted_idx]
            h = h[inputs_unsorted_idx]
        else:
            outputs, cell_state = self.rnn(inputs, cell_state)
            h = self.get_hidden(cell_state)

        if self.return_sequence:
            return outputs
        else:
            return h


class GRUEncoder(RNNEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rnn = torch.nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=self.bidirectional,
            num_layers=self.nb_layers, batch_first=True
        )


class LSTMEncoder(RNNEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rnn = torch.nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            bidirectional=self.bidirectional, num_layers=self.nb_layers, batch_first=True
        )

    def get_hidden(self, cell_state):
        return super().get_hidden(cell_state[0])

    def zero_state(self, batch_size):
        h0 = super().zero_state(batch_size)
        c0 = torch.zeros_like(h0)

        return h0, c0
