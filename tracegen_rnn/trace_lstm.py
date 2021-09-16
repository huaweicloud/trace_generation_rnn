"""Generic long-short-term memory neural network that outputs a vector
of logits, for flavor or duration modeling.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""

import torch
import torch.nn as nn


class TraceLSTM(nn.Module):
    """Generic LSTM for flavors or duration modeling.

    """

    def __init__(self, ninput, nhidden, noutput, nlayers):
        """Depending on the size of the input, output, and the array of hidden
        layers, add attributes for the inner LSTM and the
        fully-connected layers (including both weights and a bias
        term).

        """
        super().__init__()
        self.ninput = ninput
        self.nhidden = nhidden
        self.noutput = noutput
        self.nlayers = nlayers
        self.lstm = nn.LSTM(ninput, self.nhidden, self.nlayers)
        self.fc_out = nn.Linear(self.nhidden, noutput)
        self.hidden = self.init_hidden()

    def init_hidden(self, device=None, batch_size=1):
        """Before doing each new sequence, re-init hidden state to zeros.

        """
        hid0 = torch.zeros(self.nlayers, batch_size, self.nhidden)
        c_hid0 = torch.zeros(self.nlayers, batch_size, self.nhidden)
        if device is not None:
            hid0 = hid0.to(device)
            c_hid0 = c_hid0.to(device)
        return (hid0, c_hid0)

    def forward(self, minibatch):
        """Pass in a tensor of training examples of dimension LENGTH x
        BATCHSIZE x NINPUT, then run the forward pass. Returns tensor
        of LENGTH x BATCHSIZE x NOUTPUT.

        """
        lstm_out, self.hidden = self.lstm(minibatch, self.hidden)
        all_logits = self.fc_out(lstm_out)
        return all_logits

    def save(self, outfn):
        """Use the state-dict method of saving:

        """
        torch.save(self.state_dict(), outfn)

    @classmethod
    def create_from_path(cls, filename, device=None):
        """Factory method to return an instance of this class, given the model
        state-dict at the current filename. If device given,
        dynamically move model to device.

        """
        if device is not None:
            torch_device = torch.device(device)
            state_dict = torch.load(filename, map_location=torch_device)
        else:
            state_dict = torch.load(filename)
        nhidden = state_dict['fc_out.weight'].shape[1]
        noutput = state_dict['fc_out.weight'].shape[0]
        # LSTM layers have 4 values: ih/hh weights and ih/hh biases:
        nlayers = len(state_dict.keys()) // 4
        ninput = state_dict['lstm.weight_ih_l0'].shape[1]
        new_model = cls(ninput, nhidden, noutput, nlayers)
        new_model.load_state_dict(state_dict)
        return new_model
