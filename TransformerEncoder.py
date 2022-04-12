import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from Dataset import get_loader

class TransformerEncoderModel(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()
        self.model_type = 'TransformerEncoder'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self) -> None:
        # Inizialization according to - Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, src_mask: Tensor, pad_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
            pad_mask:

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, pad_mask)
        output = self.decoder(output)
        return output


    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(1)].permute(1,0,2)
        return self.dropout(x)


class LabelSmoothingDistribution(nn.Module):
    """
        Implementation of the Label Smoothing target distribution as described in the report
    """

    def __init__(self, smoothing_value, pad_id, trg_vocab_size, device):
        assert 0.0 <= smoothing_value <= 1.0

        super(LabelSmoothingDistribution, self).__init__()
        self.confidence = 1.0 - smoothing_value
        self.smoothing_value = smoothing_value
        self.device = device
        self.pad_id = pad_id
        self.trg_vocab_size = trg_vocab_size


    def forward(self, trg_token_ids_batch):

        batch_size = trg_token_ids_batch.shape[0]
        smooth_target_distributions = torch.zeros((batch_size, self.trg_vocab_size), device=self.device)
        smooth_target_distributions.fill_(self.smoothing_value / (self.trg_vocab_size - 2))
        smooth_target_distributions.scatter_(1, trg_token_ids_batch, self.confidence)
        smooth_target_distributions[:, self.pad_id] = 0.
        smooth_target_distributions.masked_fill_(trg_token_ids_batch == self.pad_id, 0.)

        return smooth_target_distributions





