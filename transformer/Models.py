''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s, _ = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder_embedding(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_src_vocab, d_word_vec, d_model, n_position, dropout):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        # print(n_src_vocab)
        self.src_word_emb = nn.Linear(n_src_vocab, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq):

        enc_output = self.src_word_emb(src_seq)
        # enc_output = self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        return enc_output


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, enc_input, src_mask, return_attns=False):

        enc_slf_attn_list = []
        enc_output = enc_input
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder_embedding(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, d_model, n_position, dropout):

        super().__init__()

        self.trg_word_emb = nn.Linear(n_trg_vocab, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq):

        dec_output = self.dropout(self.trg_word_emb(trg_seq))
        dec_output = self.layer_norm(dec_output)

        return dec_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, dec_input, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = dec_input
        for dec_layer in self.layer_stack:
            # print(dec_output.size(), enc_output.size())
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab_list, d_word_vec, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, n_position):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = -2333.4445, -2333.4445

        self.encoder_embedding = Encoder_embedding(
            n_src_vocab=n_src_vocab, d_word_vec=d_word_vec, d_model=d_model,
            n_position=n_position, dropout=dropout)

        self.encoder_1 = Encoder(
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model,
            d_inner=d_inner, dropout=dropout)

        self.encoder_2 = Encoder(
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model,
            d_inner=d_inner, dropout=dropout)

        # self.decoder_embedding = Decoder_embedding(
        #     n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec, d_model=d_model,
        #     dropout=dropout, n_position=n_position)

        self.decoder_1 = Encoder(
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model,
            d_inner=d_inner, dropout=dropout)

        # self.decoder_2 = Encoder(
        #     n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model,
        #     d_inner=d_inner, dropout=dropout)

        self.trg_embed_list_11, self.trg_embed_list_12 = [], []
        self.trg_embed_list_21, self.trg_embed_list_22 = [], []

        for n_trg_vocab in n_trg_vocab_list:
            self.trg_word_prj_11 = nn.Linear(d_model, 1, bias=False)
            self.trg_word_prj_12 = nn.Linear(n_position, n_trg_vocab, bias=False)
            self.trg_word_prj_21 = nn.Linear(d_model, 1, bias=False)
            self.trg_word_prj_22 = nn.Linear(n_position, n_trg_vocab, bias=False)
            self.trg_embed_list_11.append(self.trg_word_prj_11)
            self.trg_embed_list_12.append(self.trg_word_prj_12)
            self.trg_embed_list_21.append(self.trg_word_prj_21)
            self.trg_embed_list_22.append(self.trg_word_prj_22)

        self.trg_embed_list_11 = nn.ModuleList(self.trg_embed_list_11)
        self.trg_embed_list_12 = nn.ModuleList(self.trg_embed_list_12)
        self.trg_embed_list_21 = nn.ModuleList(self.trg_embed_list_21)
        self.trg_embed_list_22 = nn.ModuleList(self.trg_embed_list_22)
        self.trg_word_prj_3 = nn.Linear(d_model, n_src_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_seq, num_models):

        src_mask = None

        enc_input = self.encoder_embedding(src_seq)
        enc_output_1, *_ = self.encoder_1(enc_input, src_mask)
        enc_output_2, *_ = self.encoder_2(enc_output_1, src_mask)
        dec_output_1, *_ = self.decoder_1(enc_output_2, src_mask)
        # dec_output_2, *_ = self.decoder_2(dec_output_1, src_mask)
        dec_logit = self.trg_word_prj_3(dec_output_1)

        seq_logit_1_output, seq_logit_2_output = [], []

        enc_output_2_t = enc_output_2

        for id in range(num_models):
            seq_logit_1 = self.trg_embed_list_11[id](enc_output_2_t)
            seq_logit_1 = self.trg_embed_list_12[id](seq_logit_1.squeeze())
            seq_logit_2 = self.trg_embed_list_21[id](enc_output_2_t)
            seq_logit_2 = self.trg_embed_list_22[id](seq_logit_2.squeeze())
            seq_logit_1_output.append(seq_logit_1)
            seq_logit_2_output.append(seq_logit_2)

        seq_logit_1_inte = torch.cat(seq_logit_1_output, dim=1)
        seq_logit_2_inte = torch.cat(seq_logit_2_output, dim=1)
        seq_logit = torch.cat((seq_logit_1_inte.unsqueeze(2), seq_logit_2_inte.unsqueeze(2)), dim=2)

        return seq_logit, dec_logit