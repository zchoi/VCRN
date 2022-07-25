import torch
import torch.nn as nn
import time
import numpy as np
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn import preprocessing
# from spherecluster import SphericalKMeans
import h5py
from contextlib import contextmanager
import warnings
from typing import Union, Sequence, Tuple

TensorOrNone = Union[torch.Tensor, None]

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self._is_stateful = False
        self._state_names = []
        self._state_defaults = dict()

    def register_state(self, name: str, default: TensorOrNone):
        self._state_names.append(name)
        if default is None:
            self._state_defaults[name] = None
        else:
            self._state_defaults[name] = default.clone().detach()
        self.register_buffer(name, default)

    def states(self):
        for name in self._state_names:
            yield self._buffers[name]
        for m in self.children():
            if isinstance(m, Module):
                yield from m.states()

    def apply_to_states(self, fn):
        for name in self._state_names:
            self._buffers[name] = fn(self._buffers[name])
        for m in self.children():
            if isinstance(m, Module):
                m.apply_to_states(fn)

    def _init_states(self, batch_size: int):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)
                self._buffers[name] = self._buffers[name].unsqueeze(0)
                self._buffers[name] = self._buffers[name].expand([batch_size, ] + list(self._buffers[name].shape[1:]))
                self._buffers[name] = self._buffers[name].contiguous()

    def _reset_states(self):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)

    def enable_statefulness(self, batch_size: int):
        for m in self.children():
            if isinstance(m, Module):
                m.enable_statefulness(batch_size)
        self._init_states(batch_size)
        self._is_stateful = True

    def disable_statefulness(self):
        for m in self.children():
            if isinstance(m, Module):
                m.disable_statefulness()
        self._reset_states()
        self._is_stateful = False

    @contextmanager
    def statefulness(self, batch_size: int):
        self.enable_statefulness(batch_size)
        try:
            yield
        finally:
            self.disable_statefulness()

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out


class obj_net(nn.Module):
    def __init__(self, opt) -> None:
        super(obj_net, self).__init__()
        self.cnn_proj = nn.Linear(opt.a_feature_size + opt.m_feature_size, 1024)
        self.obj_proj = nn.Linear(opt.a_feature_size, 1024)

        self.att = nn.Linear(1024, 1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, cnn_feats, object_feat):
        '''
        cnn_feats: [bsz, num_f, 4096]
        object_feat: [bsz, num_f, num_o, 2048]
        '''

        cnn_proj = self.cnn_proj(cnn_feats)
        obj_proj = self.obj_proj(object_feat)

        att_feat = cnn_proj.unsqueeze(2) * obj_proj
        att_score = self.att(att_feat)

        att_score = self.softmax(att_score) # [bsz, num_f, num_o, 1]

        agg_obj = (att_score * obj_proj).mean(2).squeeze() # [bsz, num_f, 1024]

        return agg_obj

class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
  

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output):
        # MHA+AddNorm
        self_att = self.self_att(input, input, input, None)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        # MHA+AddNorm:Image
        enc_att = self.enc_att(self_att, enc_output, enc_output, None)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))

        ff = self.pwff(enc_att)
        return ff

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.a_feature_size = opt.a_feature_size
        self.m_feature_size = opt.m_feature_size
        self.hidden_size = opt.hidden_size
        self.concat_size = self.a_feature_size + self.m_feature_size
        self.use_multi_gpu = opt.use_multi_gpu
        self.dataset = opt.dataset

        self.layers_1 = nn.ModuleList(
            [DecoderLayer(self.concat_size, 512, 512, 8, self.concat_size, 0.1) for _ in range(1)])
        self.layers_2 = nn.ModuleList(
            [DecoderLayer(self.concat_size, 512, 512, 8, self.concat_size, 0.1) for _ in range(2)])
        self.layers_3 = nn.ModuleList(
            [DecoderLayer(self.concat_size, 512, 512, 8, self.concat_size, 0.1) for _ in range(3)])
        # frame feature embedding
        self.frame_feature_embed = nn.Linear(self.concat_size, self.hidden_size)

        if opt.dataset == 'msvd':
            print("using {}".format('msvd_concept_feat_train.h5'))
            self.msvd_concept300_train = h5py.File('data/MSVD/msvd_concept300_feat_train.h5','r')['concept_features'][:,:]
            self.msvd_concept500_train = h5py.File('data/MSVD/msvd_concept500_feat_train.h5','r')['concept_features'][:,:]
            self.msvd_concept1000_train = h5py.File('data/MSVD/msvd_concept1000_feat_train.h5','r')['concept_features'][:,:]
            self.register_buffer('concept300_feat_train', torch.from_numpy(self.msvd_concept300_train).float())
            self.register_buffer('concept500_feat_train', torch.from_numpy(self.msvd_concept500_train).float())
            self.register_buffer('concept1000_feat_train', torch.from_numpy(self.msvd_concept1000_train).float())
        else:
            print("using {}".format('msrvtt_concept_feat_train.h5'))
            self.msrvtt_concept300_train = h5py.File('data/MSRVTT/msrvtt_concept300_feat_train.h5','r')['concept_features'][:,:]
            self.msrvtt_concept500_train = h5py.File('data/MSRVTT/msrvtt_concept500_feat_train.h5','r')['concept_features'][:,:]
            self.msrvtt_concept1000_train = h5py.File('data/MSRVTT/msrvtt_concept1000_feat_train.h5','r')['concept_features'][:,:]
            self.register_buffer('concept300_feat_train', torch.from_numpy(self.msrvtt_concept300_train).float())
            self.register_buffer('concept500_feat_train', torch.from_numpy(self.msrvtt_concept500_train).float())
            self.register_buffer('concept1000_feat_train', torch.from_numpy(self.msrvtt_concept1000_train).float())


    def _init_weights(self):
        nn.init.xavier_normal_(self.frame_feature_embed.weight)
        nn.init.constant_(self.frame_feature_embed.bias, 0)

    def _init_lstm_state(self, d):
        batch_size = d.size(0)
        lstm_state_h = d.data.new(2, batch_size, self.hidden_size).zero_()
        lstm_state_c = d.data.new(2, batch_size, self.hidden_size).zero_()
        return lstm_state_h, lstm_state_c

    # forward for test
    def forward(self, cnn_feats):
        '''
        :param cnn_feats: (batch_size, max_frames, m_feature_size + a_feature_size)
        :param region_feats: (batch_size, max_frames, num_boxes, region_feature_size)
        :param spatial_feats: (batch_size, max_frames, num_boxes, spatial_feature_size)
        :return: output of Bidirectional LSTM and embedded region features
        '''
        # 2d cnn or 3d cnn or 2d+3d cnn

        assert self.a_feature_size + self.m_feature_size == cnn_feats.size(2)
        
        cnn_feats_attention, cnn_feats_attention2, cnn_feats_attention3 = cnn_feats, cnn_feats, cnn_feats
        # for i, l in enumerate(self.layers_3):
        #     cnn_feats_attention = l(cnn_feats_attention, self.concept300_feat_train.expand(cnn_feats.size(0),-1,-1))
        # for i, l in enumerate(self.layers_2):
        #     cnn_feats_attention2 = l(cnn_feats_attention2, self.concept500_feat_train.expand(cnn_feats.size(0),-1,-1))
        for i, l in enumerate(self.layers_1):
            cnn_feats_attention3 = l(cnn_feats_attention3, self.concept1000_feat_train.expand(cnn_feats.size(0),-1,-1))

        frame_feats = self.frame_feature_embed(cnn_feats + 0.2 * cnn_feats_attention3)

        # if self.training:
        #     for i, l in enumerate(self.layers):
        #         cnn_feats_attention = l(cnn_feats_attention, self.concept_feat_train.expand(cnn_feats.size(0),-1,-1))
        # else:
        #     for i, l in enumerate(self.layers):
        #         cnn_feats_attention = l(cnn_feats_attention, self.concept_feat_val.expand(cnn_feats.size(0),-1,-1))

        # frame_feats = self.frame_feature_embed(cnn_feats)

        return frame_feats


