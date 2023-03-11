# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn
from torch.nn import functional as F

from param import args
from lxrt.entry_gen_gqa_distill_latest import LXRTEncoder
from lxrt.modeling_gen_gqa_distill_latest import BertLayerNorm, GeLU

from lxrt.fc import FCNet, GTH
from lxrt.attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
from torch.nn.utils.weight_norm import weight_norm
import torch
import random
from copy import deepcopy

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


# class MyMemory(nn.Module):
#     def __init__(self, m_size, hid_size):
#         super(MyMemory, self).__init__()
#         # 记忆模块大小
#         self.m_size = m_size
#         # 记忆模块维度
#         self.m_dim = hid_size
#         # self.write_trans = weight_norm(nn.Linear(self.m_dim, self.m_dim), dim=None)
#         self.write_trans = nn.Linear(self.m_dim, self.m_dim)
#         # 写入概率变换
#         # self.write_prob = weight_norm(nn.Linear(self.m_dim, self.m_size), dim=None)
#         self.write_prob = nn.Linear(self.m_dim, self.m_size)
#         # 读取概率变换
#         # self.read_prob = weight_norm(nn.Linear(self.m_dim, self.m_size), dim=None)
#         self.read_prob = nn.Linear(self.m_dim, self.m_size)
#         # self.read_gate = weight_norm(nn.Linear(self.m_dim, 1), dim=None)
#         self.read_gate = nn.Linear(self.m_dim, 1)
#         # self.memo_layer_norm = BertLayerNorm(hid_size, eps=1e-12)

#     def write(self, memo, query_input):
#         # [b, len, dim]
#         query_content = F.tanh(self.write_trans(query_input))
#         # [b, dim]
#         query_content_mean = query_content.mean(dim=1)
#         # [b, dim]
#         query_mean = query_input.mean(dim=1)
#         # [b, m_size]
#         # 计算写入概率
#         write_prob = F.softmax(self.write_prob(query_mean) * 100, dim=0)
#         # [m_size, b] x [b, dim] = [m_size, dim]
#         updated_memo = torch.mm(write_prob.t(), query_content_mean)
#         # updated_memo = torch.mm(write_prob.t(), query_mean)
#         # 得到更新记忆特征
#         new_memo = (memo + updated_memo) / 2
#         return new_memo

#     def read(self, memo, query_input):
#         query_mean = query_input.mean(dim=1)
#         # [b, m_size]
#         # 计算读取概率
#         read_prob = F.softmax(self.read_prob(query_mean) * 100, dim=1)
#         read_gate = torch.sigmoid(self.read_gate(query_mean))
#         # [b, dim]
#         update_embed = torch.mm(read_prob, memo)
#         return update_embed, read_gate

#     def forward(self, memo, action, input_embed=None):
#         if action == 'write':
#             new_memo = self.write(memo, input_embed)
#             return new_memo
#         else:
#             output, read_gate = self.read(memo, input_embed)
#             # output = self.memo_layer_norm(output)
#             return output, read_gate.unsqueeze(2)


class MyMemory(nn.Module):
    def __init__(self, m_size, hid_size):
        super(MyMemory, self).__init__()
        # 记忆模块大小
        self.m_size = m_size
        # 记忆模块维度
        self.m_dim = hid_size
        # self.write_trans = weight_norm(nn.Linear(self.m_dim, self.m_dim), dim=None)
        self.write_trans = nn.Linear(self.m_dim, self.m_dim)
        # 写入概率变换
        # self.write_prob = weight_norm(nn.Linear(self.m_dim, self.m_size), dim=None)
        self.write_prob = nn.Linear(self.m_dim, self.m_size)
        self.compute_prob = nn.Linear(self.m_dim * 2, 1)
        # 读取概率变换
        # self.read_prob = weight_norm(nn.Linear(self.m_dim, self.m_size), dim=None)
        self.read_prob = nn.Linear(self.m_dim, self.m_size)
        # self.read_gate = weight_norm(nn.Linear(self.m_dim, 1), dim=None)
        self.read_gate = nn.Linear(self.m_dim, 1)
        # self.memo_layer_norm = BertLayerNorm(hid_size, eps=1e-12)

    def write(self, memo, query_input):
        # [b, len, dim]
        query_content = F.tanh(self.write_trans(query_input))
        # [b, dim]
        query_content_mean = query_content.mean(dim=1)
        # [b, dim]
        query_mean = query_input.mean(dim=1)
        # [b, m_size]
        # 计算写入概率
        # [b, m_size, d]
        temp_memo = memo.unsqueeze(0)
        temp_memo = temp_memo.repeat(query_input.shape[0], 1, 1)
        # [b, m_size, d]
        temp_query = query_mean.unsqueeze(dim=1)
        temp_query = temp_query.repeat(1, memo.shape[0], 1)
        # [b, m_size, 2d]
        temp_feat = torch.cat([temp_memo, temp_query], dim=2)
        # [b, m_size]
        prob = F.softmax(torch.squeeze(self.compute_prob(temp_feat)), dim=1)
        updated_memo = torch.mm(prob.t(), query_content_mean)
        # write_prob = F.softmax(self.write_prob(query_mean) * 100, dim=0)
        # [m_size, b] x [b, dim] = [m_size, dim]
        # updated_memo = torch.mm(write_prob.t(), query_content_mean)
        # updated_memo = torch.mm(write_prob.t(), query_mean)
        # 得到更新记忆特征
        new_memo = (memo + updated_memo) / 2
        return new_memo

    def read(self, memo, query_input):
        query_mean = query_input.mean(dim=1)
        # [b, m_size]
        # 计算读取概率
        read_prob = F.softmax(self.read_prob(query_mean) * 100, dim=1)
        read_gate = torch.sigmoid(self.read_gate(query_mean))
        # [b, m_size] * [m_size, dim] = [b, dim]
        update_embed = torch.mm(read_prob, memo)
        return update_embed, read_gate

    def forward(self, memo, action, input_embed=None):
        if action == 'write':
            new_memo = self.write(memo, input_embed)
            return new_memo
        else:
            output, read_gate = self.read(memo, input_embed)
            # output = self.memo_layer_norm(output)
            return output, read_gate.unsqueeze(2)


class FeatureGenerator(nn.Module):
    '''
    特征生成模块
    '''
    def __init__(self, hidden_size):
        super(FeatureGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.generator = FCNet([hidden_size, hidden_size, hidden_size // 2, hidden_size, hidden_size], norm='weight', act='ReLU', dropout=0)
        self.gene_norm_layer = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_feat):
        new_feat = self.generator(input_feat)
        new_feat = self.gene_norm_layer(new_feat)
        return new_feat


class NoiseGenerator(nn.Module):
    def __init__(self, hidden_size):
        super(NoiseGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.generator = FCNet([hidden_size, hidden_size, hidden_size // 2, hidden_size, hidden_size], norm='weight',
                               act='ReLU', dropout=0)
        self.gene_norm_layer = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_feat):
        new_feat = torch.zeros_like(input_feat)
        nn.init.normal_(new_feat)
        new_feat = self.generator(new_feat)
        new_feat = self.gene_norm_layer(new_feat)
        return new_feat


class squeeze(nn.Module):
    def __init__(self):
        super(squeeze, self).__init__()

    def forward(self, input):
        return input.squeeze()


class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        self.freeze_lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        # m_size = 50
        m_size = 10

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        # self.txt_generator = NoiseGenerator(hid_dim)
        # self.img_generator = NoiseGenerator(hid_dim)

        self.txt_generator = FeatureGenerator(hid_dim)
        self.img_generator = FeatureGenerator(hid_dim)

        self.txt_memory = MyMemory(m_size, hid_dim)
        self.img_memory = MyMemory(m_size, hid_dim)
        #
        self.img_memo = nn.Parameter(torch.zeros(m_size, hid_dim, dtype=torch.float), requires_grad=False)
        self.txt_memo = nn.Parameter(torch.zeros(m_size, hid_dim, dtype=torch.float), requires_grad=False)

        ########### init our layers ###########
        activation = 'ReLU'
        norm = 'weight'

        self.fuse = FCNet([hid_dim * 2, hid_dim], norm=norm, act=activation, dropout=0)

        # q_only_branch
        self.q_only = FCNet([hid_dim, hid_dim, hid_dim], norm=norm, act=activation, dropout=0)
        self.q_cls = weight_norm(nn.Linear(hid_dim, num_answers), dim=None)

        self.v_only = FCNet([hid_dim, hid_dim, hid_dim], norm=norm, act=activation, dropout=0)
        self.v_cls = weight_norm(nn.Linear(hid_dim, num_answers), dim=None)

        # q_detect_bias
        self.q_detect = nn.Sequential(
            weight_norm(nn.Linear(hid_dim, 1), dim=None),
            nn.Sigmoid()
        )

        # v detect_bias
        self.v_detect = nn.Sequential(
            weight_norm(nn.Linear(hid_dim, 1), dim=None),
            nn.Sigmoid()
        )

        # q and v bias weight
        self.q_and_v_bias_detect = nn.Sequential(
            weight_norm(nn.Linear(hid_dim, 1)),
            squeeze(),
            nn.Softmax(dim=-1)
        )

        # fusion_branch
        self.debias_only = FCNet([hid_dim, hid_dim], norm=norm, dropout=0, act=activation)
        self.debias_cls = weight_norm(nn.Linear(hid_dim, num_answers), dim=None)

    def use_bias(self, l_cls, v_cls, x, out):
        # detach, avoid backforward propogation to train front layers
        q_emb_only = l_cls.detach()
        v_emb_only = v_cls.detach()
        joint_emb = x.detach()

        # q_only
        q_only_emb = self.q_only(q_emb_only)  # [batch, num_hid]
        q_only_logits = self.q_cls(q_only_emb)  # [batch, num_ans]
        q_bias_detect = self.q_detect(q_only_emb).view(q_only_emb.size(0), 1)  # [batch, 1]

        # v_only
        v_only_emb = self.v_only(v_emb_only)
        v_only_logits = self.v_cls(v_only_emb)
        v_bias_detect = self.v_detect(v_only_emb).view(v_only_emb.size(0), 1)  # [batch, 1]

        bad_q_bias = q_bias_detect * q_only_emb
        bad_v_bias = v_bias_detect * v_only_emb

        # bias weight
        bad_bias = torch.stack([bad_q_bias, bad_v_bias], dim=1)
        bias_weight = self.q_and_v_bias_detect(bad_bias)

        bad_bias_ = bad_q_bias * bias_weight[:, 0].unsqueeze(1) + bad_v_bias * bias_weight[:, 1].unsqueeze(1)

        debias_emb_raw = joint_emb - bad_bias_
        debias_emb = self.debias_only(debias_emb_raw)
        debias_logits = self.debias_cls(debias_emb)

        out['q_logits'] = q_only_logits
        out['v_logits'] = v_only_logits
        out['debias_logits'] = debias_logits
        out['fea'] = joint_emb  # joint_repr
        out['pos_fea'] = debias_emb
        out['neg_fea'] = bad_bias_
        return out

    def forward(self, feat, pos, sent, self_sup=False, feat_gen=False, enhanced=False, out={}):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        new_feat = deepcopy(feat)
        new_pos = deepcopy(pos)
        new_sent = deepcopy(sent)

        # (lang_feats, visn_feats, l_cls, v_cls), x = self.lxrt_encoder(sent, (feat, pos), mode='none')

        if feat_gen:
            if self.training:
                # new_x, gen_lang_feat = self.freeze_lxrt_encoder(new_sent, (new_feat, new_pos), mode='txt', txt_generator=self.txt_generator, img_generator=self.img_generator, txt_memory=self.txt_memory, img_memory=self.img_memory, img_memo=self.img_memo, txt_memo=self.txt_memo)
                new_x, gen_lang_feat, freeze_vis_feat, txt_ver_loss = self.freeze_lxrt_encoder(new_sent, (new_feat, new_pos), mode='txt', enhanced=enhanced, txt_generator=self.txt_generator, img_generator=self.img_generator, txt_memory=self.txt_memory, img_memory=self.img_memory, img_memo=self.img_memo, txt_memo=self.txt_memo)
                img_matched_scores = self.freeze_lxrt_encoder.itm_output(new_x)
                # img_matched_scores = self.lxrt_encoder.itm_output(new_x)

                # new_x, gen_vis_feat = self.freeze_lxrt_encoder(new_sent, (new_feat, new_pos), mode='vis', txt_generator=self.txt_generator, img_generator=self.img_generator, txt_memory=self.txt_memory, img_memory=self.img_memory, img_memo=self.img_memo, txt_memo=self.txt_memo)
                new_x, gen_vis_feat, freeze_lang_feat, vis_ver_loss = self.freeze_lxrt_encoder(new_sent, (new_feat, new_pos), mode='vis', enhanced=enhanced, txt_generator=self.txt_generator, img_generator=self.img_generator, txt_memory=self.txt_memory, img_memory=self.img_memory, img_memo=self.img_memo, txt_memo=self.txt_memo)
                txt_matched_scores = self.freeze_lxrt_encoder.itm_output(new_x)
                # txt_matched_scores = self.lxrt_encoder.itm_output(new_x)

                out['img_matched_scores'] = img_matched_scores
                out['txt_matched_scores'] = txt_matched_scores

                out['txt_ver_loss'] = txt_ver_loss
                out['vis_ver_loss'] = vis_ver_loss

                # lang_gen, vis_gen = self.freeze_lxrt_encoder(new_sent, (new_feat, new_pos), mode='generation',
                #                                              txt_generator=self.txt_generator,
                #                                              img_generator=self.img_generator,
                #                                              txt_memory=self.txt_memory, img_memory=self.img_memory,
                #                                              img_memo=self.img_memo, txt_memo=self.txt_memo)
                # lang_gen = lang_gen.mean(dim=1)
                # vis_gen = vis_gen.mean(dim=1)
                # # fused_gen = self.fuse(torch.cat([lang_gen, vis_gen], dim=1))
                # # out['fused_gen'] = fused_gen
                # # out['ori_multimodal'] = x.detach()
                # out['lang_gen'] = lang_gen
                # out['vis_gen'] = vis_gen
                # out['lang_ori'] = l_cls.detach()
                # out['vis_ori'] = v_cls.detach()

                # vis_itm, lang_itm = self.lxrt_encoder(sent, (feat, pos), mode='itm', txt_generator=self.txt_generator,
                #                                       img_generator=self.img_generator, txt_memory=self.txt_memory,
                #                                       img_memory=self.img_memory, img_memo=self.img_memo,
                #                                       txt_memo=self.txt_memo, gen_lang_feat=gen_lang_feat,
                #                                       gen_vis_feat=gen_vis_feat)

                # vis_itm_scores = self.lxrt_encoder.itm_output(vis_itm)
                # lang_itm_scores = self.lxrt_encoder.itm_output(lang_itm)
                # out['vis_itm_scores'] = vis_itm_scores
                # out['lang_itm_scores'] = lang_itm_scores
                freeze_gen_itm = self.freeze_lxrt_encoder(sent, (feat, pos), mode='itm', txt_generator=self.txt_generator,
                                                      img_generator=self.img_generator, txt_memory=self.txt_memory,
                                                      img_memory=self.img_memory, img_memo=self.img_memo,
                                                      txt_memo=self.txt_memo, gen_lang_feat=gen_lang_feat,
                                                      gen_vis_feat=gen_vis_feat, freeze_vis_feat=freeze_vis_feat,
                                                      freeze_lang_feat=freeze_lang_feat)
                gen_matched_scores = self.freeze_lxrt_encoder.itm_output(freeze_gen_itm)

                finetune_gen_itm = self.lxrt_encoder(sent, (feat, pos), mode='itm', txt_generator=self.txt_generator,
                                                    img_generator=self.img_generator, txt_memory=self.txt_memory,
                                                    img_memory=self.img_memory, img_memo=self.img_memo,
                                                    txt_memo=self.txt_memo, gen_lang_feat=gen_lang_feat, gen_vis_feat=gen_vis_feat, freeze_vis_feat=freeze_vis_feat, freeze_lang_feat=freeze_lang_feat)
                # vis_itm_scores = self.lxrt_encoder.itm_output(vis_itm)
                gen_itm_scores = self.lxrt_encoder.itm_output(finetune_gen_itm)
                out['gen_matched_scores'] = gen_matched_scores
                out['gen_itm_scores'] = gen_itm_scores

            # x, l_cls, v_cls = self.lxrt_encoder(sent, (feat, pos), mode='both', txt_generator=self.txt_generator,
            #                                     img_generator=self.img_generator, txt_memory=self.txt_memory,
            #                                     img_memory=self.img_memory, img_memo=self.img_memo,
            #                                     txt_memo=self.txt_memo)
            x, l_cls, v_cls, _ = self.lxrt_encoder(sent, (feat, pos), mode='both', txt_generator=self.txt_generator, img_generator=self.img_generator, txt_memory=self.txt_memory, img_memory=self.img_memory, img_memo=self.img_memo, txt_memo=self.txt_memo)

        logit = self.logit_fc(x)
        out['logits'] = logit

        # out = self.use_bias(l_cls, v_cls, x, out)

        # if self_sup:
        #     # construct an irrelevant Q-I pair for each instance
        #     batch_size = lang_feats.size(0)
        #     # V
        #     index_v = random.sample(range(0, batch_size), batch_size)
        #     neg_v = feat[index_v]
        #     neg_pos = pos[index_v]
        #     _, x = self.lxrt_encoder(sent, (neg_v, neg_pos), mode='none')
        #     out['logit_neg_v'] = self.logit_fc(x)
        #     # Q
        #     index_q = random.sample(range(0, batch_size), batch_size)
        #     neg_sent = [sent[i] for i in index_q]
        #     _, x = self.lxrt_encoder(neg_sent, (feat, pos), mode='none')
        #     out['logit_neg_q'] = self.logit_fc(x)

        # if self_sup:
        #     # new self
        #     # construct an irrelevant Q-I pair for each instance
        #     batch_size = feat.size(0)
        #     # V
        #     index_v = random.sample(range(0, batch_size), batch_size)
        #     neg_v = feat[index_v]
        #     neg_pos = pos[index_v]
        #     x, _, _ = self.lxrt_encoder(sent, (neg_v, neg_pos), mode='both', txt_generator=self.txt_generator, img_generator=self.img_generator, txt_memory=self.txt_memory, img_memory=self.img_memory, img_memo=self.img_memo, txt_memo=self.txt_memo)
        #     out['logit_neg_v'] = self.logit_fc(x)
        #     # Q
        #     index_q = random.sample(range(0, batch_size), batch_size)
        #     neg_sent = [sent[i] for i in index_q]
        #     x, _, _ = self.lxrt_encoder(neg_sent, (feat, pos), mode='both', txt_generator=self.txt_generator, img_generator=self.img_generator, txt_memory=self.txt_memory, img_memory=self.img_memory, img_memo=self.img_memo, txt_memo=self.txt_memo)
        #     out['logit_neg_q'] = self.logit_fc(x)

        return out