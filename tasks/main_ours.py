# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
from os import path as osp

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import sys
import numpy as np

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from model_ours import VQAModel, FeatureGenerator
from copy import deepcopy

import os
import psutil
import progressbar
import utils_1
import utils
import json
process = psutil.Process(os.getpid())

import torch.nn.functional as F


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N or None).start()
    for v, b, q, a, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        logits = model(v, b, list(q))
        pred[idx:idx + batch_size, :].copy_(logits['logits'].data)
        qIds[idx:idx + batch_size].copy_(i)
        idx += batch_size

    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils_1.assert_eq(logits.size(0), len(qIds))

    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


def compute_score(annotations, json_file):
    annotations = sorted(annotations, key=lambda x: x['question_id'])
    predictions = sorted(json.load(open(json_file)), key=lambda x: x['question_id'])

    score = 0
    count = 0
    other_score = 0
    yes_no_score = 0
    num_score = 0
    yes_count = 0
    other_count = 0
    num_count = 0
    upper_bound = 0
    upper_bound_num = 0
    upper_bound_yes_no = 0
    upper_bound_other = 0

    for pred, anno in zip(predictions, annotations):
        if pred['question_id'] == anno['question_id']:
            G_T = max(anno['answer_count'].values())
            upper_bound += min(1, G_T / 3)
            if pred['answer'] in anno['answers_word']:
                proba = anno['answer_count'][pred['answer']]
                score += min(1, proba / 3)
                count += 1
                if anno['answer_type'] == 'yes/no':
                    yes_no_score += min(1, proba / 3)
                    upper_bound_yes_no += min(1, G_T / 3)
                    yes_count += 1
                if anno['answer_type'] == 'other':
                    other_score += min(1, proba / 3)
                    upper_bound_other += min(1, G_T / 3)
                    other_count += 1
                if anno['answer_type'] == 'number':
                    num_score += min(1, proba / 3)
                    upper_bound_num += min(1, G_T / 3)
                    num_count += 1
            else:
                score += 0
                yes_no_score += 0
                other_score += 0
                num_score += 0
                if anno['answer_type'] == 'yes/no':
                    upper_bound_yes_no += min(1, G_T / 3)
                    yes_count += 1
                if anno['answer_type'] == 'other':
                    upper_bound_other += min(1, G_T / 3)
                    other_count += 1
                if anno['answer_type'] == 'number':
                    upper_bound_num += min(1, G_T / 3)
                    num_count += 1

    print('count:', count, ' score:', round(score * 100 / len(annotations), 2))
    print('Yes/No:', round(100 * yes_no_score / yes_count, 2), 'Num:', round(100 * num_score / num_count, 2),
          'other:', round(100 * other_score / other_count, 2))

    print('count:', len(annotations), ' upper_bound:', round(score * upper_bound / len(annotations)), 2)
    print('upper_bound_Yes/No:', round(100 * upper_bound_yes_no / yes_count, 2), 'upper_bound_Num:',
          round(100 * upper_bound_num / num_count, 2), 'upper_bound_other:',
          round(100 * upper_bound_other / other_count, 2))


@torch.no_grad()
def evaluate(model, dataloader, path=''):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = 0

    ood_vis_feat = []
    ood_lang_feat = []
    ood_feat = []
    timer = utils.TimeMeter()
    for i, (v, b, q, a, q_id, cap) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        q = list(q)
        q_id = q_id.cuda()
        out_dict = model(v, b, q, list(cap), feat_gen=True)
        # out_dict = model(v, b, q, feat_gen=False)

        ood_vis_feat.append(out_dict['ood_vis_feat'].detach().cpu().numpy())
        ood_lang_feat.append(out_dict['ood_lang_feat'].detach().cpu().numpy())
        ood_feat.append(out_dict['ood_feat'].detach().cpu().numpy())
        if len(ood_vis_feat) > 200:
            ood_vis_feat = np.concatenate(ood_vis_feat, axis=0)
            ood_lang_feat = np.concatenate(ood_lang_feat, axis=0)
            np.save(path + 'ood_vis_feat.npy', ood_vis_feat)
            np.save(path + 'ood_lang_feat.npy', ood_lang_feat)
            ood_feat = np.concatenate(ood_feat, axis=0)
            np.save(path + 'ood_feat.npy', ood_feat)
            break

        pred = out_dict['logits']
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)
        timer.update()

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    print("{:.3f} seconds/batch".format(1.0 / timer.avg))
    return score, upper_bound


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


class RankingLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, max_violation=False):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        im = F.normalize(im, dim=1)
        s = F.normalize(s, dim=1)
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            mask = mask.cuda()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # return cost_s.sum() + cost_im.sum()
        return cost_s.mean() + cost_im.mean()


class Contrastive_loss(nn.Module):
    def __init__(self, tao=1):
        super(Contrastive_loss, self).__init__()
        self.sim = nn.CosineSimilarity(dim=-1)
        self.tao = tao

    def forward(self, fea, pos_fea, neg_fea):
        fea = F.normalize(fea, dim=1)
        pos_fea = F.normalize(pos_fea, dim=1)
        neg_fea = F.normalize(neg_fea, dim=1)

        pos_sim = self.sim(fea, pos_fea)
        neg_sim = self.sim(fea, neg_fea)

        logits = torch.exp(pos_sim / self.tao) / \
            (torch.exp(pos_sim / self.tao) + torch.exp(neg_sim / self.tao))
        loss = (-1.0 * torch.log(logits))

        return loss.mean()


# multi-label soft loss
def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(
        F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(
        F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)

    qice_loss = neg_top_k.mean()
    return qice_loss


def get_our_data():
    # from dataset_vqacp_lxmert_valid_ID import Dictionary, VQAFeatureDataset
    from dataset_vqacp_lxmert_valid_ID_Pre import Dictionary, VQAFeatureDataset
    # from dataset_vqav2_lxmert import Dictionary, VQAFeatureDataset
    import utils_1
    from src.param import args as opt

    dictionary = Dictionary.load_from_file(os.path.join(opt.dataroot, 'dictionary.pkl'))
    opt.ntokens = dictionary.ntoken

    print('Loading Train Data...')
    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio,
                                   adaptive=False)  # load labeld data
    print('Loading OOD Test Data...')
    ood_dset = VQAFeatureDataset('test_OOD', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False)
    print('Loading ID Test Data...')
    id_dset = VQAFeatureDataset('test_ID', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False)
    print('loading Valid Data')
    eval_dset = VQAFeatureDataset('valid', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False)
    # print('Loading Test Data...')
    # test_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False)

    # train_loader = DataLoader(train_dset, opt.batch_size, shuffle=True, num_workers=4, collate_fn=utils_1.trim_collate)
    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)
    opt.use_all = 1
    ood_loader = DataLoader(ood_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)
    id_loader = DataLoader(id_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)
    val_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)
    # ood_loader = DataLoader(ood_dset, 1800, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)
    # id_loader = DataLoader(id_dset, 1800, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)
    # val_loader = DataLoader(eval_dset, 1800, shuffle=False, num_workers=4, collate_fn=utils_1.trim_collate)
    # test_loader = DataLoader(test_dset, 1800, shuffle=False, num_workers=1, collate_fn=utils_1.trim_collate)
    return train_loader, ood_loader, id_loader, val_loader


class VQA:
    def __init__(self, folder="/", load=True):
        # Datasets
        self.train_loader, self.ood_loader, self.id_loader, self.val_loader = get_our_data()
        self.model = VQAModel(2274)
        # self.model = VQAModel(2410)

        print('Loading Weights...')
        # Load pre-trained weights
        if args.load_lxmert is not None:

            # model_data = torch.load('/home/ygsong/project/D-VQA/LXMERT/snap/vqa_ours_dvqa_best/best_freeze_logits_distill_wo_dvqa_cross_new_memo_valid_ID/BEST.pth')
            # model_data = {key.replace('module.', ''): value for key, value in model_data.items()}
            # print('loading full modal')
            # self.model.load_state_dict(model_data)
            # print('init logit_fc...')

            model_data = torch.load('/home/ygsong/project/D-VQA/LXMERT/snap/vqa_ours_dvqa_best/best_freeze_logits_distill_wo_dvqa_best_nomatch_ver_iamback/BEST.pth')
            model_data = {key.replace('module.', ''): value for key, value in model_data.items()}
            print('loading full modal')
            self.model.load_state_dict(model_data)

            # self.model.lxrt_encoder.load(args.load_lxmert)
            # self.model.freeze_lxrt_encoder.load(args.load_lxmert)

            # self.model.logit_fc.apply(self.model.lxrt_encoder.model.init_bert_weights)

        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = Contrastive_loss(tao=1)
        self.ranking_loss = RankingLoss(margin=0.2)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        if load:
            if 'bert' in args.optim:
                batch_per_epoch = len(self.train_loader)
                t_total = int(batch_per_epoch * args.epochs)
                print("BertAdam Total Iters: %d" % t_total)
                from lxrt.optimization import BertAdam
                # self.optim = BertAdam(list(self.model.parameters()),
                #                       lr=args.lr,
                #                       warmup=0.1,
                #                       t_total=t_total)
                params = self.get_optimized_para()
                self.optim = BertAdam(params,
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=t_total)
            else:
                self.optim = args.optimizer(self.model.parameters(), args.lr)
            # Output Directory
            self.output = args.output
            os.makedirs(self.output, exist_ok=True)

        self.best_model_state = None

    def get_optimized_para(self):
        # param_top = [p for n, p in self.model.named_parameters()
        #              if 'freeze_lxrt_encoder' in n or 'generator' in n or 'memory' in n]

        # param_optimizer = [p for n, p in self.model.named_parameters()
        #              if 'freeze_lxrt_encoder' not in n and 'generator' not in n and 'memory' not in n]

        # param_optimizer = [p for n, p in self.model.named_parameters()
        #                    if 'freeze_lxrt_encoder' not in n or 'itm_output' not in n]

        param_optimizer = [p for n, p in self.model.named_parameters()
                           if 'freeze_lxrt_encoder' not in n]

        # optimizer_grouped_parameters = [
        #     {'params': param_top,
        #      'lr': args.lr * 0.1,
        #      'weight_decay': args.weight_decay},
        #     {'params': param_optimizer,
        #      'lr': args.lr,
        #      'weight_decay': args.weight_decay},
        # ]

        return param_optimizer

    def softXEnt(self, logits, target, weight):
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        return -(target * logprobs * weight).sum() / logits.shape[0]

    def compute_kl(self, logits, target):
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        loss = self.kl_loss(log_prob, target)
        return loss

    def compute_ent(self, prob):
        log_prob = -torch.log(prob)
        # sum_result = 1 - torch.sum(prob * log_prob, dim=1)
        sum_result = torch.sum(prob * log_prob, dim=1)
        weight = torch.zeros(len(sum_result)).cuda()
        weight[sum_result < 0.2] = 1
        # return sum_result.unsqueeze(1)
        return weight.unsqueeze(1)

    def train(self, train_loader, ood_loader, id_loader, val_loader):
        best_valid = 0.
        best_ood = 0.

        vis_feat = []
        lang_feat = []
        gen_vis_feat = []
        gen_lang_feat = []
        feat = []
        ques_list = []
        vqa_vis_feat = []
        vqa_lang_feat = []
        vqa_feat = []
        coco_vis_feat = []
        coco_lang_feat = []
        coco_feat = []
        gen_feat = []
        path = '/home/ygsong/project/D-VQA/LXMERT/snap/vqa_ours_dvqa_best/best_freeze_logits_distill_wo_dvqa_best_nomatch_ver_iamback/'

        # train
        total_num = len(train_loader.dataset)
        # total_num = 0
        for epoch in range(args.epochs):
            self.model.train()
            total_loss = 0
            total_bce_loss = 0
            self_loss = 0
            itm_loss = 0
            distill_loss = 0
            total_self_loss = 0
            train_score_pos = 0
            train_score_neg_q = 0
            train_score_neg_v = 0
            total_q_bce_loss = 0
            total_v_bce_loss = 0
            total_debias_bce_loss = 0
            total_con_loss = 0
            total_itm_loss = 0
            total_l2_loss = 0
            total_rank_loss = 0
            total_dis_loss = 0
            total_ver_loss = 0
            total_feat_dis_loss = 0
            self_sup = epoch >= args.pretrain_epoches
            l2_flag = epoch >= args.pretrain_epoches
            # ranking_flag = epoch >= args.pretrain_epoches
            ranking_flag = False
            # ranking_flag = True
            # feat_gen = False
            feat_gen = True
            # enhance_flag = epoch >= 2
            enhance_flag = False
            # self.model.txt_generator = FeatureGenerator(768)
            # self.model.img_generator = FeatureGenerator(768)
            # self.model = self.model.cuda()
            timer = utils.TimeMeter()
            # for i, (feats, boxes, sent, target, ques_id) in tqdm(enumerate(train_loader)):
            for i, (feats, boxes, sent, target, ques_id, cap) in tqdm(enumerate(train_loader)):

                self.optim.zero_grad()
                batch_size = feats.size(0)
                match_label = torch.ones(batch_size, dtype=torch.long).cuda()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                # out_dict = self.model(feats, boxes, list(sent), self_sup=self_sup, feat_gen=feat_gen, ranking=ranking_flag, enhanced=enhance_flag)
                out_dict = self.model(feats, boxes, list(sent), list(cap), self_sup=self_sup, feat_gen=feat_gen, ranking=ranking_flag, enhanced=enhance_flag)

                ques_list.extend(ques_id.cpu())
                # vis_feat.append(out_dict['vis_feat'].detach().cpu().numpy())
                # lang_feat.append(out_dict['lang_feat'].detach().cpu().numpy())
                gen_vis_feat.append(out_dict['gen_vis_feat'].detach().cpu().numpy())
                gen_lang_feat.append(out_dict['gen_lang_feat'].detach().cpu().numpy())
                # feat.append(out_dict['feat'].detach().cpu().numpy())

                vqa_vis_feat.append(out_dict['vqa_vis_feat'].detach().cpu().numpy())
                vqa_lang_feat.append(out_dict['vqa_lang_feat'].detach().cpu().numpy())
                vqa_feat.append(out_dict['vqa_feat'].detach().cpu().numpy())

                coco_vis_feat.append(out_dict['coco_vis_feat'].detach().cpu().numpy())
                coco_lang_feat.append(out_dict['coco_lang_feat'].detach().cpu().numpy())
                coco_feat.append(out_dict['coco_feat'].detach().cpu().numpy())

                gen_feat.append(out_dict['gen_feat'].detach().cpu().numpy())
                # gen_feat_vis2lang.append(out_dict['gen_feat_vis2lang'].detach().cpu().numpy())

                if len(vqa_vis_feat) > 200:
                    # vis_feat = np.concatenate(vis_feat, axis=0)
                    # lang_feat = np.concatenate(lang_feat, axis=0)
                    # np.save(path + 'vis_feat.npy', vis_feat)
                    # np.save(path + 'lang_feat.npy', lang_feat)
                    gen_vis_feat = np.concatenate(gen_vis_feat, axis=0)
                    gen_lang_feat = np.concatenate(gen_lang_feat, axis=0)
                    np.save(path + 'gen_vis_feat.npy', gen_vis_feat)
                    np.save(path + 'gen_lang_feat.npy', gen_lang_feat)
                    # feat = np.concatenate(feat, axis=0)
                    # np.save(path + 'feat.npy', feat)
                    # np.save(path + 'ques_list.npy', np.array(ques_list))

                    vqa_vis_feat = np.concatenate(vqa_vis_feat, axis=0)
                    vqa_lang_feat = np.concatenate(vqa_lang_feat, axis=0)
                    vqa_feat = np.concatenate(vqa_feat, axis=0)
                    np.save(path + 'vqa_vis_feat.npy', vqa_vis_feat)
                    np.save(path + 'vqa_lang_feat.npy', vqa_lang_feat)     
                    np.save(path + 'vqa_feat.npy', vqa_feat)

                    coco_vis_feat = np.concatenate(coco_vis_feat, axis=0)
                    coco_lang_feat = np.concatenate(coco_lang_feat, axis=0)
                    coco_feat = np.concatenate(coco_feat, axis=0)
                    np.save(path + 'coco_vis_feat.npy', coco_vis_feat)
                    np.save(path + 'coco_lang_feat.npy', coco_lang_feat)     
                    np.save(path + 'coco_feat.npy', coco_feat)

                    gen_feat = np.concatenate(gen_feat, axis=0)
                    # gen_feat_vis2lang = np.concatenate(gen_feat_vis2lang, axis=0)
                    np.save(path + 'gen_feat.npy', gen_feat)
                    # np.save(path + 'gen_feat_vis2lang.npy', gen_feat_vis2lang)     


                    ood_score, upper_bound = evaluate(self.model, ood_loader, path=path)

                    print('saving features done...')
                    sys.exit(0)

                # base VQA model
                bce_loss = instance_bce_with_logits(
                    out_dict['logits'], target, reduction='mean')
                # # only_q
                # bce_loss_q = instance_bce_with_logits(
                #     out_dict['q_logits'], target, reduction='mean')
                # # only_v
                # bce_loss_v = instance_bce_with_logits(
                #     out_dict['v_logits'], target, reduction='mean')
                # # debias
                # bce_loss_debias = instance_bce_with_logits(
                #     out_dict['debias_logits'], target, reduction='mean')
                # con_loss = self.contrastive_loss(
                #     out_dict['fea'], out_dict['pos_fea'], out_dict['neg_fea'])
                #
                # loss = bce_loss + bce_loss_q + bce_loss_debias + \
                #     bce_loss_v + con_loss

                loss = bce_loss

                # if l2_flag:
                #     gt_index = torch.max(target, 1)[1].data
                #     weight_matrix = torch.mm(self.model.logit_fc[3].weight, self.model.logit_fc[0].weight)
                #     l2_loss = torch.mean(weight_matrix[gt_index] ** 2)
                #     loss = loss + l2_loss

                if feat_gen:
                    # ITM matching
                    img_itm_loss = F.cross_entropy(out_dict['img_matched_scores'], match_label, reduction='mean')
                    # img_itm_loss = 0
                    txt_itm_loss = F.cross_entropy(out_dict['txt_matched_scores'], match_label, reduction='mean')
                    itm_loss = (img_itm_loss + txt_itm_loss) * 1.0
                    loss = loss + itm_loss
                    t = 0.5
                    # vis_label = F.softmax(out_dict['img_matched_scores'].detach() / t, dim=1)
                    gen_label = F.softmax(out_dict['gen_matched_scores'].detach() / t, dim=1)

                    # vis_itm_loss = F.cross_entropy(out_dict['vis_itm_scores'], match_label, reduction='mean')
                    # vis_dis_loss = self.softXEnt(out_dict['vis_itm_scores'], vis_label, out_dict['bias_score'])
                    # vis_dis_loss = self.softXEnt(out_dict['vis_itm_scores'], vis_label, self.compute_ent(vis_label))
                    # vis_dis_loss = self.compute_kl(out_dict['vis_itm_scores'], vis_label)

                    # vis_dis_loss = self.softXEnt(out_dict['vis_itm_scores'], vis_label, weight=1)
                    gen_dis_loss = self.softXEnt(out_dict['gen_itm_scores'], gen_label, weight=1)

                    # img_itm_loss = 0
                    # lang_itm_loss = F.cross_entropy(out_dict['lang_itm_scores'], match_label, reduction='mean')
                    # lang_dis_loss = self.softXEnt(out_dict['lang_itm_scores'], lang_label, out_dict['bias_score'])
                    # lang_dis_loss = self.softXEnt(out_dict['lang_itm_scores'], lang_label, self.compute_ent(lang_label))
                    # lang_dis_loss = self.compute_kl(out_dict['lang_itm_scores'], lang_label)

                    # lang_dis_loss = self.softXEnt(out_dict['lang_itm_scores'], lang_label, weight=1)

                    # lang_itm_loss = 0
                    # dis_loss = (vis_dis_loss + lang_dis_loss) * 1.0
                    dis_loss = gen_dis_loss * 1.0
                    loss = loss + dis_loss

                    # ver_loss = out_dict['txt_ver_loss'] + out_dict['vis_ver_loss']
                    # loss = loss + ver_loss * 1.0

                    # feat_dis_loss = 0.05 * out_dict['feat_distill']
                    # loss = loss + feat_dis_loss

                #
                #     # rank_loss = self.ranking_loss(out_dict['fused_gen'], out_dict['ori_multimodal'])
                if ranking_flag:
                    # print('Ranking Loss')
                    rank_loss = self.ranking_loss(out_dict['lang_gen'], out_dict['vis_ori']) + self.ranking_loss(out_dict['vis_gen'], out_dict['lang_ori'])
                    loss = loss + rank_loss

                # if self_sup:
                #     self_loss_q = compute_self_loss(out_dict['logit_neg_q'], target)
                #     self_loss_v = compute_self_loss(out_dict['logit_neg_v'], target)
                #
                #     self_loss = self_loss_v + args.self_loss_q * self_loss_q
                #     loss = loss + args.self_loss_weight * self_loss

                total_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score_pos = compute_score_with_logits(out_dict['logits'], target.data).sum()
                train_score_pos += score_pos.item()
                total_loss += loss.item() * batch_size
                total_bce_loss += bce_loss.item() * batch_size
                # total_con_loss += con_loss.item() * batch_size
                # total_q_bce_loss += bce_loss_q.item() * batch_size
                # total_debias_bce_loss += bce_loss_debias.item() * batch_size
                # total_v_bce_loss += bce_loss_v.item() * batch_size

                # if l2_flag:
                #     total_l2_loss += l2_loss.item() * batch_size

                if feat_gen:
                    total_itm_loss += itm_loss.item() * batch_size
                    # total_ver_loss += ver_loss.item() * batch_size
                    # total_dis_loss += dis_loss.item() * batch_size
                    # total_feat_dis_loss += feat_dis_loss.item() * batch_size

                if ranking_flag:
                    total_rank_loss += rank_loss.item() * batch_size
                    # total_matched_loss += matched_all_loss.item() * batch_size

                # if self_sup:
                #     score_neg_q = compute_score_with_logits(
                #     out_dict['logit_neg_q'], target.data).sum()
                #     score_neg_v = compute_score_with_logits(
                #         out_dict['logit_neg_v'], target.data).sum()
                #     total_self_loss += self_loss.item() * batch_size
                #     train_score_neg_q += score_neg_q.item()
                #     train_score_neg_v += score_neg_v.item()
                timer.update()
                if i and i % 100 == 0:
                    # log_str = 'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, q_bce_loss: %.6f, v_bce_loss: %.6f, debias_bce_loss: %.6f, constrast_loss: %.6f, self_loss: %.6f, neg_train_q_acc: %.6f, neg_train_v_acc: %.6f, pos_train_acc: %.6f, itm_loss: %.6f' %(i, len(train_loader), total_loss / i,
                    #  total_bce_loss /i, total_q_bce_loss / i, total_v_bce_loss / i,
                    #  total_debias_bce_loss /
                    #  i, total_con_loss /
                    #  i, total_self_loss / i,
                    #  100 * train_score_neg_q / i, 100 * train_score_neg_v / i, 100 * train_score_pos / i, total_itm_loss / i)

                    log_str = 'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, q_bce_loss: %.6f, v_bce_loss: %.6f, debias_bce_loss: %.6f, constrast_loss: %.6f, self_loss: %.6f, neg_train_q_acc: %.6f, neg_train_v_acc: %.6f, pos_train_acc: %.6f, itm_loss: %.6f, l2_loss: %.6f, rank_loss: %.6f, dis_loss: %.6f, feat_distill: %.6f, ver_loss: %.6f' % (
                    i, len(train_loader), total_loss / total_num,
                    total_bce_loss / total_num, total_q_bce_loss / total_num, total_v_bce_loss / total_num,
                    total_debias_bce_loss /
                    total_num, total_con_loss /
                    total_num, total_self_loss / total_num,
                    100 * train_score_neg_q / total_num, 100 * train_score_neg_v / total_num, 100 * train_score_pos / total_num,
                    total_itm_loss / total_num, total_l2_loss/total_num, total_rank_loss/total_num, total_dis_loss/total_num, total_feat_dis_loss/total_num, total_ver_loss/total_num)

                    # log_str = 'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, itm_loss: %.6f' % (
                    #     i, len(train_loader), total_loss / total_num, total_bce_loss / total_num,
                    #     total_itm_loss / total_num)

                    print(log_str)
                    print("{:.3f} seconds/batch".format(1.0 / timer.avg))
                    # print(self.model.img_memo.data)
                    # print(self.model.txt_memo.data)

            self.save("LAST")
            
            # if self.valid_tuple is not None:  # Do Validation
            self.model.train(False)
            val_score, upper_bound = evaluate(self.model, val_loader)
            ood_score, upper_bound = evaluate(self.model, ood_loader)
            id_score, upper_bound = evaluate(self.model, id_loader)
            self.model.train(True)
            if val_score > best_valid:
                best_valid = val_score
                # self.best_model_state = deepcopy(self.model.state_dict())
                self.save("BEST")
            if ood_score > best_ood:
                best_ood = ood_score
            log_str = "Epoch %d: Valid %0.2f\n" % (epoch, val_score * 100.) + \
                      "Epoch %d: OOD %0.2f\n" % (epoch, ood_score * 100.) + \
                      "Epoch %d: Best OOD %0.2f\n" % (epoch, best_ood * 100.) + \
                      "Epoch %d: ID %0.2f\n\n" % (epoch, id_score * 100.)

            print(log_str)

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()
            
            if epoch == 9:
                break

        return best_valid

    def save(self, name):
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def test(self):
        print('Start Testing...')
        self.model.load_state_dict(self.best_model_state)
        self.model.eval()

        logits, qIds = get_logits(self.model, self.test_loader)
        results = make_json(logits, qIds, self.test_loader)
        model_label = 'best'  # opt.label

        if False:  # opt.logits:
            utils_1.create_dir('logits/' + model_label)
            torch.save(logits, 'logits/' + model_label + '/logits%d.pth' % opt.s_epoch)

        utils_1.create_dir(self.output + "/result/")
        if 0 <= args.s_epoch:
            model_label += '_epoch%d' % 0

        with open(self.output + '/result/test_%s.json' \
                  % (model_label), 'w') as f:
            json.dump(results, f)

        anno_path = osp.join(args.dataroot, 'cache/%s_target_count.pth' % 'test')
        annotations = torch.load(anno_path)
        json_path = self.output + '/result/test_%s.json' % (model_label)
        compute_score(annotations, json_path)


if __name__ == "__main__":
    vqa = VQA()
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)
    vqa.train(vqa.train_loader, vqa.ood_loader, vqa.id_loader, vqa.val_loader)
    # vqa.test()
