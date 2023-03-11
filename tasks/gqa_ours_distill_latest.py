# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model_ours_distill_latest import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

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


class GQA:
    def __init__(self):
        print('loading training dataset...')
        self.train_tuple = get_tuple(
            '/hdd/ygsong/D-VQA/gqa/questions/ood/processed/train_balanced_questions.json', bs=args.batch_size, shuffle=True, drop_last=True
        )
        
        # if args.valid != "":
        valid_bsize = 2048 if args.multiGPU else 512
        print('loading validation dataset...')
        self.valid_tuple = get_tuple(
            '/hdd/ygsong/D-VQA/gqa/questions/ood/processed/val_all.json', bs=valid_bsize,
            shuffle=False, drop_last=False
        )
        print('loading test dataset...')
        self.test_tuple = get_tuple(
            '/hdd/ygsong/D-VQA/gqa/questions/ood/processed/testdev_all.json', bs=valid_bsize,
            shuffle=False, drop_last=False
        )
        # else:
        #     self.valid_tuple = None

        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
            self.model.freeze_lxrt_encoder.load(args.load_lxmert)

        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
            # self.model.lxrt_encoder.multi_gpu()
            # self.model = nn.DataParallel(self.model)

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.ranking_loss = RankingLoss(margin=0.2)
        self.contrastive_loss = Contrastive_loss(tao=1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            # self.optim = BertAdam(list(self.model.parameters()),
            #                       lr=args.lr,
            #                       warmup=0.1,
            #                       t_total=t_total)
            param_ft, param_lp = self.get_optimized_para()
            # params = self.get_optimized_para()
            # self.optim = BertAdam(param_ft,
            #                       lr=args.lr,
            #                       warmup=0.1,
            #                       t_total=t_total)
            self.optim_ft = BertAdam(param_ft,
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=int(batch_per_epoch * (args.epochs - 1)))
            self.optim_lp = BertAdam(param_lp,
                                    lr=args.lr,
                                    warmup=0.1,
                                    t_total=int(batch_per_epoch))
            # self.optim_lp = torch.optim.Adam(param_lp, 0.001, weight_decay=1e-4)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_lp, 10, eta_min=0)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def get_optimized_para(self):
        # param_top = [p for n, p in self.model.named_parameters()
        #              if 'freeze_lxrt_encoder' not in n]

        # param_top = [p for n, p in self.model.named_parameters()
        #                    if 'freeze_lxrt_encoder' not in n or 'itm_output' not in n]

        param_ft = [p for n, p in self.model.named_parameters()
                     if 'freeze_lxrt_encoder' not in n]

        param_lp = [p for n, p in self.model.named_parameters()
                           if 'logit_fc' in n]
        # param_optimizer = [p for n, p in self.model.named_parameters()
        #              if 'freeze_lxrt_encoder' not in n and 'generator' not in n and 'memory' not in n]

        # optimizer_grouped_parameters = [
        #     {'params': param_top,
        #      'lr': args.lr * 0.1,
        #      'weight_decay': args.weight_decay},
        #     {'params': param_optimizer,
        #      'lr': args.lr,
        #      'weight_decay': args.weight_decay},
        # ]

        return param_ft, param_lp

    def softXEnt(self, logits, target, weight):
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        return -(target * logprobs * weight).sum() / logits.shape[0]

    def compute_ent(self, prob):
        log_prob = -torch.log(prob)
        sum_result = 1 - torch.sum(prob * log_prob, dim=1)
        # sum_result = torch.sum(prob * log_prob, dim=1)
        return sum_result.unsqueeze(1)

    def train(self, train_tuple, eval_tuple, test_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        total_num = len(loader.dataset)
        best_valid = 0.
        best_test = 0.
        for epoch in range(args.epochs):
            total_loss = 0
            total_bce_loss = 0
            self_loss = 0
            itm_loss = 0
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
            total_ver_loss = 0
            self_sup = epoch >= args.pretrain_epoches
            ranking_flag = False
            # ranking_flag = True
            feat_gen = True
            # feat_gen = False
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                # self.optim.zero_grad()
                self.optim_lp.zero_grad()
                self.optim_ft.zero_grad()
                batch_size = feats.size(0)
                match_label = torch.ones(batch_size, dtype=torch.long).cuda()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                out_dict = self.model(feats, boxes, sent, self_sup, feat_gen)

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
                #        bce_loss_v + con_loss

                loss = bce_loss

                if feat_gen:
                    # ITM matching
                    img_itm_loss = F.cross_entropy(out_dict['img_matched_scores'], match_label, reduction='mean')
                    txt_itm_loss = F.cross_entropy(out_dict['txt_matched_scores'], match_label, reduction='mean')
                    itm_loss = (img_itm_loss + txt_itm_loss) * 1.0
                    loss = loss + 0.001 * itm_loss
                    t = 0.5
                    # vis_label = F.softmax(out_dict['img_matched_scores'].detach() / t, dim=1)
                    gen_label = F.softmax(out_dict['gen_matched_scores'].detach() / t, dim=1)
                    # vis_label = F.softmax(out_dict['img_matched_scores'].detach(), dim=1)
                    # lang_label = F.softmax(out_dict['txt_matched_scores'].detach(), dim=1)
                    gen_dis_loss = self.softXEnt(out_dict['gen_itm_scores'], gen_label, weight=1)
                    # vis_itm_loss = F.cross_entropy(out_dict['vis_itm_scores'], match_label, reduction='mean')
                    # vis_itm_loss = self.softXEnt(out_dict['vis_itm_scores'], vis_label, weight=1)
                    # vis_itm_loss = self.softXEnt(out_dict['vis_itm_scores'], vis_label, self.compute_ent(vis_label))
                    # img_itm_loss = 0
                    # lang_itm_loss = F.cross_entropy(out_dict['lang_itm_scores'], match_label, reduction='mean')
                    # lang_itm_loss = self.softXEnt(out_dict['lang_itm_scores'], lang_label, weight=1)
                    # lang_itm_loss = self.softXEnt(out_dict['lang_itm_scores'], lang_label, self.compute_ent(lang_label))
                    # lang_itm_loss = 0
                    dis_loss = gen_dis_loss * 0.001
                    loss = loss + dis_loss

                    ver_loss = out_dict['txt_ver_loss'] + out_dict['vis_ver_loss']
                    loss = loss + ver_loss * 0.001

                if ranking_flag:
                    rank_loss = self.ranking_loss(out_dict['lang_gen'], out_dict['vis_ori']) + self.ranking_loss(out_dict['vis_gen'], out_dict['lang_ori'])
                    loss = rank_loss + loss

                # if self_sup:
                #     self_loss_q = compute_self_loss(out_dict['logit_neg_q'], target)
                #     self_loss_v = compute_self_loss(out_dict['logit_neg_v'], target)
                #
                #     self_loss = self_loss_v + args.self_loss_q * self_loss_q
                #     loss = loss + args.self_loss_weight * self_loss

                total_loss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                # self.optim.step()
                if epoch < 1:
                    self.optim_lp.step()
                else:
                    self.optim_ft.step()

                score_pos = compute_score_with_logits(out_dict['logits'], target.data).sum()
                train_score_pos += score_pos.item()
                total_loss += loss.item() * batch_size
                total_bce_loss += bce_loss.item() * batch_size
                # total_con_loss += con_loss.item() * batch_size
                # total_q_bce_loss += bce_loss_q.item() * batch_size
                # total_debias_bce_loss += bce_loss_debias.item() * batch_size
                # total_v_bce_loss += bce_loss_v.item() * batch_size

                if feat_gen:
                    total_itm_loss += itm_loss.item() * batch_size
                    total_ver_loss += ver_loss.item() * batch_size

                if ranking_flag:
                    total_rank_loss += rank_loss.item() * batch_size

                # if self_sup:
                #     score_neg_q = compute_score_with_logits(
                #     out_dict['logit_neg_q'], target.data).sum()
                #     score_neg_v = compute_score_with_logits(
                #         out_dict['logit_neg_v'], target.data).sum()
                #     total_self_loss += self_loss.item() * batch_size
                #     train_score_neg_q += score_neg_q.item()
                #     train_score_neg_v += score_neg_v.item()

                score, label = out_dict['logits'].max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()): 
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                if i and i % 100 == 0:
                    log_str = 'traing: %d/%d, train_loss: %.6f, bce_loss: %.6f, q_bce_loss: %.6f, v_bce_loss: %.6f, debias_bce_loss: %.6f, constrast_loss: %.6f, self_loss: %.6f, neg_train_q_acc: %.6f, neg_train_v_acc: %.6f, pos_train_acc: %.6f, itm_loss: %.6f, l2_loss: %.6f, rank_loss: %.6f' % (
                        i, len(loader), total_loss / total_num,
                        total_bce_loss / total_num, total_q_bce_loss / total_num, total_v_bce_loss / total_num,
                        total_debias_bce_loss /
                        total_num, total_con_loss /
                        total_num, total_self_loss / total_num,
                        100 * train_score_neg_q / total_num, 100 * train_score_neg_v / total_num,
                        100 * train_score_pos / total_num,
                        total_itm_loss / total_num, total_l2_loss / total_num, total_rank_loss / total_num)
                    print(log_str)
            
            log_str_epoch = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)
            
            # if self.valid_tuple is not None:  # Do Validation
            valid_score = self.evaluate(eval_tuple)
            test_score = self.evaluate(test_tuple)
            if valid_score > best_valid:
                best_valid = valid_score
            if test_score > best_test:
                best_test = test_score
                self.save("BEST")

            log_str_epoch += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                        "Epoch %d: Best Valid %0.2f\n" % (epoch, best_valid * 100.) + \
                        "Epoch %d: Test %0.2f\n" % (epoch, test_score * 100.) + \
                        "Epoch %d: Best Test %0.2f\n" % (epoch, best_test * 100.)

            print(log_str_epoch, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str_epoch)
                f.flush()

        # self.scheduler.step()
        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                # logit = self.model(feats, boxes, sent)
                out_dict = self.model(feats, boxes, sent, feat_gen=True)
                # out_dict = self.model(feats, boxes, sent, feat_gen=False)
                score, label = out_dict['logits'].max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        # print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            # print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple, gqa.test_tuple)