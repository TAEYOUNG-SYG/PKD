# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
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


class GQA:
    def __init__(self):
        print('loading training dataset...')
        self.train_tuple = get_tuple(
            '/hdd/ygsong/D-VQA/gqa/questions/ood/processed/train_balanced_questions.json', bs=args.batch_size, shuffle=True, drop_last=True
        )
        print('loading validation dataset...')
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

        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
            # self.model.freeze_lxrt_encoder.load(args.load_lxmert)
        
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            param_ft, param_lp = self.get_optimized_para()
            # self.optim = BertAdam(params,
            #                       lr=args.lr,
            #                       warmup=0.1,
            #                       t_total=t_total)
            self.optim_ft = BertAdam(list(self.model.parameters()),
                                      lr=args.lr,
                                      warmup=0.1,
                                      t_total=int(batch_per_epoch * (args.epochs + 2)))
            self.optim_lp = BertAdam(param_lp,
                                    lr=args.lr,
                                    warmup=0.1,
                                    t_total=int(batch_per_epoch * 2))
            # self.optim = BertAdam(list(self.model.parameters()),
            #                       lr=args.lr,
            #                       warmup=0.1,
            #                       t_total=t_total)
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
    # def get_optimized_para(self):

    #     # param_optimizer = [p for n, p in self.model.named_parameters()
    #     #                    if 'logit_fc' in n]
    #     param_optimizer = [p for n, p in self.model.named_parameters() 
    #                         if 'freeze_lxrt_encoder' not in n]

    #     return param_optimizer

    def train(self, train_tuple, eval_tuple, test_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        best_test = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                # if epoch < 1:
                #     self.optim_lp.step()
                # else:
                #     self.optim_ft.step()
                self.optim_lp.zero_grad()
                self.optim_ft.zero_grad()
                # self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                # logit = out_dict['logits']
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                if epoch < 1:
                    self.optim_lp.step()
                else:
                    self.optim_ft.step()
                # self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

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

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
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