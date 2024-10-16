"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import os
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset_vqacp_lxmert import Dictionary, VQAFeatureDataset
# from dataset_vqav2_lxmert import Dictionary, VQAFeatureDataset
import utils_1
from model_gen_ours import VQAModel
from src.param import args as opt


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
        logits = model(v, b, list(q), feat_gen=True)
        # logits = model(v, b, list(q), feat_gen=False)
        pred[idx:idx+batch_size,:].copy_(logits['logits'].data)
        qIds[idx:idx+batch_size].copy_(i)
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


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(os.path.join(opt.dataroot, 'dictionary.pkl'))
    opt.ntokens = dictionary.ntoken

    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False)
    # eval_dset = VQAFeatureDataset('val', dictionary, opt.dataroot, opt.img_root, ratio=1.0, adaptive=False)

    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device

    model = VQAModel(2274)
    # model = VQAModel(2410)
    model = model.cuda()

    eval_loader = DataLoader(eval_dset, 1800, shuffle=False, num_workers=1, collate_fn=utils_1.trim_collate)

    def process(args, model, eval_loader):

        print('loading %s' % opt.checkpoint_path)
        model_data = torch.load(opt.checkpoint_path)
        model_data = {key.replace('module.',''): value for key, value in model_data.items()}
        # model.load_state_dict(model_data, strict=False)
        model.load_state_dict(model_data)

        model = nn.DataParallel(model).cuda()

        model.train(False)

        logits, qIds = get_logits(model, eval_loader)
        results = make_json(logits, qIds, eval_loader)
        model_label = 'best'  # opt.label 
        
        if False: # opt.logits:
            utils_1.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % opt.s_epoch)
        
        utils_1.create_dir(opt.output)
        if 0 <= opt.s_epoch:
            model_label += '_epoch%d' % opt.s_epoch

        with open(opt.output+'/test_%s.json' \
            % (model_label), 'w') as f:
            json.dump(results, f)

    process(opt, model, eval_loader)
