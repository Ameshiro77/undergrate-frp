import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.hico_eval_triplet import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif hasattr(criterion, 'loss_hoi_labels'):
        metric_logger.add_meter('hoi_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    _cnt=0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        outputs = model(samples,targets)
        # print(targets)  #model返回一个字典 loss也是一个loss字典
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # print(loss_value)
        # sys.exit()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # for name, parms in model.named_parameters():
        #     if parms.grad is None:
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        #
        # import pdb
        # pdb.set_trace()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif hasattr(criterion, 'loss_hoi_labels'):
            metric_logger.update(hoi_class_error=loss_dict_reduced['hoi_class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    preds = []
    gts = []
    indices = []
    counter = 0
    # samples targets outputs | preds 
    stats_list = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
    
        #print("***\neval的样本和target情况..\n,",type(samples),type(targets),samples.tensors.shape,len(targets))
        #<class 'util.misc.NestedTensor'> <class 'tuple'> torch.Size([4, 3, 1201, 1204]) 4 (bs=4)
        #print(targets[0])  #是一个字典，‘origin_size，size，filename，boxes(2维),labels,id,sd_puts(3d),clip_inputs(3d),hois(2d?)'
        _targets = [{k: v.to(device) for k, v in t.items() if (k != 'filename' and k != 'id') } for t in targets]
  
        with torch.no_grad():
            outputs = model(samples,_targets, is_training=False)
            #print(outputs.keys()) ['pred_hoi_logits', 'pred_obj_logits', 'pred_sub_boxes', 'pred_obj_boxes', 'aux_outputs']
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)  #得到list ↓
        #print(type(results),len(results),results[0]) bs=4,<class 'list'> 4 'labels,boxes,hoi_scores,obj_scores,sub_ids,obj_ids'，记作键①
        #print(list(itertools.chain.from_iterable([results]))) 也是一个字典的列表；和上面键①一样↑
    
        # ===== preds ===== #
        #results_wout_tensor = [{k:v.to(device) for k,v in t.items()} for t in results]
        #preds.extend(list(itertools.chain.from_iterable([results_wout_tensor])))  #由于目前只有一张显卡，所以采用这种写法 且原先是[result]
        preds.extend(list(itertools.chain.from_iterable([results])))
        #print(len(preds),preds) #len不断上升；preds是字典列表，和上面键①一样
  
        # ==== targets ==== #
        # For avoiding a runtime error, the copy is used
        #print("之中,已分配存储：", torch.cuda.memory_allocated() / (1024.0 * 1024.0), "MB")
        #targets_wout_tensor = [{k:v.to(device) for k,v in t.items()} for t in copy.deepcopy(targets)]
        #gts.extend(list(itertools.chain.from_iterable([targets_wout_tensor])))  #原先是utils.all_gather(copy.deepcopy(targets))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        #print("之后,已分配存储：", torch.cuda.memory_allocated() / (1024.0 * 1024.0), "MB")
        
        # == 得到了 preds 和 targets  分不同批次测
        
        print("===",len(preds),"===")
        if len(preds) >= 500:
            # counter += 1
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            img_ids = [img_gts['id'] for img_gts in gts]
            _, indices = np.unique(img_ids, return_index=True)
            preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
            gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

            if dataset_file == 'hico':
                evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                        data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
            elif dataset_file == 'vcoco':
                evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)
            print("第"+str(len(stats_list)+1)+"批:")
            stats = evaluator.evaluate()
            stats_list.append(stats)
            del preds,targets
            torch.cuda.empty_cache()
            
    # === 对于stats列表算总map === #
    #  return_dict = {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}
    sum_m_max_recall = sum_m_ap = sum_m_ap_non_rare = sum_m_ap_rare = 0
    sum_m_ap = sum(stat["mAP"] for stat in stats_list)
    sum_m_ap_rare = sum(stat["mAP rare"] for stat in stats_list)
    sum_m_ap_non_rare = sum(stat["mAP non-rare"] for stat in stats_list)
    sum_m_max_recall = sum(stat["mean max recall"] for stat in stats_list)
    bs = len(stats_list)
    print('mAP full: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(sum_m_ap/bs, sum_m_ap_rare/bs, sum_m_ap_non_rare/bs,
                                                                                    sum_m_max_recall/bs))
    avg_stats = {}
    avg_stats["mAP"] = sum_m_ap/bs
    return stats