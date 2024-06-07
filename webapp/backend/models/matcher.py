import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcherHOI(nn.Module):

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_hoi_class: float = 1):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_hoi_class = cost_hoi_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        # 以 2 100为例
        bs, num_queries = outputs['pred_sub_boxes'].shape[:2]
        if 'pred_hoi_logits' in outputs.keys():
            out_hoi_prob = outputs['pred_hoi_logits'].flatten(0, 1).sigmoid()  #(2 600)
            tgt_hoi_labels = torch.cat([v['hoi_labels'] for v in targets]) #以5个gt为例 (5 600)
            tgt_hoi_labels_permute = tgt_hoi_labels.permute(1, 0) #(600 5)
            cost_hoi_class = -(out_hoi_prob.matmul(tgt_hoi_labels_permute) / \
                               (tgt_hoi_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                               (1 - out_hoi_prob).matmul(1 - tgt_hoi_labels_permute) / \
                               ((1 - tgt_hoi_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2   #Ha
            cost_hoi_class = self.cost_hoi_class * cost_hoi_class #(200 5)
        else:

            out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
            tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
            tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
            cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                                (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                                ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

            cost_hoi_class = self.cost_verb_class * cost_verb_class
        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])  #(5)
        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)  #(200 81)
        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]   #(200 5)  #Hc
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)  #(200 4)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets]) #(5 4)
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)  #(200 5) L1 Hb
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)  #Hu
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]  #(200 5)

        C = self.cost_hoi_class * cost_hoi_class + self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou + self.cost_obj_class * cost_obj_class  #(200 5)

        C = C.view(bs, num_queries, -1).cpu() #(2 100 5)

        sizes = [len(v['sub_boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] #[([],[]),([],[])]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                               cost_hoi_class=args.set_cost_hoi)
