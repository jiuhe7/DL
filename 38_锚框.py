import torch
from d2l import torch as d2l
from torchvision.ops import box_area

torch.set_printoptions(2)

# 生成锚框
def multibox_prior(data,sizes,ratios):
    '''data：输入图像数据（张量），形状通常为(批量大小, 通道数, 高度, 宽度).
    '从输入数据的形状中提取图像的高度和宽度（data.shape[-2:]对应最后两个维度，即高度和宽度）。'''
    in_height,in_width=data.shape[-2:]
    device,num_sizes,num_ratios=data.device,len(sizes),len(ratios)
    boxes_per_pixel=(num_sizes+num_ratios-1)
    size_tensor=torch.tensor(sizes,device=device)
    ratio_tensor=torch.tensor(ratios,device=device)

    offset_h,offset_w=0.5,0.5
    steps_h=1.0/in_height
    steps_w=1.0/in_width
    center_h=(torch.arange(in_height,device=device)+offset_h)*steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y,shift_x=torch.meshgrid(center_h,center_w,indexing='ij')
    shift_y,shift_x=shift_y.reshape(-1),shift_x.reshape(-1)

    w=torch.cat((size_tensor*torch.sqrt(ratio_tensor[0]),
                 sizes[0]*torch.sqrt(ratio_tensor[1:])))\
                 *in_height/in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations=torch.stack((-w,-h,w,h)).T.repeat(in_height*in_width,1)/2
    out_grid=torch.stack([shift_x,shift_y,shift_x,shift_y],
               dim=1).repeat_interleave(boxes_per_pixel,dim=0)
    output=out_grid+anchor_manipulations
    return output.unsqueeze(0)

img = d2l.plt.imread('D:\pycharm\DL_1\catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(Y.shape)
boxes = Y.reshape(h, w, 5, 4)
print(boxes[250, 250, 0, :])

def show_bboxes(axes,bboxes,labels=None,colors=None):
    def _make_list(obj,default_values=None):
        if obj is None:
            obj=default_values
        elif not isinstance(obj,(list,tuple)):
            obj=[obj]
        return obj
    labels=_make_list(labels)
    colors=_make_list(colors,['b','g','r','m','c'])
    for i ,bbox in enumerate(bboxes):
        color=colors[i%len(colors)]
        rect=d2l.bbox_to_rect(bbox.detach().numpy(),color)
        axes.add_patch(rect)
        if labels and len(labels)>i:
            text_color='k'  if color=='w'else'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
d2l.set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
d2l.plt.show()

# 交并比（IoU）
def box_iou(boxes1,boxes2):
    box_area=lambda boxes:((boxes[:,2]-boxes2[:,0])*
                           (boxes[:,3]-boxes[:,1]))
    areas1=box_area(boxes1)
    areas2=box_area(boxes2)
    inter_upperlefts=torch.max(boxes1[:,None,:2],boxes2[:,:2])
    inter_lowerrights=torch.min(boxes1[:,None,2:],boxes2[:,2:])
    inters=(inter_lowerrights-inter_upperlefts).clamp(min=0)
    inter_areas=inters[:,:,0]*inters[:,:,1]
    union_areas=areas1[:,None]+areas2-inter_areas
    return inter_areas/union_areas


def assign_anchor_to_bbox(ground_truth,anchors,device,iou_threshod=0.5):
    num_anchors,num_gt_boxes=anchors.shape[0],ground_truth.shape[0]
    jaccard=box_iou(anchors,ground_truth)
    anchors_bbox_map=torch.full((num_anchors,),-1,dtype=torch.long,
                                device=device)
    max_ious,indices=torch.max(jaccard,dim=1)
    anc_i=torch.nonzero((max_ious>=iou_threshod).reshape(-1))
    box_j=indices[max_ious>=iou_threshod]
    anchors_bbox_map[anc_i]=box_j
    col_discard=torch.full((num_anchors),-1)
    row_discard=torch.full((num_gt_boxes),-1)
    for _ in range(num_gt_boxes):
        max_idx=torch.argmax(jaccard)
        box_idx=(max_idx%num_gt_boxes).long()
        anc_idx=(max_idx/num_gt_boxes).long()
        anchors_bbox_map[anc_idx]=box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors,assigned_bb,eps=1e-6):
    c_anc=d2l.box_corner_to_center(anchors)
    c_assigned_bb=d2l.box_corner_to_center(assigned_bb)
    offset_xy=10*(c_assigned_bb[:,:2]-c_anc[:,:2])/c_anc[:,2:]
    offset_wh=5*torch.log(eps+c_assigned_bb[:,2:]/c_anc[:,2:])
    offset=torch.cat([offset_xy,offset_wh],axis=1)
    return offset

#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return torch.stack(out)

anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                      [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                      [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率


fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
print(output)

fig = d2l.plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)




