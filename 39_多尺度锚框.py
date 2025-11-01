import torch
from d2l import torch as d2l
img = d2l.plt.imread('D:\pycharm\DL_1\catdog.jpg')
h, w = img.shape[:2]
def display_anchors(fmap_w,fmap_h,s):
    d2l.set_figsize()
    fmap=torch.zeros((1,10,fmap_h,fmap_w))
    anchors=d2l.multibox_prior(fmap,sizes=s,ratios=[1,2,0.5])
    bbox_scale=torch.tensor((w,h,w,h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0]*bbox_scale)
    d2l.plt.show()
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])

display_anchors(fmap_w=2, fmap_h=2, s=[0.4])

display_anchors(fmap_w=1, fmap_h=1, s=[0.8])