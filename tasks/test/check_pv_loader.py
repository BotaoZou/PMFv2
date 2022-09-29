import sys 
sys.path.insert(0, "../../")

import pc_processor
import tensorboardX

dataset = pc_processor.dataset.semantic_kitti.SemanticKitti(
    root="/mnt/dataset/semantic-kitti/sequences/",
    sequences=[0,1,2,3,4,5,6,9,10],
    config_path="/home/iceory/project/PMFv2/pc_processor/dataset/semantic_kitti/semantic-kitti.yaml", 
    has_image=True, 
    has_pcd=True,
    has_label=True
)

loader_config = {
    "PVconfig": {
        "img_jitter" : [ 0.4, 0.4, 0.4 ],
        "proj_h": 768,
        "proj_w": 1952,
        "proj_ht": 512,
        "proj_wt": 1536,
        "rotation_angle": [0, 90, 180, 270],
        "fov_left": -45.,
        "fov_right": 45.,
    }
}
loader = pc_processor.dataset.PerspectiveViewLoaderV2(
    dataset, config=loader_config, data_len=-1, is_train=True,
    img_aug=True, return_uproj=False
)

# data = loader[2024]

max_h, max_w = 0, 0
for i in range(0, len(loader)):
    
    data = loader[i]
    n,c,h,w = data.size()
    if h > max_h:
        max_h = h
    if w > max_w:
        max_w = w
    print("{:06d}/{:06d} [{},{}]".format(len(loader), i, max_h, max_w))
# print(max_h, max_w)

# for i in range(100):
# data = loader[0]
# tensorboard = tensorboardX.SummaryWriter(logdir="./")
# for v in range(4):
#     tensorboard.add_image("v{}-mask".format(v), data[v:v+1, 8], 1)
#     tensorboard.add_image("v{}-r".format(v), data[v:v+1, 5], 1)

    # for c in range(10):
    #     tensorboard.add_image("v{}c{}".format(v, c), data[v, c:c+1], 1)
    # print(i, data.size())
