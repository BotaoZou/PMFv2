import numpy as np
import torch
from torch.utils.data import Dataset
from pc_processor.dataset.preprocess import augmentor
from torchvision import transforms
from scipy.spatial.transform import Rotation as R

class PerspectiveViewLoaderV2(Dataset):
    def __init__(self, dataset, config, data_len=-1, is_train=True, img_aug=False,
                 return_uproj=False):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train
        self.img_aug = img_aug
        self.data_len = data_len
        self.pv_config = self.config["PVconfig"]

        self.fov_left = self.pv_config["fov_left"] / 180.0 * np.pi
        self.fov_right = self.pv_config["fov_right"] / 180.0 * np.pi
        self.rotation_angle = self.pv_config["rotation_angle"]

        if not self.is_train:
            self.img_aug = False
        


        if self.img_aug:
            self.img_jitter = transforms.ColorJitter(
                *self.pv_config["img_jitter"])
        else:
            self.img_jitter = None

        if self.is_train:
            self.aug_ops = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomRotation(15),
                transforms.RandomCrop(
                    size=(self.pv_config["proj_ht"],
                          self.pv_config["proj_wt"])),
            ])
        else:
            self.aug_ops = None
            # self.aug_ops = transforms.Compose([
            #     transforms.CenterCrop((self.pv_config["proj_h"],
            #                            self.pv_config["proj_w"]))
            # ])
        self.return_uproj = return_uproj

    def __getitem__(self, index):
        # feature: range, x, y, z, i, rgb
        
        # get image feature
        image = self.dataset.loadImage(index)
        if self.img_aug:
            image = self.img_jitter(image)
        image = np.array(image)
        image = image.astype(np.float32) / 255.0

        # get point cloud and label
        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)
        seq_id, _ = self.dataset.parsePathInfoByIndex(index)
        
        proj_tensor_list = []
        xy_index_tensor_list = []
        depth_tensor_list = []
        keep_mask_tensor_list = []
        
        if self.is_train:
            max_h, max_w = self.pv_config["proj_ht"], self.pv_config["proj_wt"] # 0, 0
        else:
            max_h, max_w = self.pv_config["proj_h"], self.pv_config["proj_w"]

        for angle in self.rotation_angle:
            rot_pointcloud = pointcloud.copy()
            rot_matrix = R.from_euler("zyx", [angle, 0, 0], degrees=True).as_matrix()
            rot_pointcloud[:, :3] = np.matmul(rot_pointcloud[:, :3], rot_matrix.T)
            yaw = -np.arctan2(rot_pointcloud[:, 1], rot_pointcloud[:, 0])
            fov_keep_mask = (yaw > self.fov_left) * (yaw <= self.fov_right)
            rot_pointcloud = rot_pointcloud[fov_keep_mask]

            xy_index, keep_mask = self.dataset.mapLidar2Camera(
                seq_id, rot_pointcloud[:, :3])
            # print(xy_index.shape, keep_mask.shape)
            # print(keep_mask.sum())

            x_data = xy_index[:, 0].astype(np.int32)
            y_data = xy_index[:, 1].astype(np.int32)
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            
            h, w = x_max-x_min+1, y_max-y_min+1
            if h > max_h:
                max_h = h 
            if w > max_w:
                max_w = w

            proj_xyzi = np.zeros(
                (h, w, rot_pointcloud.shape[1]), dtype=np.float32)
            proj_xyzi[x_data-x_min, y_data-y_min] = rot_pointcloud[keep_mask]
            
            proj_depth = np.zeros((h, w), dtype=np.float32)
            # compute image view pointcloud feature
            depth = np.linalg.norm(rot_pointcloud[:, :3], 2, axis=1)
            proj_depth[x_data-x_max, y_data-y_min] = depth[keep_mask]
            
            proj_label = np.zeros((h, w), dtype=np.int32)
            fov_sem_label = sem_label[fov_keep_mask]
            proj_label[x_data-x_min, y_data-y_min] = self.dataset.labelMapping(fov_sem_label[keep_mask])
            

            proj_mask = np.zeros((h, w), dtype=np.int32)
            proj_mask[x_data-x_min, y_data-y_min] = 1

            proj_rgb = np.zeros((h, w, 3), dtype=np.float32)
            if angle == 0:
                print(image.shape, x_min, x_max)
                if x_min >= 0:
                    proj_rgb[:image.shape[0]-x_min, -y_min:image.shape[1]-y_min] = image[x_min:]
                else:
                    proj_rgb[-x_min:image.shape[0]-x_min, -y_min:image.shape[1]-y_min] = image


            # convert data to tensor 
            proj_tensor = torch.cat(
                (
                    torch.from_numpy(proj_depth).unsqueeze(0),
                    torch.from_numpy(proj_xyzi).permute(2, 0, 1),
                    torch.from_numpy(proj_rgb).permute(2, 0, 1),
                    torch.from_numpy(proj_mask).float().unsqueeze(0),
                    torch.from_numpy(proj_label).float().unsqueeze(0)
                ), dim=0
            )
            xy_index_tensor = torch.from_numpy(xy_index.copy())
            depth_tensor = torch.from_numpy(depth[keep_mask])
            keep_mask_tensor = torch.from_numpy(fov_keep_mask)

            proj_tensor_list.append(proj_tensor)
            xy_index_tensor_list.append(xy_index_tensor)
            depth_tensor_list.append(depth_tensor)
            keep_mask_tensor_list.append(keep_mask_tensor)
            
        if self.return_uproj:
            return proj_tensor_list, xy_index_tensor_list, depth_tensor_list, keep_mask_tensor_list
            # return proj_tensor[:8], proj_tensor[8], proj_tensor[9], torch.from_numpy(x_data), torch.from_numpy(
            #     y_data), torch.from_numpy(depth)
        else:
            pad_proj_tensor_list = []
            for tensor in proj_tensor_list:
                # print(tensor.size())
                h_pad = max_h - tensor.size(1)
                w_pad_left = (max_w - tensor.size(2)) // 2
                w_pad_right = max_w - tensor.size(2) - w_pad_left
                # print(h_pad, w_pad_left, w_pad_right)
                pad_op = transforms.Pad((w_pad_left, 0, w_pad_right, h_pad))
                tensor = pad_op(tensor)
                pad_proj_tensor_list.append(tensor.unsqueeze(0))
            pad_proj_tensor = torch.cat(pad_proj_tensor_list, dim=0)
            
            # tensor augmentation
            if self.aug_ops is not None:
                pad_proj_tensor = self.aug_ops(pad_proj_tensor)
            return pad_proj_tensor

    def __len__(self):
        if self.data_len > 0 and self.data_len < len(self.dataset):
            return self.data_len
        else:
            return len(self.dataset)
