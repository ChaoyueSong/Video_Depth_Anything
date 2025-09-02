import numpy as np
import cv2


# class Resize(object):
#     """Resize sample to given size (width, height).
#     """

#     def __init__(
#         self,
#         width,
#         height,
#         resize_target=True,
#         keep_aspect_ratio=False,
#         ensure_multiple_of=1,
#         resize_method="lower_bound",
#         image_interpolation_method=cv2.INTER_AREA,
#     ):
#         """Init.

#         Args:
#             width (int): desired output width
#             height (int): desired output height
#             resize_target (bool, optional):
#                 True: Resize the full sample (image, mask, target).
#                 False: Resize image only.
#                 Defaults to True.
#             keep_aspect_ratio (bool, optional):
#                 True: Keep the aspect ratio of the input sample.
#                 Output sample might not have the given width and height, and
#                 resize behaviour depends on the parameter 'resize_method'.
#                 Defaults to False.
#             ensure_multiple_of (int, optional):
#                 Output width and height is constrained to be multiple of this parameter.
#                 Defaults to 1.
#             resize_method (str, optional):
#                 "lower_bound": Output will be at least as large as the given size.
#                 "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
#                 "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
#                 Defaults to "lower_bound".
#         """
#         self.__width = width
#         self.__height = height

#         self.__resize_target = resize_target
#         self.__keep_aspect_ratio = keep_aspect_ratio
#         self.__multiple_of = ensure_multiple_of
#         self.__resize_method = resize_method
#         self.__image_interpolation_method = image_interpolation_method

#     def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
#         y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

#         if max_val is not None and y > max_val:
#             y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

#         if y < min_val:
#             y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

#         return y

#     def get_size(self, width, height):
#         # determine new height and width
#         scale_height = self.__height / height
#         scale_width = self.__width / width

#         if self.__keep_aspect_ratio:
#             if self.__resize_method == "lower_bound":
#                 # scale such that output size is lower bound
#                 if scale_width > scale_height:
#                     # fit width
#                     scale_height = scale_width
#                 else:
#                     # fit height
#                     scale_width = scale_height
#             elif self.__resize_method == "upper_bound":
#                 # scale such that output size is upper bound
#                 if scale_width < scale_height:
#                     # fit width
#                     scale_height = scale_width
#                 else:
#                     # fit height
#                     scale_width = scale_height
#             elif self.__resize_method == "minimal":
#                 # scale as least as possbile
#                 if abs(1 - scale_width) < abs(1 - scale_height):
#                     # fit width
#                     scale_height = scale_width
#                 else:
#                     # fit height
#                     scale_width = scale_height
#             else:
#                 raise ValueError(f"resize_method {self.__resize_method} not implemented")

#         if self.__resize_method == "lower_bound":
#             new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
#             new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
#         elif self.__resize_method == "upper_bound":
#             new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
#             new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
#         elif self.__resize_method == "minimal":
#             new_height = self.constrain_to_multiple_of(scale_height * height)
#             new_width = self.constrain_to_multiple_of(scale_width * width)
#         else:
#             raise ValueError(f"resize_method {self.__resize_method} not implemented")

#         return (new_width, new_height)

#     def __call__(self, sample):
#         width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        
#         # resize sample
#         sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

#         if self.__resize_target:
#             if "depth" in sample:
#                 sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)
                
#             if "mask" in sample:
#                 sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        
#         return sample


# class NormalizeImage(object):
#     """Normlize image by given mean and std.
#     """

#     def __init__(self, mean, std):
#         self.__mean = mean
#         self.__std = std

#     def __call__(self, sample):
#         sample["image"] = (sample["image"] - self.__mean) / self.__std

#         return sample


# class PrepareForNet(object):
#     """Prepare sample for usage as network input.
#     """

#     def __init__(self):
#         pass

#     def __call__(self, sample):
#         image = np.transpose(sample["image"], (2, 0, 1))
#         sample["image"] = np.ascontiguousarray(image).astype(np.float32)

#         if "depth" in sample:
#             depth = sample["depth"].astype(np.float32)
#             sample["depth"] = np.ascontiguousarray(depth)
        
#         if "mask" in sample:
#             sample["mask"] = sample["mask"].astype(np.float32)
#             sample["mask"] = np.ascontiguousarray(sample["mask"])
        
#         return sample

import torch
import torch.nn.functional as F
import math

class Resize(object):
    """
    使用 PyTorch 实现的可微分图像缩放。
    支持对 image 及 target（如 depth、mask）进行缩放。
    """
    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        interpolation_mode="area"  # 支持 F.interpolate 的插值模式，如 "area", "bilinear", "bicubic"
    ):
        self.__width = width
        self.__height = height
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__interpolation_mode = interpolation_mode

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        # 计算约束后的尺寸（整数），这里使用 Python 的 math 模块
        y = int(round(x / self.__multiple_of) * self.__multiple_of)
        if max_val is not None and y > max_val:
            y = int(math.floor(x / self.__multiple_of) * self.__multiple_of)
        if y < min_val:
            y = int(math.ceil(x / self.__multiple_of) * self.__multiple_of)
        return y

    def get_size(self, width, height):
        # 计算缩放比例
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # 保证输出尺寸至少达到预设尺寸，选择更大的缩放比例
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # 保证输出尺寸至多达到预设尺寸，选择更小的缩放比例
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # 尽量使缩放比例接近 1
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        # 根据不同的 resize_method 计算最终宽高
        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        # sample["image"] 要求是一个 torch.Tensor，形状为 (H, W, C)
        H, W = sample["image"].shape[0], sample["image"].shape[1]
        target_width, target_height = self.get_size(W, H)
        
        # 将 image 转换为 (N, C, H, W)
        image = sample["image"].permute(2, 0, 1).unsqueeze(0)
        # 调用 F.interpolate 进行缩放
        # 注意：如果插值模式为 'bilinear' 或 'bicubic' 需要设置 align_corners=False，
        # 对于 'area' 或 'nearest' 则不需要 align_corners 参数。
        if self.__interpolation_mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            image_resized = F.interpolate(image, size=(target_height, target_width), mode=self.__interpolation_mode, align_corners=False)
        else:
            image_resized = F.interpolate(image, size=(target_height, target_width), mode=self.__interpolation_mode)
        # 转回 (H, W, C)
        sample["image"] = image_resized.squeeze(0).permute(1, 2, 0)

        if self.__resize_target:
            # 对 depth 进行 resize，如果存在的话
            if "depth" in sample:
                depth = sample["depth"]
                if depth.dim() == 2:
                    depth = depth.unsqueeze(-1)
                depth = depth.permute(2, 0, 1).unsqueeze(0)
                depth_resized = F.interpolate(depth, size=(target_height, target_width), mode="nearest")
                sample["depth"] = depth_resized.squeeze(0).permute(1, 2, 0)
            # 对 mask 进行 resize，如果存在的话
            if "mask" in sample:
                mask = sample["mask"].float()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                mask = mask.permute(2, 0, 1).unsqueeze(0)
                mask_resized = F.interpolate(mask, size=(target_height, target_width), mode="nearest")
                sample["mask"] = mask_resized.squeeze(0).permute(1, 2, 0)
        return sample


class NormalizeImage(object):
    """
    使用 PyTorch 对图像进行归一化处理：
    对每个通道减去均值后除以标准差。注意这里操作保持梯度信息。
    """
    def __init__(self, mean, std):
        # 将均值和标准差转换为 Tensor，并调整为 (1, 1, C) 格式
        self.mean = torch.tensor(mean).view(1, 1, -1)
        self.std = torch.tensor(std).view(1, 1, -1)

    def __call__(self, sample):
        # 确保均值和 std 处在和 image 相同的设备上
        sample["image"] = (sample["image"] - self.mean.to(sample["image"].device)) / self.std.to(sample["image"].device)
        return sample


class PrepareForNet(object):
    """
    调整图像数据格式，使其符合网络输入要求：
    将 image 从 (H, W, C) 转换为 (C, H, W) 并确保为 contiguous FloatTensor 。
    同时对 depth、mask 也执行类似处理（如果存在）。
    """
    def __call__(self, sample):
        sample["image"] = sample["image"].permute(2, 0, 1).contiguous().float()
        if "depth" in sample:
            sample["depth"] = sample["depth"].contiguous().float()
        if "mask" in sample:
            sample["mask"] = sample["mask"].contiguous().float()
        return sample
