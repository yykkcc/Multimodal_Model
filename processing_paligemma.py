from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# 定义默认的图像均值和方差（以 ImageNet 数据集的统计值为参考）
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    将图像 token 插入到文本前面，同时在文本前加上一个 bos_token（开头标记）并在结尾添加换行符。
    假设在 PaLi-Gemma 中，模型需要在文本之前看到固定数量的 <image> 占位符。
    
    Args:
        prefix_prompt (str): 原始的文本prompt
        bos_token (str): 模型的开头token（一般为 <bos>）
        image_seq_len (int): 需要插入的 <image> 占位符个数
        image_token (str): 用于表示图像的特殊token，默认 "<image>"
        
    Returns:
        str: 带有若干 <image> token、一个 bos_token、以及原文本和换行符拼接后的字符串。
    """
    # 这里用字符串乘法重复插入 image_seq_len 次 <image>
    # 并紧跟一个 bos_token，最后加上原本的 prompt + 换行符
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    将图像像素值缩放到 [0, 1] 等区间或其它区间。
    
    Args:
        image (np.ndarray): 图像数组，shape 通常是 [H, W, C] 或 [C, H, W]
        scale (float): 缩放因子 (e.g. 1/255.)
        dtype (np.dtype): 转换后的数据类型（默认 float32）
        
    Returns:
        np.ndarray: 缩放并转换数据类型后的图像数组
    """
    # 先乘上 scale
    rescaled_image = image * scale
    # 转换到指定的浮点类型
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """
    调整图像尺寸的函数，内部使用 PIL.Image 的 resize。
    
    Args:
        image (PIL.Image): 输入的 PIL 图像
        size (Tuple[int, int]): 期望输出尺寸 (height, width)
        resample (Image.Resampling): 插值方式（如 BICUBIC）
        reducing_gap (Optional[int]): PIL 减少分辨率时，可能的额外优化参数
        
    Returns:
        Image.Image: 改变尺寸后的 PIL 图像
    """
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    对图像做标准化处理（减去 mean 并除以 std）。
    
    Args:
        image (np.ndarray): 图像数组，一般是 [H, W, C] 形状
        mean (Union[float, Iterable[float]]): 均值，可为标量或数组
        std (Union[float, Iterable[float]]): 标准差，可为标量或数组
        
    Returns:
        np.ndarray: 标准化后的图像数组
    """
    # 将 mean 和 std 转为与图像相同 dtype 的 np.array
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    # 执行 (image - mean) / std
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    针对一批 PIL.Image 进行一系列预处理操作，包括：
    1. resize -> 2. 转numpy -> 3. rescale -> 4. normalize -> 5. 调整通道顺序 (CHW)
    
    Args:
        images (List[Image.Image]): 输入的一组 PIL 图像
        size (Dict[str, int]): 目标尺寸，形如 (height, width)
        resample (Image.Resampling): 图像插值方式
        rescale_factor (float): 缩放因子
        image_mean (float or List[float]): 归一化所用的均值
        image_std (float or List[float]): 归一化所用的标准差
        
    Returns:
        List[np.ndarray]: 处理后的一组图像，每个图像是 [C, H, W] 形状的 np.ndarray
    """
    # 从 size 中解构出 height, width
    height, width = size[0], size[1]
    
    # 1. 对每张图做 resize
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    
    # 2. 将每张 PIL.Image 转成 numpy 数组，shape 通常是 [H, W, C]
    images = [np.array(image) for image in images]
    
    # 3. 将像素值缩放到 [0,1] 或其它区间（具体范围由 rescale_factor 决定）
    images = [rescale(image, scale=rescale_factor) for image in images]
    
    # 4. 用 (image - mean) / std 做标准化，让图像均值大约为 0，标准差为 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    
    # 5. 将通道维度（channel）移动到前面，变成 [C, H, W]
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images

class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        初始化处理器：
        1. 记录图像序列长度和图像输入尺寸
        2. 为 tokenizer 添加 <image> 特殊token 和 <loc****> / <seg***> 等额外 token
        3. 关闭 BOS/EOS 的自动添加
        """
        super().__init__()

        # 要插入多少个 <image> token
        self.image_seq_length = num_image_tokens 
        
        # 图像输入的宽高
        self.image_size = image_size 

        # 往 tokenizer 里添加额外的 special tokens，用于多模态模型中表示图像占位符
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        # 这些 <loc****> 主要用于目标检测（表示 bounding box 的信息）
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]

        # 这些 <seg***> 主要用于图像分割任务（表示 segmentation mask 的信息）
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]
        tokenizer.add_tokens(EXTRA_TOKENS)

        # 将 <image> token 转成对应的 token id，以后在模型推理时可能需要用到
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        当我们调用 PaliGemmaProcessor(...) 时，这个 __call__ 方法会被执行。
        它完成了：
        1. 对图像做 resize, rescale, normalize 等预处理，并转成 [Batch, C, H, W] 的张量
        2. 在文本前插入 <image> tokens
        3. 用 tokenizer 转文本为 input_ids / attention_mask
        4. 将 'pixel_values' 和文本相关的张量一起返回
        """

        # 首先确保一次只传入 1 个图和 1 条文本
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # 调用 process_images 函数对图片做一系列预处理
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # process_images 返回的是一个列表 (List[np.ndarray])，每个元素形状为 [C, H, W]
        # 利用 np.stack 把这个列表合并成一个 shape 为 [Batch_Size, C, H, W] 的 4D 数组
        pixel_values = np.stack(pixel_values, axis=0)

        # 把这个 numpy 数组转换成 PyTorch tensor
        pixel_values = torch.tensor(pixel_values)

        # 给文本前面插入 image_seq_length 个 <image> token，并加上 bos_token，结尾加换行
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # 用 tokenizer 把 input_strings 转成 input_ids 和 attention_mask
        # return_tensors="pt" 表示返回 PyTorch 张量
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,       # 'longest' 表示会将序列补齐到同样长度
            truncation=truncation, # 超长时是否截断
        )

        # 最后把图像张量 pixel_values 和 tokenizer 输出的其他张量合并到一个字典
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data


