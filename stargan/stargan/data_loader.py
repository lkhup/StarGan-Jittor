from torch.utils import data #PyTorch 的数据加载核心模块。
from torchvision import transforms as T #图像变换，如裁剪、归一化。
from torchvision.datasets import ImageFolder #适用于普通文件夹组织的图像数据
from PIL import Image #PIL，用于读取图片。
import torch #
import os #
import random #用于随机打乱数据（测试/训练划分用）。


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""
    # image_dir: 图像目录（如：img_align_celeba）。
    #
    # attr_path: 属性文件路径（如：list_attr_celeba.txt）。
    #
    # selected_attrs: 选择的属性标签，如['Black_Hair', 'Blond_Hair', 'Male']。
    #
    # transform: 图像的预处理流程。
    #
    # mode: train or test，决定使用哪个数据子集。

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        # train_dataset, test_dataset：保存图片路径和对应属性标签。
        #
        # attr2idx, idx2attr：属性名称与索引之间的映射。
        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]#打开 self.attr_path 指定的文本文件，把每一行末尾的换行符去掉，并把所有行组成一个字符串列表。
        all_attr_names = lines[1].split()#读取属性文件，每一行是一个图像名和 40 个属性值（1 或 -1）。
        for i, attr_name in enumerate(all_attr_names):#构建属性名和索引之间的映射。
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]# 真实数据从第3行开始
        random.seed(1234)
        random.shuffle(lines)#把 lines 这个列表原地打乱顺序，相当于洗牌操作。
        for i, line in enumerate(lines): #设置随机种子并打乱所有数据的顺序（确保实验可复现）
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):#返回一张图和它的标签（transform 后的图像和 FloatTensor 格式的标签）
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

# crop_size=178：CelebA 图片原始尺寸为 178x218，裁剪为正方形。
#
# image_size=128：缩放到模型输入尺寸。
#
# batch_size=16：每个 batch 的图片数。
#
# num_workers=1：数据加载线程数。

def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),#shuffle：训练阶段打乱数据，测试阶段不打乱。
                                  num_workers=num_workers)
    return data_loader