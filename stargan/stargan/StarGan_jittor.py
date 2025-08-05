import random
from jittor import nn, transform as T
from tqdm import tqdm
from datetime import datetime
#from torchvision.utils import save_image
from PIL import Image
import jittor as jt
import os
import numpy as np
from jittor.dataset.dataset import Dataset
jt.flags.use_cuda = 0
class ResidualBlock(nn.Module):
    """残差块，带InstanceNorm"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv1 = nn.Conv(dim_in, dim_out, 3, 1, 1, bias=False)
        self.in1 = nn.InstanceNorm(dim_out, affine=True)
        self.relu = nn.Relu()
        self.conv2 = nn.Conv(dim_out, dim_out, 3, 1, 1, bias=False)
        self.in2 = nn.InstanceNorm(dim_out, affine=True)
    def execute(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)
        return x + out
class Generator(nn.Module):
    """生成器"""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv(3+c_dim, conv_dim, 7, 1, 3, bias=False))
        layers.append(nn.InstanceNorm(conv_dim, affine=True))
        layers.append(nn.Relu())
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv(curr_dim, curr_dim*2, 4, 2, 1, bias=False))
            layers.append(nn.InstanceNorm(curr_dim*2, affine=True))
            layers.append(nn.Relu())
            curr_dim *= 2
        for i in range(repeat_num):
            layers.append(ResidualBlock(curr_dim, curr_dim))
        for i in range(2):
            layers.append(nn.ConvTranspose(curr_dim, curr_dim//2, 4, 2, 1, bias=False))
            layers.append(nn.InstanceNorm(curr_dim//2, affine=True))
            layers.append(nn.Relu())
            curr_dim //= 2
        layers.append(nn.Conv(curr_dim, 3, 7, 1, 3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
    def execute(self, x, c):
        c = c.reshape((c.shape[0], c.shape[1], 1, 1))
        c = jt.repeat(c, 1, 1, x.shape[2], x.shape[3])
        x = jt.concat([x, c], dim=1)
        return self.main(x)
class Discriminator(nn.Module):
    """判别器"""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super().__init__()
        layers = []
        layers.append(nn.Conv(3, conv_dim, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv(curr_dim, curr_dim*2, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim *= 2
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv(curr_dim, 1, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv(curr_dim, c_dim, kernel_size, bias=False)
    def execute(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.reshape(out_cls.shape[0], out_cls.shape[1])
class CelebA(Dataset):

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode,
                 batch_size=16, shuffle=True, num_workers=1, drop_last=False, keep_numpy_array=False):
        super().__init__()
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.keep_numpy_array = keep_numpy_array

        # 预处理，填充训练和测试数据列表
        self.preprocess()

        # 根据mode设置数据集指向
        if self.mode == 'train':
            self.data = self.train_dataset
        else:
            self.data = self.test_dataset

        # 设置总长度
        self.total_len = len(self.data)

        # 设置Jittor数据集属性
        self.set_attrs(
            total_len=self.total_len,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            keep_numpy_array=self.keep_numpy_array
        )

    def preprocess(self):
        """解析属性文件，划分训练/测试集"""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name
        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            label = [values[self.attr2idx[attr]] == '1' for attr in self.selected_attrs]
            if (i + 1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        print(f"Total lines in attr file (excluding header): {len(lines)}")
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        filename, label = self.data[index]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = np.array(label, dtype=np.float32)
        return image, label

    def __len__(self):
        return self.total_len
def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train',num_workers=1):
    """Build and return a Jittor data loader."""
    transform_list = []
    if mode == 'train':
        transform_list.append(T.RandomHorizontalFlip())
    transform_list += [
        T.CenterCrop((crop_size, crop_size)),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    transform = T.Compose(transform_list)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform=transform, mode=mode)
    else:
        raise NotImplementedError("Only CelebA dataset supported in this loader")

    # Jittor的 Dataset 支持 set_attrs() 来返回 DataLoader
    data_loader = dataset.set_attrs(batch_size=batch_size, shuffle=(mode == 'train'))

    return data_loader


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = celeba_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()

    #         if self.use_tensorboard:
    #             self.build_tensorboard()

    def build_model(self):
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)
        # elif self.dataset == 'Both':
        #     self.G = Generator(self.g_conv_dim, self.c_dim + self.c2_dim + 2, self.g_repeat_num)
        #     self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim + self.c2_dim, self.d_repeat_num)

        self.g_optimizer = jt.optim.Adam(self.G.parameters(), lr=self.g_lr, betas=(self.beta1, self.beta2))
        self.d_optimizer = jt.optim.Adam(self.D.parameters(), lr=self.d_lr, betas=(self.beta1, self.beta2))

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print(f"The number of parameters: {num_params}")

    def restore_model(self, resume_iters):
        print(f'Loading the trained models from step {resume_iters}...')
        G_path = os.path.join(self.model_save_dir, f'{resume_iters}-G.ckpt')
        D_path = os.path.join(self.model_save_dir, f'{resume_iters}-D.ckpt')

        self.G.load_parameters(jt.load(G_path))
        self.D.load_parameters(jt.load(D_path))

    #     def build_tensorboard(self):
    #         from logger import Logger
    #         self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def gradient_penalty(self, y, x):
        batch_size = y.shape[0]
        gp = 0
        for i in range(batch_size):
            grad_i = jt.grad(y[i], x)  # 去掉 retain_graph 参数
            grad_i = grad_i.view(-1)
            grad_norm = jt.sqrt(jt.sum(grad_i ** 2))
            gp += (grad_norm - 1) ** 2
        gp /= batch_size
        return gp

    # def gradient_penalty(self, y, x):
    #     grad = jt.grad(y.sum(), x)  # y.sum() 是 scalar，能稳定求导
    #     grad = grad.view(grad.shape[0], -1)  # 每个样本展平
    #     grad_norm = jt.sqrt((grad ** 2).sum(dim=1) + 1e-12)  # 防止 sqrt(0)
    #     gp = ((grad_norm - 1) ** 2).mean()
    #     return gp
    def label2onehot(self, labels, dim):
        batch_size = labels.shape[0]
        out = jt.zeros((batch_size, dim))
        for i in range(batch_size):
            out[i, labels[i].int()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)
            elif dataset == 'RaFD':
                batch_size = c_org.shape[0]
                labels = jt.ones(batch_size) * i
                c_trg = self.label2onehot(labels, c_dim)
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            # Jittor binary_cross_entropy_with_logits默认reduction为mean，不用手动除batch_size
            return nn.binary_cross_entropy_with_logits(logit, target)
        # elif dataset == 'RaFD':
        #     return nn.cross_entropy(logit, target)

    def train(self):

        data_iter = iter(self.data_loader)

        x_fixed, c_org = next(data_iter)
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
        self.x_fixed = x_fixed
        self.c_fixed_list = c_fixed_list

        g_lr = self.g_lr
        d_lr = self.d_lr

        start_iters = self.resume_iters if self.resume_iters else 0
        if self.resume_iters:
            self.restore_model(self.resume_iters)

        print('Start training...')
        for i in tqdm(range(start_iters, self.num_iters), ncols=80):
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                x_real, label_org = next(data_iter)

            x_real = x_real.float()
            label_org = label_org.float()

            # 打乱原始标签，生成目标标签
            rand_idx = jt.randperm(label_org.shape[0])
            label_trg = label_org[rand_idx]

            # 根据数据集选择标签处理方式
            if self.dataset == 'CelebA':
                c_org = label_org
                c_trg = label_trg
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            # 将数据转为 Jittor 张量并送入设备
            x_real = x_real.stop_grad()
            c_org = c_org.stop_grad()
            c_trg = c_trg.stop_grad()
            label_org = label_org.stop_grad()
            label_trg = label_trg.stop_grad()
            out_src, out_cls = self.D(x_real)
            d_loss_real = - jt.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, c_org, self.dataset)

            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = jt.mean(out_src)

            alpha = jt.rand(x_real.shape[0], 1, 1, 1)
            alpha = alpha.broadcast(x_real.shape)
            x_hat = alpha * x_real + (1 - alpha) * x_fake

            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp

            self.reset_grad()
            self.d_optimizer.step(d_loss)

            # Logging
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            if (i + 1) % self.n_critic == 0:
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - jt.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, c_trg, self.dataset)

                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = jt.mean(jt.abs(x_real - x_reconst))

                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls

                self.reset_grad()
                self.g_optimizer.step(g_loss)

                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
            if (i + 1) % self.log_step == 0:
                log_str = "[{}] Iteration [{}/{}]".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i + 1, self.num_iters)
                print(log_str)
                with open(os.path.join(self.log_dir, "train_log_jittor.txt"), "a") as f:
                    f.write(log_str + '\n')

                # 输出本轮所有 loss（D 和 G），只输出实际有的
                for tag, value in loss.items():
                    line = f'{tag}: {value:.4f}'
                    print(line, end=' | ')
                    with open(os.path.join(self.log_dir, "train_log_jittor.txt"), "a") as f:
                        f.write(line + ' | ')
                print()
                with open(os.path.join(self.log_dir, "train_log_jittor.txt"), "a") as f:
                    f.write('\n')
            if (i + 1) % self.sample_step == 0:
                with jt.no_grad():
                    x_sample_list = [x_real]
                    for c_trg in self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs):
                        x_fake = self.G(x_real, c_trg)
                        x_sample_list.append(x_fake)
                    x_concat = jt.concat(x_sample_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, f'{i + 1}-images.jpg')
                    #save_image(self.denorm(x_concat), sample_path, nrow=1, padding=0)
                    print(f'Saved sample images to {sample_path}...')

            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, f'{i + 1}-G.pkl')
                D_path = os.path.join(self.model_save_dir, f'{i + 1}-D.pkl')
                jt.save(self.G.state_dict(), G_path)
                jt.save(self.D.state_dict(), D_path)
                print(f'Saved model checkpoints to {G_path} and {D_path}...')

            if (i + 1) % self.lr_update_step == 0:
                self.g_lr -= self.g_lr / float(self.num_iters)
                self.d_lr -= self.d_lr / float(self.num_iters)
                self.update_lr(self.g_lr, self.d_lr)
                print(f'Decayed learning rates, g_lr: {self.g_lr}, d_lr: {self.d_lr}')
        print("Training finished successfully!")

    def test(self):
        """使用训练好的 StarGAN 模型进行图像翻译（单一数据集）"""
        self.restore_model(self.test_iters)

        # 设置数据加载器
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        for i, (x_real, c_org) in enumerate(data_loader):
            x_real = x_real.stop_grad()  # 推理时禁用梯度
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

            x_fake_list = [x_real]
            for c_trg in c_trg_list:
                c_trg = c_trg.stop_grad()
                x_fake = self.G(x_real, c_trg)
                x_fake_list.append(x_fake)

            # 拼接所有生成图像
            x_concat = jt.concat(x_fake_list, dim=3)  # 水平拼接（B, C, H, W * n)
            for b in range(x_concat.shape[0]):
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i * x_concat.shape[0] + b + 1))
                #save_image(self.denorm(x_concat[b]), result_path)
                print('Saved real and fake images into {}...'.format(result_path))
def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # 创建目录（如不存在）
    for path in [config.log_dir, config.model_save_dir, config.sample_dir, config.result_dir]:
        os.makedirs(path, exist_ok=True)

    # 构建数据加载器
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)

    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)

    # 初始化 Solver 类
    solver = Solver(celeba_loader, rafd_loader, config)

    # 模式选择：训练或测试
    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
        elif config.dataset == 'Both':
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset == 'Both':
            solver.test_multi()
from types import SimpleNamespace

config = SimpleNamespace(
    # Model configuration
    c_dim=5,
    c2_dim=8,
    celeba_crop_size=178,
    rafd_crop_size=256,
    image_size=64,
    g_conv_dim=32,
    d_conv_dim=32,
    g_repeat_num=3,
    d_repeat_num=3,
    lambda_cls=1,
    lambda_rec=10,
    lambda_gp=10,

    # Training configuration
    dataset='CelebA',
    batch_size=4,
    num_iters=300,
    num_iters_decay=100000,
    g_lr=0.0001,
    d_lr=0.0001,
    n_critic=5,
    beta1=0.5,
    beta2=0.999,
    resume_iters=None,
    selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],

    # Test configuration
    test_iters=300,

    # Miscellaneous
    num_workers=0,
    mode='train',
    use_tensorboard=False,

    # Directories
    celeba_image_dir='./data/images',
    attr_path='./data/list_attr_celeba.txt',
    rafd_image_dir='data/RaFD/train',
    log_dir='stargan/logs',
    model_save_dir='stargan/models',
    sample_dir='stargan/samples',
    result_dir='stargan/results',

    # Step size
    log_step=10,
    sample_step=10000,
    model_save_step=300,
    lr_update_step=1000
)

main(config)
