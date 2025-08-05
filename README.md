# StarGAN (Jittor 实现)

本项目为 StarGAN 的 Jittor 实现版本，支持 CelebA  数据集的人脸图像多域翻译任务。实现参考原始 PyTorch 版本，包含完整的数据加载、训练、测试和结果可视化流程。

## 1. 环境配置

操作系统： Windows 11

Python 版本：3.8+

Jittor 版本：1.3.10

依赖库：numpy，pillow，tqdm，matplotlib

由于windows环境冲突等原因，我在尝试三天配置环境之后决定使用Docker镜像容器来运行Jittor框架代码，具体配置细节参照https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-15-00-00-docker/

原pytorch框架版本运行环境可按照https://github.com/yunjey/stargan/tree/master进行配置

## 2. 数据准备

项目支持 CelebA 数据集

```bash
git clone https://github.com/yunjey/StarGAN.git
cd StarGAN/
bash download.sh celeba
```

或者手动下载 CelebA 数据集并解压到如下目录结构：

```
│  README.md
├─img
│
└─stargan
    └─stargan
        │  data_loader.py    #原Pytorch文件
        │  download.sh       #原Pytorch文件
        │  imageji.py		 #Jittor画图
        │  imagepy.py		 #Pytorch画图
        │  logger.py		 #原Pytorch文件
        │  main.py			 #原Pytorch文件
        │  model.py			 #原Pytorch文件
        │  solver.py		 #原Pytorch文件
        │  StarGan_jittor.py	  #Jittor的StarGan实现
        │  train_log_jittor.txt   #Jittor训练log文件
        │  train_log_pytorch.txt  # Pytorch训练log文件
        │
        └─data
            └─celeba
                │list_attr_celeba.txt #标签文件
                │images	#图片数据集
```

##  3. 参数设置

| 参数               | 说明                                                 |
| ------------------ | ---------------------------------------------------- |
| **模型相关**       |                                                      |
| `c_dim`            | 目标属性数量                                         |
| `c2_dim`           | 第二数据集（如RaFD）属性数，当前没用                 |
| `celeba_crop_size` | CelebA图像中心裁剪尺寸（178×178）                    |
| `rafd_crop_size`   | RaFD图像裁剪尺寸（256×256）                          |
| `image_size`       | 训练输入图像尺寸（128×128），会resize                |
| `g_conv_dim`       | 生成器初始卷积通道数（64）                           |
| `d_conv_dim`       | 判别器初始卷积通道数（64）                           |
| `g_repeat_num`     | 生成器残差块重复次数（6）                            |
| `d_repeat_num`     | 判别器卷积层重复次数（6）                            |
| `lambda_cls`       | 分类损失权重（控制属性识别损失对训练的影响）         |
| `lambda_rec`       | 重构损失权重（cycle consistency 损失权重，通常较大） |
| `lambda_gp`        | 梯度惩罚权重（控制判别器梯度惩罚的强度）             |

| 参数              | 说明                                                 |
| ----------------- | ---------------------------------------------------- |
| **训练相关**      |                                                      |
| `dataset`         | 使用的数据集，这里是 'CelebA'                        |
| `batch_size`      | 每批样本数量（8）                                    |
| `num_iters`       | 总训练迭代次数（20次，实际可设置更大）               |
| `num_iters_decay` | 开始学习率衰减的迭代次数（100000，当前训练短不触发） |
| `g_lr`, `d_lr`    | 生成器和判别器学习率（0.0001）                       |
| `n_critic`        | 判别器更新频率：每5步训练生成器1步                   |
| `beta1`, `beta2`  | Adam优化器的超参数                                   |
| `resume_iters`    | 从指定迭代次数恢复训练，None表示从头开始             |
| `selected_attrs`  | CelebA选用的5个属性                                  |

| 参数         | 说明                 |
| ------------ | -------------------- |
| **测试相关** |                      |
| `test_iters` | 测试时加载的迭代次数 |

| 参数              | 说明                              |
| ----------------- | --------------------------------- |
| **杂项**          |                                   |
| `num_workers`     | 数据加载并行线程数（1）           |
| `mode`            | 运行模式，‘train’或‘test’         |
| `use_tensorboard` | 是否使用 Tensorboard 记录训练过程 |

| 参数               | 说明                           |
| ------------------ | ------------------------------ |
| **文件路径相关**   |                                |
| `celeba_image_dir` | CelebA 图片路径                |
| `attr_path`        | CelebA 属性标签文件路径        |
| `rafd_image_dir`   | RaFD 图片路径                  |
| `log_dir`          | 训练日志保存路径               |
| `model_save_dir`   | 模型权重保存路径               |
| `sample_dir`       | 训练过程中生成图像样本保存路径 |
| `result_dir`       | 测试时生成图像保存路径         |

| 参数              | 说明                                              |
| ----------------- | ------------------------------------------------- |
| **步骤间隔相关**  |                                                   |
| `log_step`        | 多少步打印一次日志（1步一次，训练时很频繁）       |
| `sample_step`     | 多少步保存一次生成样本图片（10000步，训练时较少） |
| `model_save_step` | 多少步保存一次模型权重（10000步）                 |
| `lr_update_step`  | 多少步更新一次学习率（1000步）                    |

常设训练参数：

```bash
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

    test_iters=300,

    num_workers=0,
    mode='train',
    use_tensorboard=False,

    celeba_image_dir='./data/images',
    attr_path='./data/list_attr_celeba.txt',
    rafd_image_dir='data/RaFD/train',
    log_dir='stargan/logs',
    model_save_dir='stargan/models',
    sample_dir='stargan/samples',
    result_dir='stargan/results',

    log_step=10,
    sample_step=10000,
    model_save_step=300,
    lr_update_step=1000
```

## 4. 训练脚本

训练完成后，使用以下命令进行测试

Pytorch版本：

```bash
python main.py --mode train --dataset CelebA --image_size 64 --c_dim 5 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

Jittor在jupyter notebook上运行：

![image-20250805170949963](img\image-20250805170949963.png)



## 5. 与 PyTorch 实现的对齐验证

实现对 PyTorch 原始版本进行了训练过程对齐（Pytorch 5000轮训练，Jittor 3000轮训练），验证如下：

### 模型训练 Log 对齐

**Pytorch：**

![image-20250805171502774](img\image-20250805171502774.png)

**Jittor**

![image-20250805172111688](img\image-20250805172111688.png)

## 训练日志片段（Jittor）

```
Iter [0/5000], D/loss_real: 0.7523, D/loss_fake: 0.6395, G/loss_adv: 1.2513, G/loss_cls: 0.1392, G/loss_rec: 8.7319
Iter [100/5000], D/loss_real: 0.6031, D/loss_fake: 0.5217, G/loss_adv: 1.1075, G/loss_cls: 0.1283, G/loss_rec: 7.8234
Iter [1000/5000], D/loss_real: 0.3824, D/loss_fake: 0.4509, G/loss_adv: 1.0913, G/loss_cls: 0.1120, G/loss_rec: 7.0045
...
```

## 文件结构说明

```
├── data/                     # 数据目录
├── models/                   # Generator / Discriminator
├── solver.py                 # Solver类：训练与测试主逻辑
├── main.py                   # 入口主函数
├── preprocess.py             # 数据预处理脚本
├── loss_curve.png            # 损失曲线图
├── assets/                   # 可视化图像
├── README.md
```

## 参考

- [StarGAN (original)](https://github.com/yunjey/stargan)
- [Jittor 官方文档](https://cg.cs.tsinghua.edu.cn/jittor/)



