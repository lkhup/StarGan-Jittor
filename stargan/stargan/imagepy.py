import matplotlib.pyplot as plt

# 设置权重超参数
lambda_cls = 1.0
lambda_gp = 10.0
lambda_rec = 10.0

log_path = 'train_log_pytorch.txt'

steps = []
d_loss_real, d_loss_fake, d_loss_cls, d_loss_gp = [], [], [], []
g_loss_fake, g_loss_rec, g_loss_cls = [], [], []
d_loss_total, g_loss_total = [], []

with open(log_path, 'r') as f:
    for line in f:
        if 'D/loss_real' in line:
            parts = line.strip().split(',')

            step = int(parts[1].split('[')[-1].split('/')[0])
            steps.append(step)

            d_real = float(parts[2].split(':')[-1])
            d_fake = float(parts[3].split(':')[-1])
            d_cls = float(parts[4].split(':')[-1])
            d_gp  = float(parts[5].split(':')[-1])

            g_fake = float(parts[6].split(':')[-1])
            g_rec  = float(parts[7].split(':')[-1])
            g_cls  = float(parts[8].split(':')[-1])

            # 保存每项损失
            d_loss_real.append(d_real)
            d_loss_fake.append(d_fake)
            d_loss_cls.append(d_cls)
            d_loss_gp.append(d_gp)

            g_loss_fake.append(g_fake)
            g_loss_rec.append(g_rec)
            g_loss_cls.append(g_cls)

            # 计算总损失
            d_total = d_real + d_fake + lambda_cls * d_cls + lambda_gp * d_gp
            g_total = g_fake + lambda_rec * g_rec + lambda_cls * g_cls

            d_loss_total.append(d_total)
            g_loss_total.append(g_total)

# 画图
plt.figure(figsize=(12, 10))

# 判别器损失图
plt.subplot(2, 1, 1)
plt.plot(steps, d_loss_real, label='D/loss_real')
plt.plot(steps, d_loss_fake, label='D/loss_fake')
plt.plot(steps, d_loss_cls, label='D/loss_cls')
plt.plot(steps, d_loss_gp, label='D/loss_gp')
plt.plot(steps, d_loss_total, label='D/loss_total', linewidth=2.5, linestyle='--', color='black')
plt.ylabel('Discriminator Loss')
plt.legend()
plt.grid(True)

# 生成器损失图
plt.subplot(2, 1, 2)
plt.plot(steps, g_loss_fake, label='G/loss_fake')
plt.plot(steps, g_loss_rec, label='G/loss_rec')
plt.plot(steps, g_loss_cls, label='G/loss_cls')
plt.plot(steps, g_loss_total, label='G/loss_total', linewidth=2.5, linestyle='--', color='black')
plt.xlabel('Step')
plt.ylabel('Generator Loss')
plt.legend()
plt.grid(True)

plt.suptitle('Training Loss Curve with Weighted Total Loss (PyTorch)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('pytorch_loss_curve_with_weighted_total.png')
plt.show()
