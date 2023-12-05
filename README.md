# 狗狗十分钟也能学会的扩散模型教程-基础篇

## 扩散模型的本质
扩散模型定义是：变分推断（Variational Inference）训练参数化的马尔可夫链。太复杂太难懂了是不是？换个说法--

扩散模型本质是：给数据添加噪声，同时基于干净的数据和加噪的数据让神经网络学会如何去噪(既加噪训练阶段，又称前向阶段)，训练完成后，让神经网络从一个高斯噪声中尝试重新恢复出图像(既采样阶段，又称后向阶段)
## 加噪训练阶段实现(前向)
在前向阶段，每次时间步骤timesteps t阶段，都会在原本的图像或者信息上添加一次高斯噪声 $z_{t}\sim N(0,1)$ 。
timestep一般选择2000， 同时有一个参数 $\beta_{t}$ 用于控制每次所加高斯噪声的方差, $\beta_{t}$ 一般从0.0001到0.02逐步增大。

在 $t=0$ 时: 图像为纯净图像: $x_{0}$

在 $t=1$ 时: 加噪后的噪声图: $x_{1}=\sqrt{1-\beta_{1}}x_{0}+\sqrt{\beta_{1}}z_{1} \quad z_{1}\sim N(0,1)$

在 $t=2$ 时: 加噪后的噪声图: $x_{2}=\sqrt{1-\beta_{2}}x_{1}+\sqrt{\beta_{2}}z_{2} \quad z_{2}\sim N(0,1)$

...

在 $t$ 时: 加噪后的噪声图: $x_{t}=\sqrt{1-\beta_{t}}x_{t-1}+\sqrt{\beta_{t}}z_{t} \quad z_{t}\sim N(0,1)$

那么如何从 $x_{0}$ 和 $t$ 直接得到 $x_{t}$ 呢?

令 $\alpha_{t}=1-\beta{t}$, 则:

$x_{t}=\sqrt{\alpha_{t}}x_{t-1}+\sqrt{1-\alpha_{t}}z_{t} \quad z_{t}\sim N(0,1)$

$x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}z_{t-1} \quad z_{t}\sim N(0,1)$

$x_{t}=\sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2}+\sqrt{\alpha_{t}-\alpha_{t}\alpha_{t-1}}z_{t-1}+\sqrt{1-\alpha_{t}}z_{t}$

其中: $\sqrt{\alpha_{t}-\alpha_{t}\alpha_{t-1}}z_{t-1} \sim N(0,\alpha_{t}-\alpha_{t}\alpha_{t-1})$ , $\sqrt{1-\alpha_{t}}z_{t} \sim N(0,1-\alpha_{t})$

所以: $x_{t}=\sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t}\alpha_{t-1}}z$, 根据数学归纳法:

$x_{t}=\sqrt{\bar{\alpha}_ {t}}x_{0}+\sqrt{1-\bar{\alpha}_{t}}z$

每个时刻的噪声图(加噪信号)均可由 $x_{0}$ , $t$ 和 $z$ 表示。在训练阶段，模型会给每个数据添加不同t时刻的噪声图，而去噪网络则会基于t和加噪后信号尝试去还原原始的信号或者添加的噪声亦是噪声的分布。
## 后向阶段实现(后向)
在推理或者说是采样阶段，就是从一个完全高斯噪声中一步一步恢复图像信号的过程

$x_{t-1}= \frac{1}{\sqrt{\alpha_{t}}} (x_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha_{t}}}}\tilde{Z})+\frac{1-\bar{\alpha} _ {t-1}} {1-\bar{\alpha}_{t}}z$

$\tilde{Z}=Denoiser(x_{t},t)\quad z\sim N(0,1)$
# 狗狗十分钟也能学会的扩散模型教程-代码篇
整个模型主要分两部分：

1. 模型的加噪去噪的Scheduler，以及去噪过后的loss函数

3. 去噪网络Denoiser，通常是UNet, 如今也有Transformer_based的出现
## Noise Scheduler模块
这一部分主要写在了Diffusion Class内部，包含了一下几个参数：

T：TimeStep, 代表加噪去噪的总次数

betas：所加噪声的方差 $\beta$

alphas： $1-\beta$

alphas_cumprod： $\bar{\alpha_{t}}$

sqrt_alphas_cumprod

sqrt_one_minus_alphas_cumprod

reciprocal_sqrt_alphas

sigma

同时，diffusion模型有三种不同的预测方式，一种是预测所加t步的噪声，一种是预测 $x_{0}$ 的原始图像， 还有一种是预测得分（梯度），主要使用的是前两种。
## Denoiser模块
经典UNet的结构，狗狗应该熟悉
除此之外输入还包括了时间t, t会被送入PositionEmbedding创造出TimeEmbedding, 然后该TimeEmbedding会与TextEmbedding相加(数值相加或者直接拼接)


