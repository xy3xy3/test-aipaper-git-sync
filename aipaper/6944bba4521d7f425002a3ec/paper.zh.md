# 混元视频：大型视频生成模型的系统化框架

“弥合闭源与开源视频基础模型之间的鸿沟，加速社区探索。” — 混元基础模型团队

# 摘要

近年来，视频生成领域的最新进展深刻改变了个人生活和各行各业。然而，主流的视频生成模型多数为闭源，导致业界与公众社区在视频生成能力上存在显著性能差距。本文报告介绍了HunyuanVideo，一种新型开源视频基础模型，其视频生成性能可匹敌甚至优于领先的闭源模型。HunyuanVideo构建了一个全面框架，融合了多项核心贡献，包括数据策划、先进架构设计、渐进式模型扩展与训练，以及旨在支持大规模模型训练和推理的高效基础设施。基于此，我们成功训练了一个超过130亿参数的视频生成模型，是所有开源模型中参数规模最大的。通过大量实验与针对性设计，确保了模型具备高视觉质量、运动动态表现、文本与视频对齐以及高级拍摄技巧。专业人类评测结果显示，HunyuanVideo优于之前的最先进模型，如Runway Gen-3、Luma 1.6，以及三款表现优异的中文视频生成模型。通过开源该基础模型及其应用代码，我们旨在缩小闭源与开源社区之间的差距，赋能社区内每一位成员自由探索和验证创新思想，促进更加活跃和多元的视频生成生态。代码可在 https://github.com/Tencent/HunyuanVideo 获得。Hunyuan基础模型团队贡献者名单见报告末尾。

# 1 引言

通过大规模预训练和先进架构，扩散模型 [51, 65, 21, 72, 5, 25, 67, 47] 在生成高质量图像和视频方面表现出优于以往生成对抗网络（GAN）方法 [6] 的性能。然而，与图像生成领域在各类开放平台上涌现出众多新算法和应用不同，基于扩散的视频生成模型仍然相对沉寂。我们认为，这种停滞的主要原因之一是缺乏如文本到图像（T2I）领域 [47] 那样的强大开源基础模型。与图像生成模型社区相比，开源与闭源视频生成模型之间存在显著差距。闭源模型往往压倒公开的开源替代方案，严重限制了公众社区在算法创新方面的潜力。尽管近期最先进的模型 MovieGen [67] 展现了良好的性能，但其开源发布的里程碑尚未确立。

![](images/1.jpg)  

Figure 2: Left: Computation resources used for closed-source and open-source video generation models. Right: Performance comparison between HunyuanVideo and other selected strong baselines.

为填补现有空白并提升公共社区的能力，本文报告了我们开源的基础视频生成模型——混元视频（HunyuanVideo）。该系统框架涵盖了训练基础设施、数据整理、模型架构优化以及模型训练。通过实验，我们发现，单纯对基于Transformer的生成模型[65]（采用Flow Matching[52]训练）在训练数据、计算资源及模型参数上的随机扩展效率不足。因此，我们探索出一种高效的扩展策略，能够在达到预期模型性能的同时，将计算资源需求降低最多达5倍。借助这一最优扩展方案和专用基础设施，我们成功训练了包含130亿参数的大型视频模型，并在互联网规模的图像与视频数据上进行预训练。经过专门设计的渐进式微调策略，混元视频在视觉质量、运动动态、视频-文本对齐及语义场景切换四个关键视频生成指标上表现优异。我们基于1500多个具有代表性的文本提示，由60人组成的评测组，对混元视频与全球领先的视频生成模型（包括Gen-3、Luma 1.6及中国市场中表现最佳的三款商业模型）进行了全面比较。结果表明，混元视频在总体满意度上获得最高评分，尤其在运动动态方面表现突出。

# 2 概述

混元视频是一个涵盖从数据处理到模型部署各个环节的综合视频训练系统。本技术报告结构如下： • 第3节介绍了我们的数据预处理技术，包括过滤和重新标注模型。 • 第4节详细阐述了混元视频各组件的架构设计，以及我们的训练与推理策略。 • 第5节探讨了加速模型训练和推理的方法，支持构建参数量达130亿的大模型。 • 第6节评估了我们的文本生成视频基础模型的性能，并将其与现有最先进的视频生成模型（包括开源和专有模型）进行了对比。 • 最后，第7节展示了基于预训练基础模型构建的多种应用，配以相关可视化内容，以及若干视频相关功能模型，如视频到音频生成模型。

![](images/2.jpg)  

Figure 3: The overall training system for Hunyuan Video.

# 3 数据预处理

我们采用图像与视频联合训练策略。视频被精细划分为五个不同类别，而图像则分为两个类别，每个类别均根据各自训练流程的具体需求进行定制。本节将主要探讨视频数据的策划细节。我们的数据采集过程严格遵循《通用数据保护条例》（GDPR）[39]框架中的原则。此外，我们还采用数据合成和隐私计算等先进技术，以确保符合这些严格标准。我们初始的原始数据池涵盖了包括人物、动物、植物、风景、车辆、物体、建筑和动画等多个领域的视频。每段视频均满足一组基本阈值要求，包括最低时长限制。同时，部分数据基于更严格标准进行收集，如空间质量、特定宽高比要求，以及构图、色彩和曝光方面的专业标准。这些严格的标准保证了我们视频的技术质量与美学价值。实验证明，纳入高质量数据对于显著提升模型性能起到了关键作用。

# 3.1 数据过滤

我们来自不同来源的原始数据在时长和质量水平上存在差异。为了解决这一问题，我们采用一系列技术对原始数据进行预处理。首先，我们使用 PySceneDetect [19] 将原始视频拆分为单镜头视频片段。接着，利用 OpenCV [18] 中的拉普拉斯算子识别清晰帧，作为每个视频片段的起始帧。利用内部 VideoCLIP 模型，我们计算这些视频片段的嵌入向量。这些嵌入向量具有双重作用：(i) 基于嵌入向量的余弦距离对相似片段进行去重；(ii) 采用 $\mathbf{k}$-均值聚类 [59] 获得约 1 万个概念质心，用于概念的重采样和平衡。为了持续提升视频的美学效果、运动表现及概念覆盖范围，我们设计了一个层级化数据过滤流水线以构建训练数据集，如图 4 所示。该流水线包含多种滤波器，帮助我们从不同角度筛选数据，具体介绍如下。我们使用 Dover [85] 评估视频片段的视觉美感，从审美和技术视角进行评价。此外，我们训练了一个模型来判断清晰度，剔除视觉模糊的视频片段。通过估计的光流 [18] 预测视频的运动速度，过滤掉静态或慢动作视频。我们结合 PySceneDetect [19] 与 Transnet v2 [76] 的结果获取场景边界信息。利用内部 OCR 模型，删除包含过多文字的视频片段，并定位裁剪字幕。我们还开发了类似 YOLOX [24] 的视觉模型，用于检测和剔除部分遮挡或敏感信息，如水印、边框及标识。为了验证这些滤波器的有效性，我们利用较小的混元视频（HunyuanVideo）模型开展简单实验，观察性能变化。实验结果在指导我们构建数据过滤流水线方面发挥了重要作用，相关内容将在后文介绍。

![](images/3.jpg)  

Figure 4: Our hierarchical data filtering pipeline. We employ various filters for data filtering and progressively increase their thresholds to build 4 training datasets, i.e., 256p, 360p, 540p, and $7 2 0 \mathrm { p }$ , while the final SFT dataset is built through manual annotation. This figure highlights some of the most important filters to use at each stage. A large portion of data will be removed at each stage, ranging from half to one-fifth of the data from the previous stage. Here, gray bars represent the amount of data filtered out by each filter while colored bars indicate the amount of remaining data at each stage.

我们针对视频数据构建的分层数据过滤流程生成了五个训练数据集，对应五个训练阶段（见第4.5节）。除最后的微调数据集外，这些数据集均通过逐步提高上述过滤器阈值进行筛选。视频空间分辨率由 $256 \times 256 \times 65$ 逐步提升至 $720 \times 1280 \times 129$。在不同阶段的阈值调整过程中，对过滤器施加不同程度的严格限制（详见图4）。接下来描述用于微调的最后一个数据集。为了提升模型在最终阶段的性能（见第4.7节），我们构建了一个包含约100万样本的微调数据集。该数据集通过人工标注精心筛选完成。标注员的任务是识别具备高视觉美感和引人入胜动作内容的视频片段。每个视频片段基于两方面进行评估：（i）拆解的美学视角，包括色彩和谐、光照、物体突出和空间布局；（ii）拆解的运动视角，包括运动速度、动作完整性和运动模糊。最终，我们的微调数据集由视觉效果优美且运动细节丰富的视频片段组成。我们还通过复用绝大多数过滤器（除去运动相关过滤器）建立了针对图像的分层数据过滤流程。类似地，我们基于数以十亿计的图文对图像库，通过逐步提高过滤阈值构建了两个图像训练数据集。第一阶段数据集包含数十亿样本，用于文本到图像预训练的初始阶段。第二阶段数据集包含数亿样本，用于文本到图像预训练的第二阶段。

# 3.2 数据标注

结构化描述。正如研究[7, 4]所证明的那样，描述的准确性和全面性在提升生成模型的提示遵循能力及输出质量方面起着关键作用。此前大多数工作侧重于提供简短描述[14, 50]或密集描述[93, 9, 10]。然而，这些方法存在不足，表现为信息不完整、话语冗余以及准确性欠缺。为实现更具全面性、信息密度和准确性的描述，我们开发并实现了一种内部视觉语言模型（VLM），用于生成图像和视频的结构化描述。这些结构化描述以JSON格式呈现，从多个维度和视角提供描述性信息，包括：1）简短描述：捕捉场景的主要内容。

![](images/4.jpg)  

Figure 5: The overall architecture of HunyuanVideo. The model is trained on a spatial-temporally compressed latent space, which is compressed through Causal 3D VAE. Text prompts are encoded using a large language model, and used as the condition. Gaussian noise and condition are taken as input, our model generates a output latent, which is decoded into images or videos through the 3D VAE decoder.

2) 详尽描述：详尽描绘场景内容，尤其包括与视觉内容相结合的场景切换和摄像机运动，例如摄像机跟随某个主体。3) 背景：描述主体所处的环境。4) 风格：刻画突出或强调特定视觉内容的视频镜头类型，如航拍镜头、特写镜头、中景镜头或远景镜头。6) 光照：描述视频的光照条件。7) 氛围：传达视频的氛围感受，如温馨、紧张或神秘。此外，我们扩展了 JSON 结构，纳入更多来源标签、质量标签和图像视频元信息中的其他相关标签。通过设计精巧的 dropout 机制，结合排列组合策略，将这些多维描述组合生成多样化的不同长度和模式的字幕，旨在提升生成模型的泛化能力，防止过拟合。我们利用该字幕生成器为训练数据集中所有图像与视频提供结构化字幕。摄像机运动类型方面，我们还训练了摄像机运动分类器，能够预测14种不同摄像机运动类别，包括推近、推远、上摇、下摇、左摇、右摇、上俯、下俯、左倾、右倾、左环绕、右环绕、静帧和手持镜头。对摄像机运动类型的高置信度预测将整合进 JSON 格式的结构化字幕中，以赋予生成模型对摄像机运动的控制能力。

# 4 模型架构设计

我们的混元视频模型概览如图5所示。本节将介绍因果三维变分自编码器、扩散主干网络以及规模定律实验。

# 4.1 三维变分自编码器设计

与先前工作[67, 93]类似，我们训练了一个3DVAE，将像素空间的视频和图像压缩到紧凑的潜在空间。为同时处理视频和图像，我们采用了CausalConv3D [95]。对于形状为$(T + 1) \times 3 \times H \times W$的视频，我们的3DVAE将其压缩为形状为$\begin{array}{r}{\left(\frac{T}{c_t} + 1\right) \times C \times \left(\frac{H}{c_s}\right) \times \left(\frac{W}{c_s}\right)}\end{array}$的潜在特征。在我们的实现中，$c_t = 4$，$c_s = 8$，且$C = 16$。该压缩大幅减少了后续扩散变换器模型的词元数量，使我们能够以原始分辨率和帧率训练视频。模型结构如图6所示。

![](images/5.jpg)  

Figure 6: The architecture of our 3DVAE.

# 4.1.1 训练

与大多数先前工作[67, 11, 104]不同，我们不依赖预训练的图像VAE进行参数初始化；相反，我们从零开始训练模型。为了平衡视频和图像的重建质量，我们按4:1的比例混合视频和图像数据。除了常用的$L_1$重建损失和KL损失$L_{kl}$之外，我们还引入了感知损失$L_{lpips}$和GAN对抗损失$L_{adv}$ [22]以提升重建质量。完整的损失函数如公式1所示。

$$
\mathrm { L o s s } = L _ { 1 } + 0 . 1 L _ { l p i p s } + 0 . 0 5 L _ { a d v } + 1 0 ^ { - 6 } L _ { k l }
$$

训练过程中，我们采用课程学习策略，逐步从低分辨率短视频训练到高分辨率长视频。为了提高高速运动视频的重建效果，我们从区间 $1 \sim 8$ 中随机选择采样间隔，以均匀抽取视频片段中的帧。

# 4.1.2 推理

在单个GPU上编码和解码高分辨率长视频可能导致内存溢出（OOM）错误。为了解决这一问题，我们采用时空瓦片策略，将输入视频沿空间和时间维度切分为重叠的瓦片。每个瓦片分别进行编码/解码，输出结果再进行拼接。对于重叠区域，我们采用线性组合进行融合。该瓦片策略使我们能够在单一GPU上处理任意分辨率和时长的视频编码/解码。我们观察到，在推理阶段直接使用瓦片策略会由于训练与推理不一致导致明显伪影。为此，我们引入了一个额外的微调阶段，在训练中随机启用或禁用瓦片策略，从而确保模型同时兼容瓦片和非瓦片策略，保持训练和推理的一致性。表1对比了我们所提出的VAE与开源的最先进VAE。在视频数据上，我们的VAE相比其他视频VAE表现出明显更高的峰值信噪比（PSNR）。在图像方面，我们的性能超越了视频VAE和图像VAE。图7展示了多个 $256 \times 256$ 分辨率的案例。我们提出的VAE在文字、小人脸和复杂纹理方面表现出显著优势。

Table 1: VAE reconstruction metrics comparison.   

<table><tr><td>Model</td><td>Downsample Factor</td><td>|z|</td><td>ImageNet (256×256) PSNR↑</td><td>MCL-JCV (33×360×640) PSNR↑</td></tr><tr><td>FLUX-VAE [47]</td><td>1×8×8</td><td>16</td><td>32.70</td><td>-</td></tr><tr><td>OpenSora-1.2 [102]</td><td>4×8×8</td><td>4</td><td>28.11</td><td>30.15</td></tr><tr><td>CogvideoX-1.5 [93]</td><td>4× 8×8</td><td>16</td><td>31.73</td><td>33.22</td></tr><tr><td>Cosmos-VAE [64]</td><td>4×8×8</td><td>16</td><td>30.07</td><td>32.76</td></tr><tr><td>Ours</td><td>4×8×8</td><td>16</td><td>33.14</td><td>35.39</td></tr></table>

![](images/6.jpg)  

Figure 7: VAE reconstruction case comparison.

![](images/7.jpg)  

Figure 8: The architecture of our HunyuanVideo Diffusion Backbone.

# 4.2 统一的图像与视频生成架构

本节中，我们介绍混元视频中的Transformer设计，其采用统一的全注意力机制，主要基于以下三点原因：首先，相较于分割的时空注意力机制[7, 67, 93, 79]，其表现更为优越；其次，该机制支持图像和视频的统一生成，简化训练流程并提升模型的可扩展性；最后，它能够更有效地利用现有大语言模型相关的加速技术，提升训练和推理效率。模型结构如图8所示。

输入。对于给定的视频-文本对，模型在第4.1节描述的三维潜在空间中运行。具体来说，对于视频分支，输入首先被压缩为形状为 $T \times C \times H \times W$ 的潜在表示。为了统一输入处理，我们将图像视为单帧视频。这些潜在表示通过核尺寸为 $k_{t} \times k_{h} \times k_{w}$ 的三维卷积进行处理，其输出形状为 $\frac{T}{k_{t}} \times \frac{H}{k_{h}} \times \frac{W}{k_{w}}$。对于文本分支，我们首先使用先进的大型语言模型（LLM）将文本编码为一系列嵌入，这些嵌入捕捉细粒度的语义信息。同时，我们采用CLIP模型提取包含全局信息的池化文本表示。该表示随后被扩展维度并加到时间步嵌入上，然后输入到模型中。

Table 2: Architecture hyperparameters for the HunyuanVideo 13B parameter foundation model.   

<table><tr><td>Dual-stream Blocks</td><td>Single-stream locks</td><td>Model Dimension</td><td>FFN Dimension</td><td>Attention Heads</td><td>Head dim</td><td>(d, dh, dw)</td></tr><tr><td>20</td><td>40</td><td>3072</td><td>12288</td><td>24</td><td>128</td><td>(16, 56, 56)</td></tr></table>

模型设计。为了有效整合文本和视觉信息，我们遵循文献[47]中用于视频生成的“二流到单流”混合模型设计策略。在二流阶段，视频和文本词元通过多个Transformer模块独立处理，使每种模态能够学习各自合适的调制机制而不受干扰。在单流阶段，我们将视频和文本词元拼接后输入后续Transformer模块，实现多模态信息的有效融合。该设计捕捉视觉与语义信息之间的复杂交互，提升整体模型性能。 位置编码。为了支持多分辨率、多纵横比及不同时长的视频生成，我们在每个Transformer模块中采用旋转位置编码（RoPE）[77]。RoPE通过对嵌入向量施加旋转频率矩阵，增强模型捕捉绝对和相对位置关系的能力，并在大语言模型中表现出一定的外推能力。鉴于视频数据中时间维度的复杂性，我们将RoPE扩展到三维。具体地，我们分别针对时间$(T)$、高度$(H)$和宽度$(W)$的坐标计算旋转频率矩阵。随后，我们将查询和键的特征通道划分为三段$(d_t, d_h, d_w)$，每段乘以对应坐标的旋转频率，最后将三段拼接。该过程生成位置感知的查询和键嵌入，用于注意力计算。详细模型设置请参见表2。

# 4.3 文本编码器

在文本到图像和文本到视频等生成任务中，文本编码器通过在潜在空间中提供引导信息起着关键作用。一些代表性工作[66, 21, 51]通常采用预训练的CLIP[69]和T5-XXL[71]作为文本编码器，其中CLIP采用Transformer编码器结构，T5采用编码器-解码器结构。相比之下，我们使用一个预训练的仅解码器结构的多模态大语言模型（MLLM）作为文本编码器，具有以下优势：（i）相比T5，经过视觉指令微调后的MLLM在特征空间中具有更好的图文对齐能力，缓解了扩散模型中遵循指令的难度；（ii）相比CLIP，MLLM在图像细节描述和复杂推理方面表现出更强的能力[53]；（iii）MLLM能够通过在用户提示前加入系统指令，实现零样本学习[8]，帮助文本特征更关注关键信息。此外，如图9所示，MLLM基于因果注意力机制，而T5-XXL采用双向注意力机制，这有利于为扩散模型提供更优质的文本引导。因此，我们借鉴[55]引入额外的双向词元优化器以增强文本特征。我们针对不同应用场景，配置了一系列MLLM[78, 17, 26]以构建混元视频系统。在各配置下，MLLM均显示出优于传统文本编码器的性能表现。此外，CLIP的文本特征作为文本信息的摘要也极具价值。如图8所示，我们采用CLIP-Large文本特征中最后一个非填充词元作为全局引导，融合进双流和单流的DiT模块中。

# 4.4 模型扩展

语言模型训练中的神经扩展定律[41, 36]为理解和优化机器学习模型的性能提供了有力工具。通过阐明模型规模$(N)$、数据集规模$(D)$和计算资源$(C)$之间的关系，这些定律促进了更有效和更高效模型的开发，最终推动了大规模模型训练的成功。

![](images/8.jpg)  

Figure 9: Text encoder comparison between T5 XXL and the instruction-guided MLLM introduced by HunyuanVideo.

![](images/9.jpg)  

Figure 10: Scaling laws of DiT-T2X model family. On the top-left (a) we show the loss curves of the T2X(I) model on a log-log scale for a range of model sizes from 92M to 6.6B. We follow [36] to plot the envelope in gray points, which are used to estimate the power-law coefficients of the amount of computation $( C )$ vs model parameters $( N )$ (b) and the computation vs tokens $( D )$ (c). Based on the scaling law of the T2X(I) model, we plot the scaling law of the corresponding T2X(V) model in (d), (e), and (f).

与先前关于大型语言模型[41, 36, 81, 1, 2]和图像生成模型[49, 42]的规模定律不同，视频生成模型通常依赖于预训练的图像模型。因此，我们的第一步是确立与文本到图像相关的基础规模定律。在此基础规模定律的基础上，我们进一步推导出了适用于文本到视频模型的规模定律。通过整合这两套规模定律，我们能够系统地确定视频生成任务中合适的模型和数据配置。

# 4.4.1 图像模型规模规律

Kaplan 等人[41]和 Hoffmann 等人[36]探讨了语言模型在交叉熵损失上的经验规模定律。在基于扩散的视觉生成领域，Li 等人[49]研究了 UNet 的规模特性，而基于 Transformer 的工作如 DiT[65]、U-ViT[3]、Lumina-T2X[23]和 SD3[21]仅研究了样本质量与网络复杂度之间的规模行为，尚未对扩散模型所用的计算资源与均方误差（MSE）损失的幂律关系进行探索。为填补这一空白，我们开发了一系列类似 DiT 的模型，命名为 DiT-T2X，以区别于原始 DiT，其中 X 可代表图像（I）或视频（V）。DiT-T2X 采用 T5-XXL[71]作为文本编码器，采用上述的 3D VAE 作为图像编码器。文本信息通过 92M 到 6.6B 规模的模型进行编码。所有模型均使用 DDPM[34]和 v-预测[73]方法，采用一致的超参数和相同的数据集，图像分辨率为 256 像素进行训练。我们遵循[36]提出的实验方法，建立神经网络规模定律进行拟合。

$$
N _ { o p t } = a _ { 1 } C ^ { b _ { 1 } } , \quad D _ { o p t } = a _ { 2 } C ^ { b _ { 2 } } .
$$

如图10(a)所示，各模型的损失曲线从左上角向右下角下降，且总是经过与之相邻的较大尺寸模型的损失曲线。这意味着在两个交点之间的计算资源范围内，中等尺寸模型是最优的（具有最低损失）。在获得横轴所有取值对应的最低损失包络线后，我们带入方程（）中，得到 $a_{1} = 5.48 \times 10^{-4}$，$b_{1} = 0.5634$，$a_{2} = 0.324$ 和 $b_{2} = 0.4325$，其中 $a_{1}$、$a_{2}$、$N_{opt}$、$D_{opt}$ 的单位为十亿级别，$C$ 的单位为千万亿次浮点运算（Peta FLOPs）。图10(b)和图10(c)显示，DiT-T2X(I)系列很好地符合幂律关系。最后，给定计算预算，我们可以计算出最优的模型规模和数据集规模。

# 4.4.2 视频模型缩放规律

基于 T2X(I) 模型的扩展规律，我们选择对应各个规模模型的最优图像检查点（即包络线上的模型）作为视频扩展规律实验的初始化模型。图 10(d)、图 10(e) 和图 10(f) 展示了 T2X(V) 模型的扩展规律结果，其中 $a_1 = 0.0189$，$b_1 = 0.3618$，$a_2 = 0.0108$，$b_2 = 0.6289$。结合图 10(b) 和图 10(e) 的结果，并综合考虑训练消耗和推理成本，我们最终将模型规模设定为 13B。然后，图像和视频训练的词元数量可按图 10(c) 和图 10(f) 所示计算。值得注意的是，依据图像和视频扩展规律计算的训练词元数量，仅涉及图像和视频各自的第一阶段训练。由低分辨率到高分辨率的渐进式训练的扩展特性将在未来工作中进一步探索。

# 4.5 模型预训练

我们采用Flow Matching [52]进行模型训练，并将训练过程分为多个阶段。首先，我们在256像素和512像素的图像上预训练模型，然后在分辨率从256像素到960像素的图像和视频上进行联合训练。

# 4.5.1 训练目标

在本工作中，我们采用流匹配框架[52, 21, 13]来训练我们的图像和视频生成模型。流匹配通过对概率密度函数进行一系列变量变换，将复杂的概率分布转化为简单的概率分布，并通过逆变换生成新的数据样本。

在训练过程中，给定训练集中图像或视频的潜在表示$\mathbf{x}_{1}$。我们首先从对数正态分布[21]中采样$t \in [0,1]$，并按照高斯分布初始化噪声$\mathbf{x}_{0} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。然后通过线性插值方法[52]构造训练样本$\mathbf{x}_{t}$。模型被训练以预测速度$\mathbf{u}_{t} = d\mathbf{x}_{t}/dt$，该速度引导样本$\mathbf{x}_{t}$朝向样本$\mathbf{x}_{1}$。通过最小化预测速度$\mathbf{v}_{t}$与真实速度$\mathbf{u}_{t}$之间的均方误差来优化模型参数，表达为损失函数。

$$
\mathcal { L } _ { \mathrm { g e n e r a t i o n } } = \mathbb { E } _ { t , { \mathbf { x } _ { 0 } } , { \mathbf { x } _ { 1 } } } \| \mathbf { v } _ { t } - { \mathbf { u } } _ { t } \| ^ { 2 } .
$$

在推理过程中，初始时从高斯分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 中采样噪声样本 $\mathbf{x}_0$。然后，利用一阶欧拉常微分方程（ODE）求解器，通过积分模型估计的 $\frac{d \mathbf{x}_t}{dt}$ 来计算 $\mathbf{x}_1$。该过程最终生成最终样本 $\mathbf{x}_1$。

# 4.5.2 图像预训练

在早期实验中，我们发现良好的预训练模型能够显著加快视频训练的收敛速度并提升视频生成性能。因此，我们引入了一个两阶段渐进式图像预训练策略，作为视频训练的热身阶段。图像阶段1（256像素训练）。模型首先使用低分辨率的256像素图像进行预训练。具体来说，我们遵循先前的工作[66]，基于256像素启用多宽高比训练，这有助于模型学习生成具有多种宽高比的图像，同时避免图像预处理过程中裁剪操作导致的文本与图像错位问题。与此同时，使用低分辨率样本进行预训练使模型能够从更多样本中学习更多低频概念。

图像阶段2（混合尺度训练）。我们引入第二个图像预训练阶段，以进一步提升模型在更高分辨率（如512px）上的能力。一种简单的解决方案是直接在512px图像上进行微调。然而，我们发现模型在512px图像上微调后的性能在256px图像生成上会严重下降，这可能会影响后续在256px视频上的预训练。因此，我们提出了混合尺度训练方法，在每个训练全局批次中包含两个或更多尺度的多宽高比桶。每个尺度对应一个锚点大小，然后基于该锚点大小构建多宽高比桶。我们在具有256px和512px锚点大小的两尺度数据集上训练模型，以提升对高分辨率图像的学习能力，同时保持对低分辨率图像的适应能力。我们还引入了针对不同图像尺度微批次的动态批量大小，以最大化GPU内存和计算资源的利用率。

# 4.5.3 视频-图像联合训练

多种长宽比和时长分桶。在第3.1节描述的数据过滤流程后，视频具有不同的长宽比和时长。为有效利用数据，我们根据时长和长宽比分别将训练数据划分到不同的桶中。我们创建了 \(B_{T}\) 个时长桶和 \(B_{AR}\) 个长宽比桶，共计 \(B_{T} \times B_{AR}\) 个桶。由于各桶中的词元数量不同，我们为每个桶分配一个最大批量大小，以防止显存溢出（OOM）错误，从而优化GPU资源利用。在训练开始前，所有数据被分配到最近的桶中。训练过程中，各计算节点随机从某一桶预取批次数据。此随机选择保证模型每一步都在不同数据规模上训练，有助于保持模型泛化能力，避免在单一尺寸数据上的限制。 渐进式视频-图像联合训练。直接从文本生成高质量长时长视频序列常导致模型难以收敛且效果不佳。因此，渐进式课程学习成为文本生成视频模型训练的主流策略。在混元视频（HunyuanVideo）中，我们设计了综合的课程学习策略，从利用文本生成图像（T2I）参数初始化模型开始，逐步增加视频时长和分辨率。 • 低分辨率短视频阶段。模型建立文本与视觉内容的基础映射，确保短时动作的一致性和连贯性。 • 低分辨率长视频阶段。模型学习更复杂的时间动态和场景变换，确保更长时段的时空一致性。 • 高分辨率长视频阶段。模型提升视频分辨率和细节质量，同时保持时间连贯性并应对复杂的时间动态。 此外，在每个阶段，我们按不同比例引入图像数据进行视频-图像联合训练。该方法解决了高质量视频数据稀缺的问题，使模型能够学习更广泛多样的世界知识。同时，有效避免因视频与图像数据分布差异导致的图像空间语义灾难性遗忘。

# 4.6 提示重写

为应对用户提供的提示语在语言风格和长度上的多样性，我们采用混元大模型（Hunyuan-Large）[78]作为提示重写模型，将原始用户提示调整为模型偏好的提示格式。该提示重写模型在无训练框架下运行，利用详细的提示指令和上下文学习示例来提升性能。该提示重写模块的主要功能包括： • 多语言输入适应：模块设计能够处理和理解多种语言的用户提示，确保语义和上下文的完整保留。 • 提示结构规范化：模块将提示语改写为符合标准化信息架构的格式，类似于训练时使用的描述文本。 • 复杂术语简化：模块将用户复杂措辞简化为更通俗易懂的表达，同时保持用户的原始意图。 此外，我们采用自我修订技术[43]对最终提示进行优化，通过对比原始提示与重写版本，确保输出准确且符合模型能力。为加速和简化应用流程，我们还基于高质量的无训练方法采集的重写对数据，采用LoRA微调混元大模型，实现提示重写的定制化优化。

# 4.7 高性能模型微调

在预训练阶段，我们使用了一个大型数据集进行模型训练。尽管该数据集信息丰富，但其数据质量存在较大差异。为了构建一个能够生成高质量动态视频、并提升在连续运动控制和角色动画方面表现的鲁棒生成模型，我们从完整数据集中精心挑选了四个特定子集进行微调。这些子集经过自动化数据筛选技术的初步筛查，随后进行了人工复核。此外，我们还实施了多种模型优化策略以最大化生成性能。

# 5 模型加速

![](images/10.jpg)  

Figure 11: (a) Different time-step schedulers. For our shifting stragty, we set a larger shifting factor $s$ for a lower inference step. (b) Generated videos with only 10 inference steps. The shifting stragty leads to significantly better visual quality.

# 5.1 推理步骤减少

为了提高推理效率，我们首先考虑减少推理步数。与图像生成相比，在较少推理步数的情况下保持生成视频的时空质量更具挑战性。受此前观察到在生成过程中前几个时间步对变化贡献最大的启发[101, 67, 98, 99]，我们采用时间步移位方法来处理较少推理步数的情况。具体来说，给定推理步数 $q \in \{ 1, 2, ..., Q \}$，生成模型的输入时间条件为 $\textstyle t = 1 - \frac{q}{Q}$，其中噪声在 $t=1$ 初始化，生成过程在 $t=0$ 结束。我们不直接使用 $t$，而是通过移位函数将 $t$ 映射为 $t'$，即 $\bar{t'} = \frac{s * t}{1 + (s - 1) * t}$，其中 $s$ 是移位因子。当 $s > 1$ 时，流模型更依赖于较早的时间步。一个关键观察是，较少的推理步数需要更大的移位因子 $s$。经验上，当推理步数为50时，$s$ 设为7，而当推理步数少于20时，$s$ 应增加到17。时间步移位策略使生成模型在减少步数的情况下，仍能匹配多步推理的生成效果。MovieGen[67] 采用线性二次调度器以实现类似目的。调度器的曲线如图11a所示。然而，我们发现，在极少步数推理（如10步）情况下，时间步移位方法比线性二次调度器更有效。如图11b所示，线性二次调度器会导致更差的视觉质量。

# 5.2 文本指导蒸馏

无分类器引导（CFG）[35]显著提升了文本条件扩散模型的样本质量和运动稳定性。然而，它增加了计算成本和推理延迟。在视频模型和高分辨率视频生成中，同时生成文本条件和非条件视频时，推理负担极其昂贵。为解决这一限制，我们将非条件和条件输入的组合输出蒸馏到单一的学生模型中[60]。具体而言，学生模型以引导尺度为条件，且结构和超参数与教师模型相同。我们以教师模型的参数初始化学生模型，并在训练时随机采样从1到8的引导尺度。实验表明，文本引导蒸馏大约带来了1.9倍的加速。

# 5.3 高效且可扩展的训练

为了实现可扩展性和高效训练，我们在腾讯天使机器学习团队的大规模预训练框架 AngelPTM [62] 上训练混元视频模型。此部分首先概述用于训练的硬件与基础设施，随后详细介绍模型并行方法及其优化策略，最后讲解自动容错机制。

# 5.3.1 硬件基础设施

为了确保大规模分布式训练中的高效通信，我们搭建了专门的分布式训练框架——腾讯星麦网络[48]，以实现高效的服务器间通信。所有训练任务的GPU调度均通过腾讯天使机器学习平台完成，该平台提供强大的资源管理与调度能力。

# 5.3.2 并行策略

混元视频训练采用五维并行策略，包括张量并行（TP）[74]、序列并行（SP）[45]、上下文并行（CP）[63]以及结合Zero优化的 数据并行（DP + ZeroCache [62]）。张量并行（TP）基于矩阵块计算原理，将模型参数（张量）拆分到不同GPU上，以减少显存占用并加速计算。每个GPU负责计算层内张量的不同部分。序列并行（SP）建立在TP基础上，将输入序列维度切分，减少像LayerNorm和Dropout等算子的重复计算，降低相同激活值的存储，从而有效减少计算资源和显存的浪费。此外，对于不满足SP要求的输入数据，工程上支持等效的SP填充功能。上下文并行（CP）在序列维度切片以支持长序列训练，每个GPU负责计算不同序列切片的Attention。具体而言，采用Ring Attention [30]实现多GPU高效长序列训练，突破单GPU显存限制。此外，利用数据并行结合ZeroCache支持通过数据并行的横向扩展，满足训练数据集规模增长的需求。随后，基于数据并行进一步采用ZeroCache优化策略，减少模型状态（模型参数、梯度及优化器状态）的冗余，统一管理GPU显存使用，最大化显存利用效率。

# 5.3.3 优化

注意力优化。随着序列长度的增加，注意力计算成为训练的主要瓶颈。我们通过融合注意力（FusedAttention）加速了注意力计算。重计算与激活卸载优化。重计算是一种以计算换存储的技术，主要包括三个部分：a）指定某些层或块进行重计算，b）在前向计算中释放激活，c）在反向计算中通过重计算获取依赖激活，从而显著减少训练时GPU内存的使用。此外，考虑到PCIe带宽和主机内存容量，采用了基于层的激活卸载策略。在不降低训练性能的前提下，将GPU内存中的激活卸载至主机内存，进一步节省GPU内存。

# 5.3.4 自动容错

在混元视频大规模训练的稳定性方面，采用了自动容错机制，能够快速恢复因常见硬件故障导致的训练中断，避免了训练任务频繁人工恢复的问题。通过自动检测错误并迅速替换健康节点以接管训练任务，训练稳定性达到了99.5%。

# 6 基础模型性能

文本对齐 视频生成模型的一个关键指标是其准确遵循文本提示的能力。这一能力对于模型的有效性至关重要。然而，一些开源模型在捕捉所有主体或准确表达多个主体之间关系时常常存在困难，尤其是在输入文本提示较为复杂的情况下。HunyuanVideo 展现了在生成严格遵循给定文本提示的视频方面的强大能力。如图12所示，它能够有效处理场景中的多个主体。

![](images/11.jpg)  

Figure 12: Prompt: A white cat sits on a white soft sofa like a person, while its long-haired male owner, with his hair tied up in a topknot, sits on the floor, gazing into the cat's eyes. His child stands nearby, observing the interaction between the cat and the man.

高质量 我们还进行了微调过程，以提升生成视频的空间质量。如图13所示，鸿元视频能够生成极致细腻的内容。高动态表现 在这一部分，我们展示了鸿元视频根据给定提示生成高动态视频的能力。如图14所示，我们的模型擅长生成涵盖丰富场景和多种运动类型的视频。概念泛化 生成模型最理想的特性之一是概念泛化能力。如图15所示，文本提示描述了一个场景：“在遥远的星系中，一名宇航员漂浮在熠熠生辉的粉色宝石状湖面上，湖水映射着周围天空的绚丽色彩，营造出惊艳的画面。宇航员轻柔地漂浮在湖面上，湖水的轻声低语仿佛诉说着行星的秘密。他伸出手指……” (b) 提示：一位时尚女性走在东京街头，街道上充满温暖的霓虹灯光和动感的城市氛围，她戴着主干网络风格的太阳镜，涂着红色口红。她自信而随意地行走。街道湿润且有反光效果，营造出五彩灯光的镜面效果。许多行人穿行其间。

![](images/12.jpg)

![](images/13.jpg)  
(a) Prompt: the ultra-wide-angle lens follows closely from the hood, with raindrops continuously splattering aginst theenshead  sports car speearou a corer, is res violentl skiiagainst he we r, hee   

Figure 13: High-quality videos generated by HunyuanVideo.

![](images/14.jpg)

（a）提示：在日落时分，一辆改装过的福特F-150 Raptor在越野赛道上轰鸣而过。加高的悬挂使得巨大的防爆轮胎能够在泥地上自由翻转，泥浆飞溅到防滚架上。

![](images/15.jpg)

() 镜头缓慢推移，画面中景深聚焦在中景，温暖的夕阳光洒满屏幕。画面中的女孩裙摆飞扬地奔跑，转身跳跃。

![](images/16.jpg)

操作：在健身房，一名穿着运动服的女性在跑步机上跑步，室内环境专业真实。

![](images/17.jpg)

(d) 提示词：游泳者在水下游动，慢动作。写实风格，水下灯光，宁静氛围。

![](images/18.jpg)  

Figure 14: High-motion dynamics videos generated by HunyuanVideo.

在现实、自然光照和休闲场景下拍摄了一张16岁男孩在凉爽、平滑水面上的棒球照片。值得注意的是，该特定场景在训练数据集中并未出现。此外，很明显所描绘的场景融合了多个训练数据中同样缺失的概念。

![](images/19.jpg)  

Figure 15: HunyuanVideo's performance on concept generalization. The results of the three rows correspond to the text prompts (1) 'In a distant galaxy, an astronaut foats on a shimmering, pik, gemstone-like lak that re n color  the undiy, eat sseThet nyri lak' rae, teo   a hiper e plan et He ee uthisrti o hecol  watr.   Amae apture t hestrist playistrets (The night-blooming cactus fowers in the evening, with a bri, rapid closure.Time-lapse shot, extreme close-up. Realistic, Night lighting, Mysterious.' respectively.

动作推理与规划 利用大语言模型的能力，混元视频能够根据给定的文本提示生成连续动作。如图16所示，混元视频以逼真的真实感风格有效地捕捉了所有动作。

![](images/20.jpg)  

Figure 16: Prompt: The woman walks over and opens the red wooden door. As the door swings open, seawater bursts forth, in a realistic style.

角色理解与书写 混元视频能够生成场景文本和如图17所示逐渐显现的手写文本。

# 6.1 与最先进模型的比较

为了评估混元视频的性能，我们选取了五个强大的闭源视频生成模型作为基线。共使用了1533条文本提示，并在一次运行中用混元视频生成了等量的视频样本。为确保公平比较，我们仅进行了一次推理，避免了挑选最佳结果的情况。在与基线方法比较时，所有选定模型均保持默认设置，并确保视频分辨率一致。60名专业评审进行了评估，结果如表3所示。视频评估基于三个指标：文本匹配度、动作质量和视觉质量。值得注意的是，混元视频在整体表现上领先，尤其在动作质量方面表现出色。我们从1533个视频中随机抽取600个供公众访问1。

![](images/21.jpg)  

Figure 17: High text-video alignment videos generated by HunyuanVideo. Top row: Prompt: A close-up of a wave crashing against the beach, the sea foam spells out "WAKE UP" on the sand. Bottom row: Prompt: In a garden filled with blooming flowers, "GROW LOVE" has been spelled out with colorful petals.

Table 3: Model Performance Evaluation   

<table><tr><td>Model Name</td><td>Duration</td><td>Text Alignment</td><td>Motion Quality</td><td>Visual Quality</td><td>Overall</td><td>Ranking</td></tr><tr><td>HunyuanVideo (Ours)</td><td>5s</td><td>61.8%</td><td>66.5%</td><td>95.7%</td><td>41.3%</td><td>1</td></tr><tr><td>CNTopA (API)</td><td>5s</td><td>62.6%</td><td>61.7%</td><td>95.6%</td><td>37.7%</td><td>2</td></tr><tr><td>CNTopB (Web)</td><td>5s</td><td>60.1%</td><td>62.9%</td><td>97.7%</td><td>37.5%</td><td>3</td></tr><tr><td>GEN-3 alpha (Web)</td><td>6s</td><td>47.7%</td><td>54.7%</td><td>97.5%</td><td>27.4%</td><td>4</td></tr><tr><td>Luma1.6 (API)</td><td>5s</td><td>57.6%</td><td>44.2%</td><td>94.1%</td><td>24.8%</td><td>5</td></tr><tr><td>CNTopC (Web)</td><td>5s</td><td>48.4%</td><td>47.2%</td><td>96.3%</td><td>24.6%</td><td>6</td></tr></table>

# 7 应用

# 7.1 基于视频的音频生成

我们的视频到音频（V2A）模块旨在通过融合同步的音效和情境恰当的背景音乐来增强生成的视频内容。在传统电影制作流程中，拟音设计（Foley）是不可或缺的组成部分，对视觉媒体的听觉真实感和情感深度具有重要贡献。然而，拟音音频的制作既耗时又需要高度专业的技能。随着文本到视频（T2V）生成模型的不断增多，大多数模型缺乏对应的拟音生成能力，因而限制了其制作全方位沉浸式内容的能力。我们的V2A模块通过自主生成与输入视频和文本提示相匹配的电影级拟音音频，解决了这一关键瓶颈，从而实现了连贯且整体吸引人的多媒体体验合成。

# 7.1.1 数据

与文本到视频（T2V）模型不同，视频到音频（V2A）模型对数据有不同的要求。如上所述，我们构建了一个包含视频-文本对的视频数据集。然而，该数据集中并非所有数据都适合用于训练V2A模型。例如，一些视频没有音频流，另一些则包含大量配音内容，或者其环境音轨已被移除并替换为无关元素。为了解决这些问题并确保数据质量，我们设计了一个专门针对V2A训练的鲁棒数据过滤流程。首先，我们剔除无音频流或静音比例超过80%的视频。接着，我们采用类似[38]的帧级音频检测模型，对音频流中的语音、音乐和一般声音进行检测。基于此分析，我们将数据分类为四种类型：纯声音、含语音的声音、含音乐的声音和纯音乐。随后，为了优先选择高质量数据，我们训练了一个受CAVP [54]启发的模型，以计算视觉-音频一致性分数，该分数量化每个视频视觉与听觉成分之间的对齐度。结合该评分系统与音频类别标签，我们系统地从每个类别中采样数据，最终保留了约25万小时的原始数据用于预训练。在有监督微调阶段，我们进一步优化筛选，精选出千万级高质量片段（8万小时）子集进行训练。

![](images/22.jpg)  

Figure 18: The architecture of sound effect and music generation model.

在特征提取方面，我们使用 CLIP [70] 以每秒 4 帧的时间分辨率获取视觉特征，随后对这些特征进行重采样以与音频帧率对齐。为了生成字幕，我们采用 [29] 作为声音字幕生成模型，采用 [20] 作为音乐字幕生成模型。当同时获得声音事件检测（sod）和音乐识别（mui）的字幕时，我们将其合并为结构化字幕格式，遵循文献 [67] 中详细描述的方法。

# 7.1.2 模型

与上述文本到视频模型类似，我们的视频到音频生成模型也采用基于流匹配的扩散变换器（DiT）作为其架构主干。模型的详细设计如图18所示，展示了从三流结构向单流DiT框架的转变。该模型在通过变分自编码器（VAE）编码的潜在空间中运行，该VAE是基于梅尔频谱图训练的。具体而言，音频波形首先被转换为二维梅尔频谱图表示，然后利用预训练的VAE将该频谱图编码到潜在空间中。对于特征提取，我们分别利用预训练的CLIP[70]和T5[71]编码器独立提取视觉和文本特征。随后，这些特征通过独立的线性映射并经过SwiGLU激活投射到与DiT兼容的潜在空间中，如图18所示。为有效融合多模态信息，我们引入了堆叠的三流变换器块，分别独立处理视觉、音频和文本模态，之后接入单流变换器块，以确保各模态之间的无缝融合和对齐。这种设计增强了音视频及音频文本表示之间的对齐，促进多模态一致性的提升。当扩散变换器生成潜在表示后，VAE解码器重构对应的梅尔频谱图，最终利用预训练的HifiGAN声码器[44]将梅尔频谱图转换回音频波形。该框架确保了音频信号的高保真重构，同时保持了良好的多模态对齐效果。

# 7.2 混元图像到视频

# 7.2.1 预训练

![](images/23.jpg)  

Figure 19: Hunyuan Video-I2V Diffusion Backbone.

图像到视频（Image-to-video, I2V）任务是视频生成任务中的一个常见应用。通常指的是给定一张图像和一段字幕，模型以该图像作为首帧，生成与字幕内容相匹配的视频。尽管原始的混元视频（HunyuanVideo）模型是一个文本到视频（Text-to-video, T2V）模型，但它可以轻松扩展为I2V模型。如图19所示，I2V模型采用词元替换技术，辅助模型更准确地在输出中重建原始图像信息。参考图像的潜变量直接用作首帧潜变量，相应的时间步设为0。其他帧潜变量的处理方式与T2V训练保持一致。为了增强模型理解输入图像语义的能力，更有效地融合图像和字幕的信息，I2V模型引入了语义图像注入模块。该模块先将图像输入多模态大模型（MLLM）以获取语义图像词元，随后将这些词元拼接进视频潜变量词元中进行全注意力计算。我们在与T2V模型相同的数据上对I2V模型进行了预训练，结果如图20所示。

# 7.2.2 下游任务微调：人像图像到视频生成

我们在两百万张人像视频上对I2V模型进行有监督微调，以提升人体动作表现和整体美学效果。除了第三节中描述的标准数据过滤流程外，我们还应用人脸和人体检测器，过滤掉含有超过五人的训练视频。同时，我们剔除主体较小的视频。最后，对剩余视频进行人工审查，以获得最终高质量的人像训练数据集。在训练过程中，我们采用渐进式微调策略，逐步解冻各层模型参数，其余部分保持冻结状态。该方法使模型在保证固有泛化能力的前提下，在人像领域取得较高性能，确保在自然风景、动物和植物领域也有良好表现。此外，我们的模型还支持基于首尾帧作为条件的视频插帧。训练时我们以一定概率随机丢弃文本条件，以增强模型性能。部分示例结果见图21。

![](images/24.jpg)  

Figure 20: Sample results of the I2V pre-training model.

![](images/25.jpg)  

Figure 21: Sample results of our portrait I2V model.

# 7.3 虚拟形象动画

混元视频在多方面赋能可控的虚拟人动画。它支持使用明确的驱动信号（如语音信号、表情模板和姿态模板）来驱动角色动画。此外，还集成了基于文本提示的隐式驱动范式。图22展示了我们如何利用混元视频的能力，从多模态条件驱动角色动画。为了保持严格的外观一致性，我们通过在架构中插入参考图像的潜变量作为强指导来修改混元视频结构。如图22(b, c)所示，我们使用3DVAE对参考图像进行编码，得到潜变量 \( z_{\mathrm{ref}} \in \mathbb{R}^{1 \times c \times h \times w} \)，其中 \( c=16 \)。然后沿时间维度将其重复 \( t \) 次，并在通道维度与 \( z_t \) 连接，得到修改后的噪声输入 \(\hat{z}_t \in \mathbb{R}^{t \times 2c \times h \times w}\)。以下将详细介绍该可控动画框架及其各类适配器的设计与应用。

![](images/26.jpg)  

Figure 22: Overview of Avatar Animation built on top of HunyuanVideo. We adopt 3D VAE to encode and inject reference and pose condition, and use additional cross-attention layers to inject audio and expression signals. Masks are employed to explicitly guide where they are affecting.

# 7.3.1 上半身会说话虚拟形象生成

近年来，音频驱动的数字人算法取得了显著进展，尤其是在说话头部表现方面。早期的算法，如loopy [94]、emo [80]和hallo [87]，主要集中在头部区域，通过分析音频信号驱动数字人的面部表情和唇形。更早的算法，如wav2lip [68]和DINet [97]，则侧重于修改输入视频中的口部区域，以实现唇形与音频的一致性。然而，这些算法通常仅限于头部区域，忽略了身体其他部位。为了实现更自然、生动的数字人表现，我们提出了一种延展至上半身的音频驱动算法。在该算法中，数字人在说话时不仅同步面部表情和唇形，还能随着音频节奏有节奏地移动身体。

基于音频驱动 根据输入的音频信号，我们的模型能够自适应地预测数字人脸部表情和姿态动作信息。这使得驱动角色能够带有情感和表情地说话，增强数字人的表现力和真实感。如图22(b)所示，对于单音频信号驱动部分，音频经过 Whisper 特征提取模块以获取音频特征，然后通过交叉注意力机制注入主网络。需要注意的是，注入过程会乘以面部掩码，以控制音频的作用区域。在增强头部和肩部控制能力的同时，也大大降低了身体变形的概率。为了获得更生动的头部动作，引入了头部姿态运动参数和表情运动参数，并以嵌入的方式加入时间步中。在训练过程中，头部运动参数由鼻尖关键点序列的方差给出，表情参数由面部关键点的方差给出。

# 7.3.2 全控型全身虚拟形象生成

显式控制数字角色的动作和表情一直是学术界和产业界的长期难题，近期扩散模型的发展迈出了实现逼真虚拟形象动画的第一步。然而，由于基础视频生成模型能力有限，当前的虚拟形象动画方案存在可控性不足的问题。我们证明了更强大的文本到视频（T2V）模型能够推动虚拟形象视频生成达到完全可控的阶段。我们展示了鸿元视频（Hunyuan Video）如何作为强大的基础，在有限改动下将通用文本到视频模型扩展为完全可控的虚拟形象生成模型，如图22(c)所示。基于姿态驱动，我们可以使用姿态模板显式控制数字角色的身体动作。我们采用Dwpose [92]从任意源视频中检测骨骼视频，并使用3DVAE将其映射到潜在空间表示为 $z_{\mathrm{pose}}$。我们认为这简化了微调过程，因为输入视频与驱动视频均为图像表示，且使用共享的变分自编码器（VAE）进行编码，最终映射到相同的潜在空间。随后，我们通过逐元素相加的方式将驱动信号注入模型，形式为 $\hat{z}_t + z_{\mathrm{pose}}$。需要注意的是，$\hat{z}_t$包含了参考图像的外观信息。我们采用全参数微调方法，以上预训练的文本到视频权重作为初始化。

![](images/27.jpg)  

Figure 23: Audio-Driven. HunyuanVideo can generate vivid talking avatar videos.

表情驱动 我们还可以使用隐式表情表示来控制数字角色的面部表情。虽然面部关键点在该领域被广泛采用[58, 16]，但我们认为使用关键点会导致身份信息泄露，原因是跨身份的关键点未对齐。相反，我们使用隐式表示作为驱动信号，利用其身份与表情的解耦能力。在本工作中，我们使用VASA [88] 作为表情提取器。如图22(c)所示，我们采用轻量级表情编码器将表情表示转换为潜在空间中的词元序列，记为 $\bar{z}_{\mathrm{exp}} \in \mathbb{R}^{t \times n \times c}$，其中 $n$ 是每帧的词元数量，通常设置为 $n=16$。与姿态条件不同的是，我们使用交叉注意力注入 $z_{\mathrm{exp}}$，因为 $\hat{z}_t$ 和 $z_{\mathrm{exp}}$ 在空间上并非自然对齐。我们在每 $K$ 层双流和单流DiT层后加入交叉注意力层 $\mathrm{Attn}_{\mathrm{exp}}(q,k,v)$ 以注入表情潜变量。设第 $i$ 层DiT层后的隐状态为 $h_i$，则向 $h_i$ 注入表情 $z_{\mathrm{exp}}$ 的公式为： $$h_i + \mathrm{Attn}_{\mathrm{exp}}(h_i, z_{\mathrm{exp}}, z_{\mathrm{exp}}) * \mathcal{M}_{\mathrm{face}}$$ 其中 $\mathcal{M}_{\mathrm{face}}$ 为面部区域掩码，用于指导 $z_{\mathrm{exp}}$ 应用于何处，$^*$ 表示元素乘。并且采用全参数微调策略。 混合条件驱动 将姿态与表情驱动策略结合，得到混合控制方法。在此场景下，身体动作由显式骨骼姿态序列控制，面部表情由隐式表情表示决定。我们以端到端方式联合微调文本到视频（T2V）模块及新增模块。推理时，身体动作和面部动作可以由不同的驱动信号分别控制，从而实现更丰富的编辑能力。

# 7.4 应用演示

我们展示了大量虚拟形象动画的实验结果，以体现由混元视频驱动的虚拟形象动画在下一代技术中的优势与潜力。

![](images/28.jpg)  

Figure 24: Pose-Driven. HunyuanVideo can animate wide variety of characters with high quality and appearance consistency under various poses.

![](images/29.jpg)  

Figure 25: Expression-Driven. HunyuanVideo can accurately control facial movements of widevariety of avatar styles.

音频驱动 图23显示，混元视频作为音频驱动头像动画的强大基础模型，能够合成生动且高保真的视频。我们总结了本方法的三大优势： • 上半身动画。本方法不仅可以驱动肖像角色，还能驱动上半身头像图像，拓展了其应用场景的范围。 • 动态场景建模。本方法能够生成具有生动且真实背景运动的视频，如波浪起伏、群体运动和微风吹动树叶。 • 生动的头像动作。本方法能够仅凭音频驱动角色在说话时做出生动的手势动作。 基于姿态驱动 我们还展示了混元视频在多方面显著提升了基于姿态驱动的动画性能，如图24所示：

![](images/30.jpg)  

Figure 26: Hybrid Condition-Driven. Hunyuan Video supports full control with multiple driving sources across various avatar characters.

• 高身份一致性。即使在大幅度姿势变化下，我们的方法也能很好地保持身份一致性，实现无换脸效果，因此可用作真正的端到端动画解决方案。 • 精准跟踪复杂姿势。我们的方法能够处理诸如转身和双手交叉等非常复杂的姿势。 • 高运动质量。我们的方法在动态建模方面表现出色，例如在服装动态和纹理一致性方面展现出良好性能。 • 优秀的泛化能力。我们的方法具有惊人的泛化能力，能够驱动各种头像图像的动画，如真实人体、动漫、陶瓷雕像甚至动物。 表情驱动 图25展示了混元视频（HunyuanVideo）在头像表情动画上的三倍提升： • 夸张表情。我们的方法能够驱动给定头像模拟任意面部动作，即使是大幅度姿势和夸张表情也能精准还原。 • 精准模拟眼球注视。我们可基于任意表情模板精准控制头像的眼球运动，甚至包括极端和大幅度的眼球转动。 • 泛化能力。我们的方法具备很高的泛化能力，不仅能驱动真实人像，还可应用于动漫或CGI角色。 混合驱动 最后，我们展示了图26中混合条件控制揭示的全可控可编辑头像的潜力。其优势如下： • 混合条件控制。我们的方法首次实现了通过独立或多重信号对身体和面部动作的全面控制，开辟了从演示到应用的头像动画之路。 • 半身动画。我们的方法支持上半身的完整控制，既能实现丰富的编辑功能，又能保持高质量和高保真度。 • 泛化能力。我们的方法能广泛适用于真实人体图像和CGI角色。

# 8 相关工作

由于扩散模型在图像生成领域的成功[72, 34]，视频生成领域的探索[28, 40, 75, 84, 91, 96, 58, 12, 90, 57]也日益受到关注。VDM[32]是最早将图像扩散模型中的二维U-Net扩展为三维U-Net以实现基于文本的视频生成的工作之一。后续工作如MagicVideo[103]和Mindscope[82]引入了一维时间注意力机制，基于潜空间扩散模型减少了计算量。在本报告中，我们没有采用二维+一维时间模块的方式进行运动学习，而是采用了与FLUX[47]中类似的双流注意力模块，用于处理所有视频帧。继Imagen之后，Imagen Video[33]采用了级联采样流程，通过多阶段生成视频。除了传统的端到端文本到视频（T2V）生成外，基于其他条件的视频生成也是一个重要方向。这类方法利用深度图[27, 31]、姿态图[89, 37, 83, 56]、RGB图像[5, 15, 61]或其他受控运动视频[100, 86]等辅助控制生成视频。尽管最近开源模型如Stable video diffusion[5]、Open-sora[102]、Open-sora-plan[46]、Mochi-1[79]和Allegro[104]在生成性能上表现优异，但其性能仍远落后于闭源的最先进视频生成模型，如Sora[7]和MovieGen[67]。

# 项目贡献者

项目赞助人：江杰，刘玉红，王迪，杨勇 项目负责人：钟凯撒，王洪发，周达祥，刘松涛，陆庆林，陶阳宇 核心贡献者： 基础设施：民若克，薛锦宝，彭元博，杨芳，李帅，王维艳，王凯 数据与重标注：戴作卓，李鑫，周晋，袁均坤，谭昊，邓新驰，何志宇，黄多俊，王安东，刘萌洋，李鹏宇 VAE与模型蒸馏：吴波，民若克，李昌林，白佳望，李扬，吴建兵 算法、模型架构与预训练：孔伟杰，田琦，张建伟，张子建，吴卡斯琳娜，熊江峰，龙彦鑫 下游任务：宋杰博，周晋，崔宇涛，王阿拉丁，余文清，徐志勇，周子翔，于振涛，陈奕，王红梅，徐祖南，王乔伊，林秦 贡献者：张继宏，陈猛，朱建辰，胡温斯顿，饶永明，刘凯，徐丽飞，林思桓，孙逸夫，黄世锐，牛林，黄世胜，邓永军，曹凯博，杨轩，张昊，林佳欣，张超，游飞，陈源斌，胡玉辉，董亮，方铮，焦典，徐志江，任旭华，马兵，程佳祥，李文岳，俞凯，郑天祥

参考文献 [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat 等. GPT-4 技术报告. arXiv 预印本 arXiv:2303.08774, 2023. 9 [2] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen 等. PaLM 2 技术报告. arXiv 预印本 arXiv:2305.10403, 2023. 9 [3] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, Jun Zhu. All are worth words: 基于 ViT 主干网络的扩散模型. IEEE/CVF 计算机视觉与模式识别大会论文集, 页码 22669-22679, 2023. 9 [4] Jmes Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo 等. 通过更优质的字幕提升图像生成. 计算机科学. https://cdn.openai.com/papers/dall-e-3.pdf, 2(3):8, 2023. 4 [5] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts 等. 稳定视频扩散：将潜空间视频扩散模型扩展至大规模数据集. arXiv 预印本, 2023. 2, 27 [6] Andrew Brock, Jeff Donahue, Karen Simonyan. 大规模 GAN 训练用于高保真自然图像合成. arXiv 预印本 arXiv:1809.11096, 2018. 2 [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Wing Yin Ng, Ricky Wang, Aditya Ramesh. 视频生成模型作为世界模拟器. 2024. 4, 7, 27 [8] Tom B Brown. 语言模型是少样本学习者. arXiv 预印本 arXiv:2005.14165, 2020. 8 [9] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, Dahua Lin. ShareGPT4V：通过更优质字幕提升大规模多模态模型. arXiv 预印本 arXiv:2311.12793, 2023. 4 [10] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang 等. ShareGPT4Video：通过更优质字幕提升视频理解与生成能力. arXiv 预印本 arXiv:2406.04325, 2024. 4 [11] Liuhan Chen, Zongjian Li, Bin Lin, Bin Zhu, Qian Wang, Shenghai Yuan, Xing Zhou, Xinghua Cheng, Li Yuan. OD-VAE：一种提升潜空间视频扩散模型的视频全维压缩器. arXiv 预印本 arXiv:2409.01199, 2024. 6 [12] Qihua Chen, Yue Ma, Hongfa Wang, Junkun Yuan, Wenzhe Zhao, Qi Tian, Hongmei Wang, Shaobo Min, Qifeng Chen, Wei Liu. Follow-your-canvas：基于大规模内容生成的高分辨率视频外扩. arXiv 预印本 arXiv:2409.01055, 2024. 27 [14] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-Wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, Sergey Tulyakov. Panda·70M：基于多模态教师的 7000 万视频字幕生成. 2024 IEEE/CVF 计算机视觉与模式识别大会（CVPR）, 页码 13320-13331, IEEE, 2024.6. 4 [15] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, Ziwei Liu. Seine：面向生成式过渡与预测的短至长视频扩散模型. arXiv 预印本, 2023. 27 [16] Zhiyuan Chen, Jiajiong Cao, Zhiquan Chen, Yuming Li, Chenguang Ma. EchoMimic：基于可编辑地标条件的拟音驱动肖像动画. arXiv 预印本 arXiv:2407.08136, 2024. 23 [17] XTuner 贡献者. XTuner：高效微调大语言模型的工具包. https://github.com/InternLM/xtuner, 2023. 8 [18] OpenCV 开发者. OpenCV. https://opencv.org/. 3 [19] PySceneDetect 开发者. PySceneDetect. https://www.scenedetect.com/. 3 [20] SeungHeon Doh, Keunwoo Choi, Jongpil Lee, Juhan Nam. LP-MusicCaps：基于大语言模型的伪音乐字幕生成. arXiv 预印本 arXiv:2307.16372, 2023. 19 [21] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel 等. 扩展整流流变换器以实现高分辨率图像合成. 第四十一届国际机器学习大会论文集, 2024. 2, 8, 9, 10 [22] Patrick Esser, Robin Rombach, Bjorn Ommer. 驯服变换器以实现高分辨率图像合成. IEEE/CVF 计算机视觉与模式识别大会论文集, 页码 12873-12883, 2021. 6 [23] Peng Gao, Le Zhuo, Ziyi Lin, Chris Liu, Junsong Chen, Ruoyi Du, Enze Xie, Xu Luo, Longtian Qiu, Yuhang Zhang 等. Lumina-T2X：基于流式大规模扩散变换器的多模态文本转换模型（支持任意模态、分辨率与时长）. arXiv 预印本 arXiv:2405.05945, 2024. 9 [24] Z Ge. YOLOX：2021 年超越 YOLO 系列的新一代目标检测. arXiv 预印本 arXiv:2107.08430, 2021. 3 [25] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, Ishan Misra. EMU Video：通过明确图像条件分解文本到视频生成. arXiv 预印本 arXiv:2311.10709, 2023. 2

[26] GLM团队，曾奥涵，许宾，王博文，张陈辉，尹达，Diego Rojas，冯冠宇，赵涵霖，赖涵宇，余昊，王宏宁，孙家岱，张佳杰，程嘉乐，桂嘉怡，唐捷，张景，李娟滋，赵磊，吴林东，钟露岑，刘明道，黄敏烈，张鹏，郑琴凯，卢瑞，段帅奇，张书丹，曹淑霖，杨书勋，谭永林，赵文逸，刘晓，夏潇，张晓涵，顾晓涛，吕欣，刘兴涵，刘欣怡，杨心悦，宋熙萱，张寻凯，安一凡，许一凡，钮意林，杨元涛，李越岩，白玉石，董宇霄，齐泽涵，王朝宇，杨震，杜正孝，侯振宇，王子涵。《Chatglm：从GLM-130B到GLM-4 All Tools的一系列大语言模型》，2024。8 [27] 郭宇伟，杨策远，饶安毅，Maneesh Agrawala，林大华，戴博。《SparseCtrl：为文本到视频扩散模型添加稀疏控制》，arXiv预印本，2023。27 [28] 郭宇伟，杨策远，饶安毅，梁正阳，王耀辉，乔宇，Maneesh Agrawala，林大华，戴博。《AnimateDiff：无需特定微调即可驱动个性化文本到图像扩散模型动画化》，ICLR，2024。27 [29] Moayed Haji-Ali，Willi Menapace，Aliaksandr Siarohin，Guha Balakrishnan，Sergey Tulyakov，Vicente Ordonez。《驾驭音频生成中的数据与变换器》，arXiv预印本 arXiv:2406.19388，2024。19 [30] Pieter Abbeel，刘浩，Matei Zaharia。《基于块的变换器的环形注意力实现几乎无限上下文》，arXiv预印本 arXiv:2310.01889，2023。14 [31] 何英清，夏孟涵，陈昊鑫，存晓东，龚源，邢进波，张永涛，翁超，单颀，单燕，等。《Animate-a-Story：结合检索增强的视频生成讲故事》，arXiv预印本，2023。27 [32] Jonathan Ho，T Salimans，A Gritsenko，W Chan，M Norouzi，DJ Fleet。《视频扩散模型》，arXiv预印本，2022。27 [33] Jonathan Ho，William Chan，Chitwan Saharia，Jay Whang，Ruiqi Gao，Alexey Gritsenko，Diederik P Kingma，Ben Poole，Mohammad Norouzi，David J Fleet，等。《Imagen Video：基于扩散模型的高清视频生成》，arXiv预印本，2022。27 [34] Jonathan Ho，Ajay Jain，Pieter Abbeel。《去噪扩散概率模型》，NeurIPS，2020。10，27 [35] Jonathan Ho，Tim Salimans。《无分类器的扩散引导》，arXiv预印本，2022。13 [36] Jordan Hoffmann，Sebastian Borgeaud，Arthur Mensch，Elena Buchatskaya，Trevor Cai，Eliza Rutherford，Diego de Las Casas，Lisa Anne Hendricks，Johannes Welbl，Aidan Clark，等。《训练计算最优的大规模语言模型》，arXiv预印本 arXiv:2203.15556，2022。8，9，10 [37] 胡力，郜新，张鹏，孙科，张邦，薄烈风。《Animate Anyone：面向角色动画的一致性与可控性图像到视频合成》，arXiv预印本，2023。27 [38] 洪云宁，吴志伟，Iroro Orife，Aaron Hipple，William Wolcott，Alexander Lerch。《大型电视数据集用于语音与音乐活动检测》，《EURASIP音频、语音与音乐处理期刊》，2022(1):21，2022。18 [39] Investopedia。《通用数据保护条例（GDPR）》，无日期。访问日期：2023年10月10日。3 [40] 姜育明，杨帅，柯同良，吴伟，刘晨曌，刘子威。《Text2Performer：文本驱动的人类视频生成》，arXiv预印本，2023。27 [41] Jared Kaplan，Sam McCandlish，Tom Henighan，Tom B Brown，Benjamin Chess，Rewon Child，Scott Gray，Alec Radford，Jeffrey Wu，Dario Amodei。《神经语言模型的扩展规律》，arXiv预印本 arXiv:2001.08361，2020。8，9 [42] Maciej Kilian，Varun Japan，Luke Zettlemoyer。《图像合成中的计算权衡：扩散、掩码词元与下一个词元预测》，arXiv预印本 arXiv:2405.13218，2024。9 [43] Juyeon Kim，Jeongeun Lee，Yoonho Chang，Chanyeol Choi，Junseong Kim，Jy Yong Sohn。《Re-ex：解释后修正减少大语言模型回答的事实错误》，2024。12 [44] Jungil Kong，Jaehyeon Kim，Jaekyoung Bae。《HiFi-GAN：高效高保真语音合成的生成对抗网络》，《神经信息处理系统进展》，33：17022–17033，2020。19 [45] Vijay Anand Korthikanti，Jared Casper，Sangkug Lym，Lawrence McAfee，Michael Andersch，Mohammad Shoeybi，Bryan Catanzaro。《减少大规模变换器模型中的激活重计算》，《机器学习与系统会议论文集》，5：341–353，2023。13 [46] 北京大学元实验室及土砖AI等。《Open-Sora计划》，2024年4月。27 [47] Black Forest Labs。《Flux》，2024。2，6，8，27 [48] 李宝佳，王晓亮，王静竹，刘一凡，龚媛媛，卢昊，党维真，张伟峰，黄晓杰，陈明卓，等。《TCCL：GPU中心集群的集合通信与网络流量联合优化》，见2024年SIGCOMM人工智能计算网络研讨会论文集，第48–53页，2024。13 [49] 李昊，邹阳，王莹，Orchid Majumder，谢玉胜，R Manmatha，Ashwin Swaminathan，涂卓文，Stefano Ermon，Stefano Soatto。《基于扩散的文本到图像生成的可扩展性研究》，IEEE/CVF计算机视觉与模式识别会议论文集，第9400–9409页，2024。9 [50] 李俊南，李东旭，熊彩明，Steven Hoi。《BLIP：用于统一视觉语言理解与生成的语言图像预训练引导》，国际机器学习大会论文集，第12888–12900页，PMLR，2022。4 [51] 李志敏，张建伟，林沁，熊江锋，龙彦欣，邓新驰，张英芳，刘兴超，黄民斌，肖泽东，陈大友，何佳俊，李嘉昊，李文越，张晨，权荣伟，卢建祥，黄嘉斌，袁晓燕，郑潇潇，李依轩，张继宏，张超，陈萌，刘杰，方征，王蔚岩，薛锦宝，陶阳雨，朱建陈，刘凯，林思桓，孙逸夫，李云，王冬冬，陈明涛，胡志超，肖晓晓，陈燕，刘玉红，刘伟，王迪，杨勇，姜捷，陆晴林。《混元-迪特：具备细粒度中文理解的多分辨率扩散变换器》，2024。2，8 [52] Yaron Lipman，Ricky TQ Chen，Heli Ben-Hamu，Maximilian Nickel，Matt Le。《生成建模的流匹配》，arXiv预印本 arXiv:2210.02747，2022。2，10 [53] 刘昊天，李春元，吴庆阳，李勇宰。《视觉指令微调》，神经信息处理系统进展，36，2024。8 [54] 罗思言，闫传浩，胡晨旭，赵航。《Diff-Foley：基于潜在扩散模型的同步视频到音频合成》，神经信息处理系统进展，36，2024。19 [55] 马炳奇，宗卓凡，宋光禄，李宏升，刘宇。《探索大语言模型在扩散模型提示编码中的作用》，arXiv预印本 arXiv:2406.11831，2024。8 [56] 马越，何英清，存晓东，翁超，单燕，李秀，陈祁峰。《Follow Your Pose：基于姿态引导的文本到视频生成，无需姿态视频》，arXiv预印本，2023。27 [57] 马越，何英清，王洪发，王安东，齐晨阳，蔡承飞，李秀，李志峰，岑香扬，刘伟，等。《Follow-Your-Click：开放域局部图像动画生成基于简短提示》，arXiv预印本 arXiv:2403.08268，2024。27 [58] 马越，刘洪宇，王洪发，潘恒，何英清，袁君坤，曾艾玲，蔡承飞，岑香扬，刘伟，等。《Follow-Your-Emoji：可精细控制且富表现力的自由风格人像动画》，arXiv预印本 arXiv:2406.01900，2024。23，27 [59] J MacQueen。《多元观测分类与分析方法》，《第5届伯克利数理统计与概率研讨会论文集》，加州大学出版社，1967。3 [60] 孟晨林，Robin Rombach，高瑞奇，Diederik Kingma，Stefano Ermon，Jonathan Ho，Tim Salimans。《引导扩散模型蒸馏研究》，IEEE/CVF计算机视觉与模式识别会议论文集，第14297–14306页，2023。13 [61] 倪浩淼，史长浩，李凯，黄晓欣，Min Martin Renqiang。《条件图像到视频生成的潜流扩散模型》，CVPR，2023。27 [62] 聂晓楠，刘毅，付方成，薛锦宝，焦点，苗旭鹏，陶阳雨，崔斌。《Angel-PTM：腾讯可扩展且经济的大规模预训练系统》，arXiv预印本 arXiv:2303.02868，2023。13 [63] NVIDIA。《上下文并行概览》，2024。13 [64] NVIDIA。《Cosmos-Tokenizer》，2024。6 [65] William Peebles，谢赛宁。《具有变换器的可扩展扩散模型》，IEEE/CVF国际计算机视觉大会论文集，第4195–4205页，2023。2，9 [66] Dustin Podell，Zion English，Kyle Lacey，Andreas Blattmann，Tim Dockhorn，Jonas Müller，Joe Penna，Robin Rombach。《SDXL：改进潜在扩散模型以实现高分辨率图像合成》，arXiv预印本，2023。8，11 [67] Adam Polyak，Amit Zohar，Andrew Brown，Andros Tjandra，Animesh Sinha，Ann Lee，Apoorv Vyas，Bowen Shi，Chih-Yao Ma，Ching-Yao Chuang， 等。《MovieGen：一套媒体基础模型》，arXiv预印本 arXiv:2410.13720，2024。2，5，6，7，13，19，27 [68] KR Prajwal，Rudrabha Mukhopadhyay，Vinay P Namboodiri，CV Jawahar。《野外语音到唇形生成只需唇形同步专家》，第28届ACM国际多媒体大会论文集，第484–492页，2020。22 [69] Alec Radford，Jong Wook Kim，Chris Hallacy，Aditya Ramesh，Gabriel Goh，Sandhini Agarwal，Girish Sastry，Amanda Askell，Pamela Mishkin，Jack Clark，等。《基于自然语言监督的可迁移视觉模型学习》，ICML，2021。8 [70] Alec Radford，Jong Wook Kim，Chris Hallacy，Aditya Ramesh，Gabriel Goh，Sandhini Agarwal，Girish Sastry，Amanda Askell，Pamela Mishkin，Jack Clark，等。《基于自然语言监督的可迁移视觉模型学习》，ICML论文集，第8748–8763页，PMLR，2021。19 [71] Colin Raffel，Noam Shazeer，Adam Roberts，Katherine Lee，Sharan Narang，Michael Matena，Yanqi Zhou，Wei Li，Peter J Liu。《探索统一文本到文本变换器的迁移学习极限》，《机器学习研究杂志》，21(140):167，2020。8，10，19 [72] Robin Rombach，Andreas Blattmann，Dominik Lorenz，Patrick Esser，Björn Ommer。《基于潜在扩散模型的高分辨率图像合成》，CVPR，2022。2，27 [73] Tim Salimans，Jonathan Ho。《扩散模型快速采样的渐进蒸馏》，arXiv预印本 arXiv:2202.00512，2022。10 [74] Mohammad Shoeybi，Mostofa Patwary，Raul Puri，Patrick LeGresley，Jared Casper，Bryan Catanzaro。《Megatron-LM：利用模型并行训练数十亿参数语言模型》，arXiv预印本 arXiv:1909.08053，2019。13 [75] Uriel Singer，Adam Polyak，Thomas Hayes，Xi Yin，Jie An，Songyang Zhang，Qiyuan Hu，Harry Yang，Oron Ashual，Oran Gafni，等。《Make-A-Video：无需文本视频数据的文本到视频生成》，arXiv预印本，2022。27 [76] Tomá Souek，Jakub Loko。《TransNet v2：用于快速镜头切换检测的高效深度网络结构》，arXiv预印本 arXiv:2008.04838，2020。3 [77] 苏建林，陆煜，潘胜锋，Ahmed Murtadha，温博，刘云峰。《RoFormer：增强型带旋转位置编码的变换器》，2023。8

[78] 孙星武，陈彦峰，黄奕青，谢若冰，朱嘉祺，张凯，李帅鹏，杨震，韩仲伟，舒晓波，卜家豪，陈中智，黄学梅，连凤宗，杨赛勇，严建峰，曾育源，任晓琴，余超，吴璐璐，毛越，杨涛，郑孙聪，吴侃，焦典，薛金宝，张希鹏，吴德成，刘凯，吴登鹏，徐光辉，陈少华，陈爽，冯晓，洪怡更，郑军强，许成成，李宗伟，匡雄，胡江路，陈奕琦，邓宇池，李贵阳，刘奥，张辰辰，胡仕辉，赵子龙，吴子凡，丁尧，王伟超，刘涵，Roberts Wang，费昊，奢培杰，赵泽，曹勋，王海，向福生，黄梦远，熊志远，胡斌，侯学斌，姜雷，吴佳佳，邓亚平，沈怡，王倩，刘伟杰，刘杰，陈萌，董亮，贾伟文，陈虎，刘飞飞，袁睿，徐慧林，严振翔，曹腾飞，胡志超，冯新华，杜东，奢亭豪，陶阳宇，张锋，朱建晨，许成忠，李熙睿，查崇，欧阳文，夏银本，李翔，何泽坤，陈荣鹏，宋嘉伟，陈锐斌，姜凡，赵重庆，王博，龚昊，甘荣，胡林斯顿，康占辉，杨勇，刘玉鸿，王迪，姜杰。《混元-large：腾讯开源的52亿激活参数的专家模型》，2024年。8, 11 [79] Genmo团队。《Mochi 1：开源视频生成的新先进水平》。https://github.com/genmoai/models，2024年。7, 27 [80] 田林睿，王奇，张邦，薄烈锋。《EMO：Emote Portrait Alive ——基于音频到视频扩散模型在弱条件下生成具有表现力的人像视频》，2024年。22 [81] Hugo Touvron，Thibaut Lavril，Gautier Izacard，Xavier Martinet，Marie-Anne Lachaux，Timothée Lacroix，Baptiste Rozière，Naman Goyal，Eric Hambro，Faisal Azhar 等。《LLaMA：开放高效的基础语言模型》，arXiv预印本 arXiv:2302.13971，2023年。9 [82] 王九牛，袁杭杰，陈大有，张颖雅，王翔，张世伟。《ModelScope文本到视频技术报告》，arXiv预印本，2023年。27

[83] Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu 和 Lijuan Wang. Disco：真实世界中的指向性人体舞蹈生成的解耦控制。arXiv 预印本，2023年。27 [84] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang 等。Lavie：基于级联潜在扩散模型的高质量视频生成。arXiv 预印本，2023年。27 [85] Haoning Wu, Erli Zhang, Liang Liao, Chaofeng Chen, Jingwen Hou, Annan Wang, Wenxiu Sun, Qiong Yan 和 Weisi Lin. 从美学和技术视角探索用户生成内容的视频质量评估。发表于 IEEE/CVF 国际计算机视觉大会论文集，第20144-20154页，2023年。3 [86] Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li 和 Xiangyu Zhang. LAMP：基于少样本学习的视频生成运动模式学习。arXiv 预印本，2023年。27 [87] Mingwang Xu, Hui Li, Qingkun Su, Hanlin Shang, Liwei Zhang, Ce Liu, Jingdong Wang, Yao Yao 和 Siyu Zhu. HALLO：基于层级音频驱动的人像动画视觉合成，2024年。22 [88] Sicheng Xu, Guojun Chen, Yu-Xiao Guo, Jiaolong Yang, Chong Li, Zhenyu Zang, Yizhong Zhang, Xin Tong 和 Baining Guo. VASA-1：实时生成逼真音频驱动的口型动画。arXiv 预印本 arXiv:2404.10667，2024年。23 [89] Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng 和 Mike Zheng Shou. MAGICANIMATE：基于扩散模型的时序一致人体图像动画。arXiv 预印本，2023年。27 [90] Jingyun Xue, Hongfa Wang, Qi Tian, Yue Ma, Andong Wang, Zhiyuan Zhao, Shaobo Min, Wenzhe Zhao, Kaihao Zhang, Heung-Yeung Shum 等。Follow-your-pose v2：多条件引导的人物图像动画实现稳定姿态控制。arXiv 预印本 arXiv:2406.03035，2024年。27 [91] Mengjiao Yang, Yilun Du, Bo Dai, Dale Schuurmans, Joshua B Tenenbaum 和 Pieter Abbeel. 文本到视频模型的概率适应。arXiv 预印本，2023年。27 [92] Zhendong Yang, Ailing Zeng, Chun Yuan 和 Yu Li. 基于两阶段蒸馏的高效全身姿态估计。发表于 IEEE/CVF 国际计算机视觉大会论文集，第4210-4220页，2023年。22 [93] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng 等。COGVIDEoX：基于专家变换器的文本到视频扩散模型。arXiv 预印本 arXiv:2408.06072，2024年。4, 5, 6, 7 [94] Zhenhui Ye, Tianyun Zhong, Yi Ren, Ziyue Jiang, Jiawei Huang, Rongjie Huang, Jinglin Liu, Jinzheng He, Chen Zhang, Zehan Wang, Xize Chen, Xiang Yin 和 Zhou Zhao. MIMICTALK：几分钟内模仿个性化且富有表现力的三维口型动画，2024年。22 [95] Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu 等。语言模型胜出，扩散分词器是视觉生成的关键。arXiv 预印本 arXiv:2310.05737，2023年。5 [96] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao 和 Mike Zheng Shou. SHOW-1：融合像素与潜在扩散模型的文本到视频生成。arXiv 预印本，2023年。27 [97] Zhimeng Zhang, Zhipeng Hu, Wenjin Deng, Changjie Fan, Tangjie Lv 和 Yu Ding. DINET：用于高分辨率视频上真实脸部视觉配音的形变修复网络。发表于 AAAI 人工智能会议论文集，第37卷，第3543-3551页，2023年。22 [98] Zijian Zhang, Zhou Zhao 和 Zhijie Lin. 从预训练扩散概率模型中进行无监督表示学习。神经信息处理系统进展，第35卷：22117-22130，2022年。13 [99] Zijian Zhang, Zhou Zhao, Jun Yu 和 Qi Tian. SHIFTDDPMS：通过移动扩散轨迹探索条件扩散模型。发表于 AAAI 人工智能会议论文集，第37卷，第3552-3560页，2023年。13 [100] Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, Weijia Wu, Jussi Keppo 和 Mike Zheng Shou. MOTIONDIRECTOR：文本到视频扩散模型的动作定制。arXiv 预印本，2023年。27 [101] Xuanlei Zhao, Xiaolong Jin, Kai Wang 和 Yang You. 基于金字塔注意力广播的实时视频生成。arXiv 预印本 arXiv:2408.12588，2024年。13 [102] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li 和 Yang You. OPEN-SORA：全民高效视频制作民主化，2024年3月。6, 27 [103] Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu 和 Jiashi Feng. MAGICVIDEO：基于潜在扩散模型的高效视频生成。arXiv 预印本，2023年。27 [104] Yuan Zhou, Qiuyue Wang, Yuxuan Cai 和 Huan Yang. ALLEGRO：解密商业级视频生成黑箱。arXiv 预印本 arXiv:2410.15458，2024年。6, 27