# CosyVoice 2：基于大语言模型的可扩展流式语音合成

杜志豪，王宇轩，陈倩，施贤，吕翔，赵天宇，高志富 杨也欣，高长风，王辉，喻帆，刘华岱，盛正焱 顾悦，邓翀，王文良，张世良，闫志杰，周景仁 阿里巴巴集团，中国 {neo.dzh,sly.zsl}@alibaba-inc.com

###### 摘要

在我们之前的工作中，我们介绍了CosyVoice，这是一种基于监督离散语音词元的多语言语音合成模型。通过采用渐进语义解码与两种流行的生成模型（语言模型和流匹配），CosyVoice在语音上下文学习中展现出了高水平的韵律自然性、内容一致性和说话人相似性。最近，多模态大语言模型（LLMs）取得了显著进展，其中语音合成的响应延迟和实时因素在互动体验中起着关键作用。因此，在本报告中，我们提出了一种改进的流式语音合成模型CosyVoice 2，该模型结合了全面和系统的优化。具体而言，我们引入了有限标量量化，以提高语音词元的码本利用率。对于文本-语音语言模型，我们简化了模型架构，以允许直接使用预训练的大语言模型作为主干。此外，我们开发了一种块感知因果流匹配模型，以支持各种合成场景，使得在单个模型中能够实现流式和非流式合成。通过在大规模多语言数据集上进行训练，CosyVoice 2实现了与人类相当的自然性、最小的响应延迟，以及在流式模式下几乎无损的合成质量。我们邀请读者访问 https://funaudiollm.github.io/cosyvoice2 聆听演示。

## 1 引言

近年来，神经文本转语音（TTS）合成模型因超越传统的拼接法和统计参数方法而备受关注 *[1, 2, 3, 4, 5, 6, 7]*。这些模型在预定义特定说话人上实现了高保真度和自然感。近期研究表明，零-shot TTS 模型能够通过模仿参考语音的音色、韵律和风格，为任何说话人合成语音 *[8]*。除了其上下文学习（ICL）能力外，零-shot TTS 模型还受益于大规模训练数据，合成质量和自然程度几乎与人类语音无法区分。最近的零-shot TTS 模型可以大致分为三类：编解码语言模型、特征扩散模型及其混合系统。编解码语言模型利用语音编解码模型提取离散语音表示 *[9, 10, 11]*，并采用自回归 *[12, 13, 14, 15, 16, 17]* 或掩码 *[18]* 语言模型预测语音词元，然后通过编解码器将其合成波形 *[19, 20]*。连续语音表示在 *[21]* 中也有所探讨。基于语言模型的 TTS 能够通过自回归采样生成多样的、韵律一致的语音。

受到图像生成、去噪扩散*[22, 23]*和流匹配模型*[24]*的进展启发，这些技术已被引入非自回归（NAR）语音合成。早期基于扩散的文本到语音（TTS）模型需要对每个文本（音素）进行时长预测，以解决文本与语音特征之间的长度差异*[25, 26, 27, 28]*。然而，这种刚性的对齐方式可能会影响自然性，导致平坦的韵律。为了解决这个问题，交叉注意力和扩散变换器（DiT）被引入到NAR TTS模型中*[29, 30]*。最近的研究指出了一些更简单的文本-语音对齐方法在NAR TTS模型中，例如E2 TTS*[31]*、F5-TTS*[32]*和Seed-TTS*[33]*。在这些模型中，输入文本通过特殊标记进行填充，以匹配总的语音长度，这一长度要么由发话时长预测模块自动预测，要么由用户提前指定。由于NAR TTS模型不受编码器的限制，它们能够实现优越的语音质量。混合系统结合了文本到编码器语言模型和编码器到特征扩散模型*[34, 35, 33]*。语言模型解决文本与语音之间的对齐问题以及发话时长的预测，而编码器到特征扩散模型则根据生成的编码器和其他条件合成语音特征（梅尔谱）。通过利用两种生成模型的优势，混合系统实现了高多样性、韵律一致性和语音质量。尽管近期的零-shot TTS模型取得了成功，但通常以非流式（离线）模式运行，这需要完整输入文本并在返回波形之前合成整个发话。这导致了高延迟，对语音聊天等应用的用户体验产生了负面影响*[36, 37]*。为了解决这个问题，已探索基于语言模型的零-shot TTS模型的流式合成*[38, 39, 40, 41]*，但基于扩散的TTS模型和混合系统缺乏成熟的流式解决方案。在CosyVoice*[34]*成功的基础上，我们推出了CosyVoice 2，这是一个具有更好韵律自然性、内容一致性和说话者相似性的流式零-shot TTS模型。我们的贡献包括：- 在一个统一框架中整合流式和非流式合成，提出统一的文本-语音语言模型和块感知因果流匹配模型，与离线模式相比，实现了无损流式合成。- 通过去除文本编码器和说话者嵌入，简化了语言模型架构，允许预训练的文本大语言模型（LLMs）作为主干，从而增强上下文理解。- 将语音标记器中的向量量化（VQ）替换为有限标量量化（FSQ），提高了码本利用率并捕捉到了更多语音信息。- 提升了指令TTS能力，以支持更多指令，包括情感、口音、角色风格和细粒度控制。在CosyVoice 2中，指令和零-shot能力整合为一个模型，实现了更通用和生动的合成。通过上述系统的修改和优化，CosyVoice 2实现了人类水平的合成质量，并在流式模式下几乎无损。统一的框架放宽了部署要求，使单一模型能够支持流式和非流式合成。升级的指令TTS能力为用户生成各种语音提供了更强大和更简单的方法。此外，块感知流匹配设计也可以应用于NAR TTS模型，表明流式NAR模型的潜力。

## 2 CosyVoice 2

CosyVoice 2 基于其前身相似的设计哲学，通过将语音信号的语义信息和声学信息分开并独立建模。语音生成过程被重新定义为逐步的语义解码过程，其中条件信息逐步被纳入。具体而言，文本-语音语言模型 (LM) 仅关注语义信息，将高级文本词元解码为监督的语义语音词元。在流匹配模型中，通过说话者嵌入和参考语音引入声学细节，如音色，将语音词元转换为特定说话者的梅尔频谱。最后，一个预训练的声码器模型恢复相位，将梅尔频谱转换回原始音频信号。以下部分将从五个方面介绍 CosyVoice 2 的细节及流式合成的修改：文本分词器、监督语义语音分词器、用于流式/非流式合成的统一文本-语音 LM 和块感知流匹配模型。图 1 提供了 CosyVoice 2 的概述。

![img-0.jpeg](images/1.jpeg)

Figure 1: An overview of CosyVoice 2. (a) demonstrates the supervised speech tokenizer, where dashed modules are only used at the training stage. (b) is a unified text-speech language model for streaming and non-streaming synthesis. Dashed lines indicate the autoregressive decoding at the inference stage. (c) illustrates the causal flow matching model conditioning on a speaker embedding  $\mathbf{v}$ , semantic tokens  $\mu$ , masked speech features  $\bar{X}$  and intermediate state  $X_{t}$  at timestep  $t$  on the probabilistic density path.

# 2.1 文本词元化器

CosyVoice 2 直接使用原始文本作为输入，通过基于 BPE 的文本分词器进行分词。这消除了需要通过图形到音素（g2p）转换获取音素的前端模型。此方法不仅简化了数据预处理工作流，还使模型能够以端到端的方式学习在各种上下文中单词的发音。与常用于文本大型语言模型的分词器不同，CosyVoice 2 会屏蔽一对多的词元。这防止了一个词元的发音过长，并减少了由数据稀疏性引起的特殊情况。具体来说，如果一个 BPE 词元编码了多个汉字，它将被屏蔽，而每个汉字将在分词过程中单独编码。其他语言，如英语、日语和韩语，则不需要特殊处理。

# 2.2 监督语义语音词元处理器

如图1（a）所示，我们将有限标量量化（FSQ）模块 [42] 插入到 SenseVoice-Large ASR 模型的编码器中 [43]。在训练阶段，输入语音 $X$ 通过 $\mathrm{Encoder}_1$ 得到中间表示，其中 $\mathrm{Encoder}_1$ 由六个带有旋转位置嵌入的 Transformer 块组成 [44]。随后，中间表示被送入 FSQ 模块进行量化，量化后的表示通过其余的 SenseVoice-Large 模块，包括 $\mathrm{Encoder}_2$ 和 ASR 解码器，预测相应文本标记的后验概率。在 FSQ 模块中，中间表示 $H$ 首先投影到一个 $D$ 维低秩空间中，每个维度的值被量化到 $[-K, K]$ 的范围内，采用有界取整操作 ROUND。然后，量化后的低秩表示 $\hat{H}$ 被投影回原始维度 $\tilde{H}$ 以供后续模块使用：

$$
\begin{array}{l} \bar {H} = \operatorname {R O U N D} \left(\operatorname {P r o j} _ {\text {d o w n}} (H)\right) \tag {1} \\ \hat {H} = \operatorname {P r o j} _ {\mathrm {u p}} (\bar {H}) \\ \end{array}
$$

在训练阶段，直通估计用于近似 FSQ 模块和 $\mathrm{Encoder}_1$ 的梯度。语音词元 $\mu_{i}$ 可以通过计算量化低秩表示 $\bar{h}_{i}$ 在 $(2K+1)$ 进制系统中的索引得到：$\mu_{i}=\sum_{j=0}^{D-1}\bar{h}_{i,j}(2K+1)^{j}$ (2) $\mathrm{Encoder}_{1}$、FSQ 模块的低秩投影器、限幅取整操作和索引计算构成 CosyVoice 2 的语音分词器。我们的语音分词器以 25 Hz 的速率工作，即每秒生成 25 个语音词元。

![img-1.jpeg](images/2.jpeg)

Figure 2: A diagram of the unified text-speech language model for streaming and non-streaming synthesis in CosyVoice 2.

### 2.3 统一文本-语音语言模型

在CosyVoice 2中，预训练的文本大语言模型Qwen2.5-0.5B *[45]*被用作文本到语音的语言模型，通过将输入文本作为提示，自回归地生成语音词元。与其他语言模型类似，文本到语音语言模型也采用了下一个词预测的训练方案，如图1(b)所示。与之前的CosyVoice不同，我们移除了说话者嵌入，以避免信息泄露。更重要的是，我们发现这种话语级向量不仅包含说话者身份信息，还包含语言和副语言信息，这会影响文本到语音语言模型的韵律自然度和跨语言能力。此外，我们还放弃了之前CosyVoice的文本编码器，因为我们发现Qwen2.5-0.5B模型已经足够强大，可以对齐文本和语音词元，文本编码器不再需要。得益于文本到语音语言模型的简洁性，我们能够构建一个统一的模型，用于流式和非流式合成。这里的“流式模式”意味着输入文本是连续接收的，而不是事先作为完整句子得知。在CosyVoice 2中，流式和非流式模式之间的区别仅在于语言模型的序列构建方式：- 在非流式模式中，以“序列开始”、“所有文本词元”、“话语转变”词元、“所有语音词元”和“序列结束”按顺序连接，如图2底部所示。忽略词元意味着在最小化交叉熵目标函数时忽略它们的损失。- 在流式模式中，我们以预定义的比例$N{:}M$将文本词元和语音词元混合，即每$N$个文本词元后跟$M$个语音词元，如图2顶部所示。如果下一个词元是文本词元，模型预计要预测一个填充词元（而不是文本词元），这表示接下来的$N$个文本词元应在推断阶段连接。一旦文本词元用完，“话语转变”词元和剩余的语音词元按顺序连接，形成流式模式下的混合文本-语音词元序列。在我们的实验中，$N$和$M$分别设置为5和15。通过在上述两种序列上同时训练文本到语音语言模型，我们可以在单个统一模型中执行流式和非流式语音生成。在现实场景中，例如说话者微调（SFT）和上下文学习（ICL），推断序列的差异如下：

![img-2.jpeg](images/3.jpeg)

Figure 3: A diagram of the unified chunk-aware flow matching model for streaming and non-streaming synthesis in CosyVoice 2.

- ICL，非流式：在 ICL 中，语言模型需要参考音频中的提示文本和语音词元以模仿口音、韵律、情感和风格。在非流式模式下，提示和待合成文本词元被连接为一个整体，提示语音词元被视为预生成的结果并保持固定：“ $\mathbf{S}$ , prompt_text, text, $\mathbf{T}$ , prompt_speech”。语言模型的自回归生成从此序列开始，直到检测到“序列结束”词元 $\mathbf{E}$ 。 - ICL，流式：在这种情况下，我们假定待生成的文本是已知的，语音词元应以流式的方式生成。同样，我们将提示和待生成文本视为一个整体。然后，我们按 $N:M$ 的比例将其与提示语音词元混合：“ $\mathbf{S}$ , mixed_text_speech, $\mathbf{T}$ , remaining_speech”。如果文本的长度大于提示语音词元的长度，语言模型将生成“填充词元”。在这种情况下，我们手动填充 $N$ 个文本词元。如果文本词元用尽，将添加“语音转变”词元 $\mathbf{T}$。在流式模式中，我们每 $M$ 个词元返回生成结果，直到检测到 $\mathbf{E}$ 。 - SFT，非流式：在 SFT 场景中，语言模型在特定说话者上进行微调，不再需要提示文本和语音。因此，初始序列非常简单：“ $\mathbf{S}$ , text, $\mathbf{T}$ ”。从这里开始，文本语音语言模型能够自回归生成语音词元，直到 $\mathbf{T}$ 。 - SFT，流式：在 SFT 的流式模式中，我们从以下序列开始语音生成：“ $\mathbf{S}$ , first_N_text”。然后，语言模型将生成 $M$ 个语音词元，我们手动填充接下来的 $N$ 个文本词元。我们重复上述过程，直到所有文本词元用尽，然后添加 $\mathbf{T}$。注意，此模式也可被语音到语音的多模态大语言模型采用，以获得极低的延迟。

# 2.4 语块感知流匹配

在CosyVoice 2中，我们采用梅尔谱图作为声学特征，帧率为$50\mathrm{Hz}$，采样率为24000。由于语音标记与梅尔特征之间存在帧率不匹配，我们将语音标记以2:1的比例进行上采样，以匹配梅尔谱图的帧率。在上采样操作之前，我们添加了一个额外的前瞻卷积层，以提供未来信息供后续因果模块使用。前瞻层是通过填充右侧的1维卷积实现的，填充大小为$P$，核大小为$P + 1$。在此之后，几个块感知因果Transformer块接续在后，以对齐语音标记的表示空间，使其与声学特征相匹配。随后，我们的目标是进一步将语音标记解码为由说话者嵌入和参考语音指定的梅尔谱图。为此，我们采用条件流匹配（CFM）模型，根据语音标记、参考语音和说话者嵌入作为条件来采样梅尔谱图。在CFM模型中，目标梅尔谱图的分布由一个从先验分布$p_0(X)$到数据分布$q(X)$的概率密度路径描述。概率密度路径可以通过时间依赖的矢量场进行定义。为了提高采样效率，我们采用最优传输（OT）流动来匹配矢量场$\omega_{t}$，该矢量场由常微分方程（ODE）给出：$\omega_{t}(\phi_{t}^{OT}(X_{0},X_{1})|X_{1})=X_{1}-X_{0}$ $\phi_{t}^{OT}(X_{0},X_{1})=(1-t)X_{0}+tX_{1}$ $X_{0} \sim p_{0}(X)=\mathcal{N}(0,I)$ $X_{1} \sim q(X)$ 。采用因果卷积Transformer UNet来学习上述ODE，输入为上采样的标记$\mu$、掩蔽梅尔谱图$\tilde{X}_{1}$、说话者嵌入$\mathbf{v}$和时间步$t$作为条件：$\nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta)=\mathrm{UNet}_{\theta}\left(\phi_{t}^{OT}(X_{0},X_{1}),t;\mathbf{v},\{\mu\}_{1:L},\tilde{X_{1}}\right)$。在训练阶段，通过随机掩蔽$X_{1}$中70%到100%的最后帧来获得掩蔽梅尔谱图。而在推理阶段，使用从参考语音提取的梅尔谱图。通过最小化预测和真实ODE之间的L1损失，我们可以优化UNet参数$\theta$如下：$\theta=\arg\min_{\theta}\mathbb{E}_{p_{0}(X),q(X),t}\Big{|}\omega_{t}(\phi_{t}^{OT}(X_{0},X_{1}))-\nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta;\mu,\tilde{X}_{1},\mathbf{v})\Big{|}_{1}$。在训练阶段，时间步遵循均匀分布$U[0,1]$。然而，在推理阶段，我们采用余弦调度器提供更多的初始生成阶段步骤：$t:=1-\cos\left(\frac{1}{2}t\pi\right)$。此外，我们还在条件和非条件情况下对模型进行训练，以在推理阶段实现无分类器指导（CFG）：$\tilde{\nu}_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta;\Psi)=(1+\beta)\cdot\nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta;\Psi)-\beta\cdot\nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta)$，其中$\Psi$表示条件$\{\mathbf{v},\mu,\tilde{X_{1}}\}$。根据实验结果，CFG强度$\beta$和流估计数量（NFE）分别设置为$0.7$和$10$。当前的流匹配模型总是以离线模式运行，即仅在生成所有语音标记后，梅尔谱图才能被采样，这对于流式合成并不友好。为了克服这一问题，我们将多步骤流估计视为一个堆叠的深度神经网络，重复10次UNet。因此，通过使展开的神经网络因果化，我们可以将其应用于流式合成。我们构建了四种掩蔽以满足不同的应用场景：- 非因果掩蔽用于离线模式，通过关注所有条件帧来实现最佳性能。非因果掩蔽适用于对延迟不敏感的情况。- 全因果掩蔽设计用于要求极低延迟的场景，其中仅能关注过去的帧。- Chunk-$M$掩蔽是在延迟和性能之间的折中，能够利用过去的和$M$个未来帧的信息。该掩蔽更适用于低延迟的生成首块。- Chunk-$2M$掩蔽通过牺牲更多延迟来实现接近离线模式的性能，可用于更好性能的级联生成块。对于一个小批次中的每个训练案例，我们根据均匀分布随机从上述四种掩蔽中抽样。通过这种方式，一个流匹配模型可以兼容不同的场景，降低部署复杂性。这种块感知训练的另一个优点是，具有更多上下文的掩蔽可以作为教师为上下文较少的掩蔽提供指导，从而受益于隐式自蒸馏方案。

## 2.5 流模式下的延迟分析

首个包延迟是流式合成模型的重要指标，它显著影响用户体验，特别是在基于大型语言模型的语音聊天应用中，例如 GPT-4o [36]。在文本到语音转换（TTS）的背景下，待合成的文本是提前已知的，延迟来自于语音词元生成、梅尔频谱重构和波形合成等方面。因此，CosyVoice 2 的首个包延迟 $L_{TTS}$ 可以通过以下方式获得：

$$
L_{TTS} = M \cdot d_{lm} + M \cdot d_{fm} + M \cdot d_{voc} \tag{11}
$$

其中 $d_{lm}$ 表示语言模型生成一个语音词元所需的计算时间，$d_{fm}$ 代表流匹配模型生成一个语音词元对应的梅尔谱帧所需的计算时间，$d_{voc}$ 表示声码器合成与一个语音词元对应的波形所需的计算时间。在基于大语言模型的语音聊天中，还应考虑第一包所需文本的长度，因此第一包延迟 $L_{Chat}$ 如下所示：

$$
L_{Chat} \leq N \cdot d_{llm} + L_{TTS} \tag{12}
$$

其中 $d_{llm}$ 表示大型语言模型生成一个文本词元的计算时间。请注意，由于多字符词元在 CosyVoice 2 的文本分词器中被屏蔽，因此文本大型语言模型使用的文本词元始终编码比 CosyVoice 2 更长的原始文本。因此，首包延迟 $L_{Chat}$ 必须低于 $N \cdot d_{llm}$ 和 $L_{TTS}$ 的总和。

## 2.6 指令生成

为增强 CosyVoice 2 的可控性，我们将指令数据集整合到了基础训练集中。我们收集了1500小时的指令训练数据，包含自然语言指令和细粒度指令，如表1所示。对于自然语言指令，在待合成的输入文本之前，我们加上自然语言描述和特殊的结束标记“<endofprompt>”。这些描述涵盖情感、语速、角色扮演和方言等方面。对于细粒度指令，我们在文本标记之间插入声音片段，使用标记如“[laughter]”和“[breath]”。此外，我们还为短语应用声学特征标签；例如，“<strong>XXX</strong>”表示对某些单词的强调，而“<laughter>XXX</laughter>”表示带有笑声的说话方式。

|  Natural Language Instruction  |
| --- |
|  Emotion: 高兴(Happy), 悲伤(Sad), 惊讶(Surprised), 愤怒(Angry), 恐惧(Fearful), 厌恶(Disgusted), 冷静(Calm), 严肃(Serious)  |
|  Speaking Rate: 快速(Fast), 非常快速(Very Fast), 慢速(Slow), 非常慢速(Very Slow)  |
|  Dialect: 粤语, 四川话, 上海话, 郑州话, 长沙话, 天津话  |
|  Role-playing: 神秘(Mysterious), 凶猛(Fierce), 好奇(Curious), 优雅(Elegant), 孤独(Lonely), 机器人(Robot), 小猪佩奇(Peppa), etc.  |
|  Fine-grained Instruction  |
|  Vocal Bursts: [laughter], [breath], etc.  |
|  Vocal Features: <laughter></laughter>, <strong></strong>  |
|  Examples  |
|  - 你能用高兴的情感说吗？ < |endofprompt| > 今天真是太开心了，马上要放假了！ I'm so happy, Spring Festival is coming!  |
|  - Please speaking very fast. < |endofprompt| > Today is a happy day, full of laughter and joy.  |
|  - 请问你能模仿粤语的口音吗？ < |endofprompt| > 多保重，早休息。  |
|  - 尝试一下以机器人的角色和我交流。 < |endofprompt| > 接收知识光波！  |
|  - [laughter] 有时候，看着小孩子们的天真行为[laughter]，我们总会会心一笑。  |
|  - She pursued her dreams with **enthusiasm**  |
|  and **grit**  |

Table 1: Examples of natural language instructions and fine-grained instructions.

## 2.7 多说话者微调

对特定说话者进行微调（SFT）可以进一步提高生成质量和说话者相似性。在本报告中，我们介绍了多说话者微调（mSFT），在这种方法中，预训练模型会同时在多个说话者上进行微调，而不是单一说话者。这种方法确保了在多个说话者之间全面覆盖韵律和发音，并减少了预训练模型可能出现的灾难性遗忘。为了避免不同说话者之间的音色混淆，我们在特定说话者的输入文本前添加了说话者提示标签“Speaker A&lt;|endofprompt|&gt;”。如果训练样本未标注到某个说话者，会使用特殊标签“unknown&lt;|endofprompt|&gt;”。在整个多说话者微调过程中，学习率设定为1e-5。

### 2.8 强化学习用于SFT

强化学习是训练大型语言模型中常用的方法，可以使语言模型的输出与人类偏好对齐。在CosyVoice 2中，我们采用来自自动语音识别（ASR）系统的说话人相似度（SS）和识别词错误率（WER）作为奖励函数，以在微调阶段提高说话人相似度和发音准确性。我们使用WER和SS来区分优选样本$x^{w}$和拒绝样本$x^{l}$，并使用直接偏好优化（DPO）来优化文本到语音（TTS）系统，如下所示：$L_{DPO}(\pi_{\theta};\pi_{\text{ref}})=-\log\sigma(\beta\log\frac{\pi_{\theta}(\mu^{w}|y)}{\pi_{\text{ref}}(\mu^{w}|y)}-\beta\log\frac{\pi_{\theta}(\mu^{l}|y)}{\pi_{\text{ref}}(\mu^{l}|y)})$ (13)，其中$\mu^{w}$和$\mu^{l}$分别是从优选样本$x^{w}$和拒绝样本$x^{l}$中提取的语音词元。然而，这种方法消耗时间和计算资源，因为需要通过TTS系统重复合成音频以获得可区分的优选和拒绝样本。在训练期间，一次训练步骤需要进行四次前向运算。为了简化该过程，我们将语言模型预测的词元$\mu_{i}\in\{0,1,\ldots,(2K+1)^{D}-1\}$恢复为量化的低秩表示$\bar{H}$，并直接使用语音分词器的ASR后端重新预测输入文本。然后，预测的对数后验可以看作ASR奖励函数，以优化文本到语音的语言模型。在训练过程中，ASR后端参数保持不变。$\bar{h}_{i,j}=\left\lfloor\frac{\mu_{i}}{(2K+1)^{j}}\right\rfloor\mod(2K+1)$ (14) $\hat{H}=\text{Proj}_{up}(\bar{H})$ (15) $L_{ASR}=-\log P(Y|\hat{H};\theta_{ASR})$，其中$Y$是输入文本，$\bar{H}$是恢复的语音低秩表示。由于操作$u_{i}\sim P(\mu_{i}|\mu_{1:i-1},Y;\theta_{LM})$仍然阻止我们直接优化模型，我们使用Gumbel Softmax抽样使其可微分，然后通过$\mathcal{L}_{ASR}$优化$\theta_{LM}$。

## 3 实验设置

### 3.1 语音分词器的训练数据

使用200,000小时的数据集来训练语音分词器，规范化的转录文本作为标签。详细的数据说明列在表2中。训练数据来自三个不同的来源：开源自动语音识别（ASR）数据集、内部工业数据集和文本到语音（TTS）生成数据集。尽管在训练语音分词器时仅使用了中文和英文数据，如表2所示，但后续实验表明，该语音分词器对其他语言具有零样本能力。它也可以用于日语和韩语等语言的语音合成。

|  Language | Duration (hours)  |
| --- | --- |
|  Chinese | 110,884  |
|  English | 99,918  |

Table 2: Details of training data for speech tokenizer.

# 3.2 CosyVoice 2 的训练数据

CosyVoice 2 与其前一版本共享相同的训练数据 [34]。我们首先使用内部语音处理工具收集仅包含语音的数据。随后，分别使用 Paraformer [50] 和 SenseVoice [43] 为中文和其他语言生成伪文本标签。我们还使用内部强对齐模型过滤低质量数据，并提高标点的准确性。数据详情见表 3。

|  Language | Duration (hours)  |
| --- | --- |
|  Chinese | 130,000  |
|  English | 30,000  |
|  Japanese | 4,600  |
|  Korean | 2,200  |

Table 3: Details of training data for CosyVoice 2.

# 3.3 评估设置

我们在两个测试集上评估了我们的CosyVoice 2。第一个测试集来自Librispeech语料库的test-clean集[51]，称为test-clean。该测试集用于评估CosyVoice 2在有限英语领域的表现。我们使用Whisper-large V3作为ASR模型来评估内容一致性。至于说话人相似性（SS），我们采用ERes2Net模型[52]提取提示和生成语句的说话人嵌入，其原始余弦相似性被视为说话人相似性。我们使用NMOS评分[53]来评估客观质量。第二次评估是在广泛用于评估近期TTS模型的SEED测试集[33]上进行的，涵盖了各种文本领域和参考演讲。在此评估中，从CommonVoice数据集中选择了约2000个中文样本和1000个英文样本，分别称为test-zh和test-en。此外，还包括约400个困难测试案例，以评估TTS模型在文本重复、绕口令和其他挑战性合成案例上的鲁棒性，称为本报告中的test-hard。我们采用Paraformer识别test-zh和test-hard的合成结果，而Whisper-large V3用于test-en以评估内容一致性。我们采用两个说话人验证（SV）模型来评估说话人相似性：WavLM微调SV模型和ERes2Net。

# 3.4 日语和韩语的基准测试

我们准备了两个测试集，分别标记为 test-ja 和 test-ko，以评估日语和韩语的语音合成。test-ja 包含从 CommonVoice 数据集中提取的 1,000 个样本，用于测量模型在各种指标上的性能，如错误词率 (WER)，说话流畅性 (SS) 和主观评分 (MOS)。具体来说，我们随机打乱并配对整个 CommonVoice JA 测试集的参考话语和目标话语。考虑到 JA 测试集中话语文本长度的广泛分布，我们从 8 到 32 个字符的长度范围内随机选择了 1,000 对参考-目标话语作为我们的最终测试集。对于 test-ko，我们选择了 1,000 个错误词率 (WER) 小于 $5\%$ 且没有删除或插入错误的语音样本，利用 Whisper-Large V3 作为 ASR 模型。这些样本被用作韩语语音合成的参考话语。对于输入文本，我们从剩余数据中随机选择了 1,000 个文本样本。我们公开了这两个测试集的提示语音、提示转录和输入文本列表，以便于结果的复现。通过提供这些开源数据，我们旨在建立评估日语和韩语 TTS 模型的基准。Whisper-large V3 被用作日语和韩语评估的 ASR 模型。

# 4 实验结果

# 4.1 对语音标记器的评估

理想的语音标记器应能够有效利用码本，高保真地保留信息，并表现出说话人独立性。在这一部分，我们从四个方面评估我们的监督语音标记器：1) 码本利用率；2) 整个编码器中的ASR错误率；3) 不同说话人的标记可视化；4) 说话人识别训练。表4显示了码本利用率和ASR错误率。结果表明，基于FSQ的标记器充分利用了码本，并在ASR方面维持了更有效的信息，这表明FSQ保持了更多的语义信息。

|  Method | Codebook |   | ASR Error Rate (%)  |   |   |   |
| --- | --- | --- | --- | --- | --- | --- |
|   |  Size | Util. | C.V. EN | C.V. CN | Fluers EN | Fluers CN  |
|  VQ | 4,096 | 963 (23%) | 18.26 | 11.56 | 7.65 | 5.03  |
|  FSQ | 6,561 | 6,561 (100%) | 10.67 | 7.29 | 6.58 | 4.43  |

Table 4: The comparison of VQ and FSQ inside Sensevoice-large encoder. C.V. stands for the CommonVoice benchmarks.

我们进一步通过 t-SNE 可视化分析了 FSQ 的特征。作为 TTS 任务的上游模型，分词器应努力最小化说话者身份信息与语音信号之间的纠缠。我们从 VoxCeleb1 数据集中选择了三位说话者中的每位各 100 条语音样本，并可视化了相应的词元。如图 4(a) 和 (b) 所示，在量化之前，编码器 $_1$ 的输出在不同说话者之间表现出不同的分布。相比之下，量化表示的分布几乎无法区分。此外，图 4(c) 还显示，分词器充分利用了代码本。随后，使用 S3prl 工具包进一步评估说话者的纠缠，通过进行说话者识别（SID）任务。我们使用带有 FSQ 的 Sensevoice-large 编码器作为上游特征提取器，并在量化前后对表示进行 SID 任务的训练。图 5 显示了训练过程中的准确率曲线。带有量化词元的 SID 层未能收敛，这证明了分词器对说话者信息的解耦功能。

![img-3.jpeg](images/4.jpeg)
(a)

![img-4.jpeg](images/5.jpeg)
(b)

Figure 4: The t-SNE visualization of speech representations before (a) and after (b) the quantization for three different speakers in Voxceb1 dataset. (c) shows the codebook utilization in terms of the token percentage on the speakers (500 tokens each bin).

![img-5.jpeg](images/6.jpeg)
(c)

# 4.2 与基线的比较结果

我们首先在有限的英语文本领域评估了我们的 CosyVoice 2 模型，并将其与几个开源模型进行了比较，如 ChatTTS、GPT-SoVITs、OpenVoice、ParlerTTS、EmotiVoice 以及其前身 CosyVoice。客观结果如表 5 所示，包括内容一致性（WER）、语音质量（NMOS）和说话者相似性（SS）。从表中可以看出，CosyVoice 2 在 Librispeech test-clean 集合上达到了最先进的性能，超越了所有基准模型的所有评估指标。值得注意的是，CosyVoice 2 甚至在内容一致性、语音质量和说话者相似性方面表现得比人类的发声更高，表明其合成质量达到了人类水平。

![img-6.jpeg](images/7.jpeg)

Figure 5: The convergence curves of SID training with tokens before or after quantization.

|  Model | WER (%) | NMOS | SS  |
| --- | --- | --- | --- |
|  Human | 2.66 | 3.84 | 0.697  |
|  ChatTTS [56] | 6.84 | 3.89 | -  |
|  GPT-SoVITs [57] | 5.13 | 3.93 | 0.405  |
|  OpenVoice [58] | 3.47 | 3.87 | 0.299  |
|  ParlerTTS [59] | 3.16 | 3.86 | -  |
|  EmotiVoice [60] | 3.14 | 3.93 | -  |
|  CosyVoice [34] | 2.89 | 3.93 | 0.743  |
|  CosyVoice 2 | 2.47 | 3.96 | 0.745  |
|  CosyVoice 2-S | 2.45 | 3.90 | 0.751  |

我们还在常用的测试集上评估了CosyVoice 2：SEED test-zh、test-en和test-hard，这些测试集包括来自各个领域的多样化输入文本和参考语音。CosyVoice 2和基线模型的实验结果如表6所示。在test-zh集上，CosyVoice 2在CER和SS方面超越了所有开源模型，仅与商业模型SEED-TTS相比稍显不足。在test-en集上，CosyVoice 2在WER和SS方面分别排名第四和第三。这可能是由于中文和英文训练数据量的不平衡。我们计划在未来的工作中探索数据扩展，以增强英文内容的一致性。在test-hard集中，离线CosyVoice 2模型在所有比较的基线中实现了最先进的性能，展示了其在挑战性合成场景中的稳健性。与人类生成的语音相比，CosyVoice 2表现出可比的内容一致性和优越的说话人相似性。考虑到识别错误也可能源于ASR模型，可以合理得出结论，CosyVoice 2实现了与人类相当的合成能力。我们还评估了流式模式，在表5和6中标记为“CosyVoice 2-S”。对于这两种评估设置，流式模式在典型测试案例中的表现几乎是无损的。仅在挑战性案例中，内容一致性略有下降，这突显了我们统一流式/非流式框架的优势。我们发现不同SV模型上的说话人相似性结果并不一致。这可能表明一个新的研究主题，即如何自动评估TTS模型的说话人相似性。由于不同TTS模型可能使用不同的SV模型来提取说话人信息，使用相同的SV模型评估说话人相似性可以更准确地评价说话人信息的利用。因此，我们在后续实验中采用ERes2Net $^{3}$ 来评估说话人相似性。

Table 5: Content consistency (WER), speaker similarity (SS) and speech quality (NMOS) results on LibriSpeech test-clean subset of baselines and CosyVoice 2. Whisper-Large V3 is employed as the ASR model and punctuations are excluded before WER calculation.

|  Model | test-zh |   | test-en |   | test-hard  |   |
| --- | --- | --- | --- | --- | --- | --- |
|   |  CER (%) ↓ | SS ↑ | WER (%) ↓ | SS ↑ | WER (%) ↓ | SS ↑  |
|  Human | 1.26 | 0.755 (0.775) | 2.14 | 0.734 (0.742) | - | -  |
|  Vocoder Resyn. | 1.27 | 0.720 | 2.17 | 0.700 | - | -  |
|  Seed-TTS† [33] | 1.12 | 0.796 | 2.25 | 0.762 | 7.59 | 0.776  |
|  FireRedTTS [35] | 1.51 | 0.635 (0.653) | 3.82 | 0.460 (0.526) | 17.45 | 0.621 (0.639)  |
|  MaskGCT [18] | 2.27 | 0.774 (0.752) | 2.62 | 0.714 (0.730) | 10.27 | 0.748 (0.720)  |
|  E2 TTS (32 NFE)† [31] | 1.97 | 0.730 | 2.19 | 0.710 | - | -  |
|  F5-TTS (32 NFE) [32] | 1.56 | 0.741 (0.794) | 1.83 | 0.647 (0.742) | 8.67 | 0.713 (0.762)  |
|  CosyVoice [34] | 3.63 | 0.723 (0.775) | 4.29 | 0.609 (0.699) | 11.75 | 0.709 (0.755)  |
|  CosyVoice 2 | 1.45 | 0.748 (0.806) | 2.57 | 0.652 (0.736) | 6.83 | 0.724 (0.776)  |
|  CosyVoice 2-S | 1.45 | 0.753 (0.812) | 2.38 | 0.654 (0.743) | 8.08 | 0.732 (0.785)  |

Table 6: Results of CosyVoice 2 and recent TTS models on the SEED test sets.  $\dagger$  denotes close-sourced models. For speaker similarity, the result in a bracket are measured by ERes2Net, while the results outside brackets are measured by WavLM-based models.

# 4.3 模块消融研究

我们对文本语音语言模型进行了模块化消融研究，以评估我们修改的影响，包括LLM初始化、去除说话者嵌入和利用FSQ。表7展示了CosyVoice 2从其前身的逐步发展。通过用预训练的LLM替换随机初始化的语言模型，我们在test-zh和test-hard数据集上分别实现了内容一致性相对提升$18.46\%$和$15.40\%$。接下来，我们从文本到语音的语言模型中去除了说话者嵌入，这有助于防止信息泄露和上下文学习中的干扰。这一变化显著减少了内容错误，同时保持了说话者相似度，表明内容信息主要由语言模型建模，而说话者信息主要由流匹配模型恢复。最后，通过用FSQ替换VQ，我们得到了CosyVoice 2模型，注意到内容一致性大幅提高且说话者相似度保持不变。通过充分利用码本，FSQ捕捉了更多的内容信息和上下文变化，从而实现了文本和语音词元之间的更好对齐。此外，我们通过在基于FSQ的语音标记器训练过程中引入音高损失作为约束，进行了对比实验。我们发现这种方法提高了下游TTS任务的性能，如表7最后一行所示。在未来的CosyVoice版本中，我们计划进行更详细的实验和分析。

|  Model | test-zh |   | test-en |   | test-hard  |   |
| --- | --- | --- | --- | --- | --- | --- |
|   |  CER (%) | SS | WER (%) | SS | WER (%) | SS  |
|  CosyVoice | 3.63 | 0.775 | 4.29 | 0.699 | 11.75 | 0.755  |
|  + LLM init. | 2.96 | 0.808 | 4.57 | 0.730 | 9.94 | 0.789  |
|  + Drop Spk Emb. | 2.56 | 0.804 | 3.81 | 0.740 | 9.66 | 0.778  |
|  + FSQ (CosyVoice 2) | 1.45 | 0.806 | 2.57 | 0.736 | 6.83 | 0.776  |
|  + Pitch Loss | 1.19 | 0.802 | 2.40 | 0.728 | 6.29 | 0.769  |

Table 7: Modular analysis on the modifications of text-speech language model.

我们还进行了另一项模块化分析，以评估流式模块对合成性能的影响。表8展示了内容一致性和说话人相似性的结果。我们发现，流式语言模型对来自 test-zh 和 test-en 数据集的典型案例影响甚微，这表明我们的统一训练框架是有效的。流式语言模型的主要影响体现在来自 test-hard 数据集的具有挑战性的案例中，这可能是由于流式模式下上下文信息的丢失。有趣的是，与离线模式相比，流式流匹配模型的说话人相似性略高。这可能是因为流式模式下初始块的提示到生成比率较高，而离线模式的提示到生成比率可能很低，且许多填充词元的存在。与流式语言模型相比，流式流匹配模型对内容一致性的负面影响要小得多，这得益于 CosyVoice 2 中语义-声学的解耦建模。

|  Model | LM | FM | test-zh |   | test-en |   | test-hard  |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |   |  CER (%) | SS | WER (%) | SS | CER (%) | SS  |
|  M1 | Offline | Offline | 1.45 | 0.806 | 2.57 | 0.736 | 6.83 | 0.776  |
|  M2 | Offline | Stream. | 1.46 | 0.811 | 2.60 | 0.743 | 7.12 | 0.788  |
|  M3 | Stream. | Offline | 1.38 | 0.806 | 2.51 | 0.737 | 7.88 | 0.773  |
|  M4 | Stream. | Stream. | 1.45 | 0.812 | 2.38 | 0.743 | 8.08 | 0.785  |

# 4.4 日本和韩国基准上的结果

除了中文和英文，CosyVoice 2 还支持日语和韩语。我们在构建的日语和韩语测试集中评估了内容一致性、说话人相似性和语音质量。如表 9 所示，CosyVoice 2 在所有评估指标上对韩语的表现显著优于日语。这种差异主要源于日语与中文之间字符集的重叠，导致在日语环境中出现中文发音。在未来的工作中，我们计划探索增强多语言合成语言上下文的方法。由于韩语与其他语言不存在字符重叠，因此其语音合成性能表现更佳。另一个问题是数据不平衡。我们认为，增加训练数据的量可以进一步提升日语和韩语的合成性能。

Table 8: Modular analysis on the impact of streaming modules in CosyVoice 2. Chunk size is set to 15 for streaming modules.

|  Model | test-ja |   |   | test-ko  |   |   |
| --- | --- | --- | --- | --- | --- | --- |
|   |  CER (%) | SS | NMOS | CER (%) | SS | NMOS  |
|  CosyVoice 2 | 18.79 | 0.630 | 3.42 | 7.98 | 0.707 | 3.73  |
|  CosyVoice 2-S | 21.41 | 0.629 | 3.35 | 9.06 | 0.714 | 3.60  |

Table 9: The content consistency (CER), speaker similarity (SS), and speech quality (NMOS) of CosyVoice 2 and its streaming counterpart on the Japanese test-ja and Korean test-ko test sets.

# 4.5 指导生成的结果

为了评估指令生成的性能，我们创建了一个包含290个样本的中文测试集。该数据集包括29种类型的指令，如表1所示，每种指令有10个不同的输入文本。我们利用来自五位说话者（其中三名女性和两名男性）的五个音频提示和说话人嵌入作为流匹配模型的条件。我们的测试是在离线模式下进行。我们客观评估内容一致性（CER）、说话人相似性（SS）和语音质量（NMOS）。主观上，我们使用指令平均意见评分（MOS-I）来评估指令的准确性和自然性，评分范围为1到5。每个样本由10名母语为中文的评估者进行评分，得分以0.5为增量。评估标准侧重于语音是否遵循所有指定指令，例如情感表达、语速调整、方言使用和角色扮演。我们评估了微观控制，包括插入笑声、带笑声说话、呼吸控制和重音的自然性和准确性。如表10所示，CosyVoice 2 显示出优越的内容一致性（CER）、说话人相似性（SS）以及指令控制的准确性和自然性（MOS-I），同时在语音质量上与CosyVoice-Instruct保持可比性。当从CosyVoice 2中移除输入指令时，MOS-I显著下降；然而，内容一致性（CER）、说话人相似性（SS）和语音质量（NMOS）有所改善。这表明，指令可控性难以从内容文本中隐式产生。

# 4.6 演讲者微调模型的结果

在微调阶段，我们对同一说话者的说话者嵌入进行无监督聚类，以确保说话者音色的稳定性。我们已经证明，目标说话者即使只有400个音频录音，也能实现相当不错的语音合成性能，不同说话者之间的客观指标仅观察到轻微的变化，如图6所示。我们的实验表明，大多数说话者可以继承零-shot TTS模型的强大上下文理解能力和感知能力，从而自然地在输入文本的基础上表达各种情绪和心情。

|  Model | CER (%) | SS | NMOS | MOS-I  |
| --- | --- | --- | --- | --- |
|  CosyVoice-Instruct [34] | 1.72 | 0.797 | 3.94 | 3.09  |
|  CosyVoice 2 | 1.52 | 0.804 | 3.94 | 4.06  |
|  CosyVoice 2 w/o Instruction | 0.97 | 0.817 | 4.02 | 2.28  |

Table 10: Evaluation results for content consistency (CER), speaker similarity (SS), speech quality (NMOS), and MOS-I (Instruction, assessing the accuracy and naturalness of instruction) on an in-house Chinese test set for CosyVoice-Instruct, CosyVoice 2, and CosyVoice 2 without instruction input. The Paraformer model is used as the ASR system, with punctuation marks excluded from the CER calculation. Dialect data is not included in the CER calculation because the Paraformer model cannot recognize Chinese dialect speech.

![img-7.jpeg](images/8.jpeg)

Figure 6: Results of CosyVoice 2 SFT Models under the SEED evaluation settings. CER is used for test-zh and test-hard, while WER is used for test-en.

# 4.7 使用强化学习进行语言模型微调

虽然SFT可以提高大多数说话者的性能，但Spk E的结果仍然不如基模型，尤其是在英语方面。这是因为Spk E的声音更加复杂，语速更快。此外，Spk E仅有中文录音可用。因此，我们对Spk E应用强化学习以进一步改善性能。对于DPO，我们通过SFT模型合成了一万对样本，以通过ASR和SS奖励来改变语言模型的偏好偏向。我们还使用可微分的ASR奖励来优化语言模型参数。经过强化学习后，我们在Spk E的测试集上评估了模型的内容一致性（WER）、说话者相似性（SS）和语音质量（NMOS），并进一步在SeedTTS测试集上评估了WER，以探讨模型是否能保持对域外或跨语言输入文本的鲁棒性。结果如表11所示。

与预训练基础模型相比，SFT模型在说话人相似性和语音质量上表现更佳，但其字错误率可能不如基础模型。我们发现，基础模型合成的音频速度通常比SFT和真实标记数据慢，这对自动语音识别系统（ASR）更为友好。针对目标说话人数据集，偏好偏向和可微分奖励均能够减少字错误率，且对其他两个指标影响较小。但对于SEED测试集，基于DPO的强化学习只对中文和英文子集有益，而在难样本上则表现较差。原因可能在于难样本包含许多重复的词或短语，可能在DPO训练过程中被视为拒绝样本。然而，可微分ASR奖励不会遭遇这一问题，因为它可以通过ASR后验直接优化TTS系统。这意味着可微分ASR奖励在跨领域情境中具有更好的泛化能力。最后，我们可以将它们结合起来以实现进一步改进。

|  Model | Inhome Target Speaker |   |   | SEED tests(%)  |   |   |
| --- | --- | --- | --- | --- | --- | --- |
|   |  WER(%) | NMOS | SS | zh | en | hard  |
|  Ground Truth | 6.00 | 3.87 | 0.697 | 1.26 | 2.14 | -  |
|  CosyVoice 2 | 5.34 | 3.91 | 0.721 | 1.45 | 2.57 | 6.83  |
|  CosyVoice 2-SFT | 7.15 | 3.96 | 0.795 | 1.50 | 4.26 | 7.90  |
|  + LASR | 6.79 | 3.96 | 0.795 | 1.29 | 3.53 | 7.30  |
|  + LDPO | 6.83 | 3.96 | 0.792 | 1.43 | 4.02 | 8.31  |
|  + LASR + LDPO | 6.64 | 3.97 | 0.796 | 1.25 | 3.17 | 6.66  |

Table 11: Content consistency (WER), speaker similarity (SS) and speech quality (NMOS) comparison for reinforcement learning models on Spk E.

# 5 结论

基于CosyVoice的成功，本报告介绍了CosyVoice 2，这是一种改进的流式语音合成模型，利用了大型语言模型。CosyVoice 2通过在单一框架内统一流式和非流式合成，达到了人类相当的自然性、最小的响应延迟以及几乎无损的流式合成质量。关键创新包括有限标量量化以充分利用完整的编码簿、一种简化的文本到语音语言模型架构，该架构整合了预训练的文本大型语言模型，以及开发了一种块感知的因果流匹配模型，以支持多样化的合成场景。此外，改进的指令式文本到语音能力允许在情感、口音、角色风格和语音崩发方面进行细粒度控制，从而实现多样化和生动的语音生成。通过系统性修改和优化，CosyVoice 2不仅提供了优越的合成质量，还降低了部署要求，使其适用于流式和非流式应用。我们相信，CosyVoice 2代表了可扩展、高质量和交互式文本到语音合成的重要进步。

# 6 限制条件

CosyVoice 2 存在若干限制需要解决。首先，它仅支持有限数量的语言。对于字符集重叠的语言，合成性能可能会下降，这为未来的研究提出了一个开放性挑战。其次，CosyVoice 2 无法通过文本指令控制声学特征，如音色，这可能成为角色扮演应用的一个有趣探索领域。此外，CosyVoice 在执行唱歌任务时表现不佳。

# References

[1] Yuxuan Wang, R. J. Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J. Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, Quoc V. Le, Yannis Agiomyrgiannakis, Rob Clark, and Rif A. Saurous. Tacotron: Towards end-to-end speech synthesis. In INTERSPEECH, pages 4006-4010. ISCA, 2017.
[2] Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, R. J. Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, and Yonghui Wu. Natural TTS synthesis by conditioning wavenet on MEL spectrogram predictions. In ICASSP, pages 4779-4783. IEEE, 2018.
[3] Wei Ping, Kainan Peng, Andrew Gibiansky, Sercan Ömer Arik, Ajay Kannan, Sharan Narang, Jonathan Raiman, and John Miller. Deep voice 3: 2000-speaker neural text-to-speech. CoRR, abs/1710.07654, 2017.
[4] Wei Ping, Kainan Peng, and Jitong Chen. Clarinet: Parallel wave generation in end-to-end text-to-speech. In ICLR (Poster). OpenReview.net, 2019.
[5] Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. Fast-speech: Fast, robust and controllable text to speech. In NeurIPS, pages 3165-3174, 2019.

[6] Naihan Li, Shujie Liu, Yanqing Liu, Sheng Zhao, and Ming Liu. Neural speech synthesis with transformer network. In AAAI, pages 6706–6713. AAAI Press, 2019.
- [7] Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. Fastspeech 2: Fast and high-quality end-to-end text to speech. In ICLR. OpenReview.net, 2021.
- [8] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, and Furu Wei. Neural codec language models are zero-shot text to speech synthesizers. CoRR, abs/2301.02111, 2023.
- [9] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE ACM Trans. Audio Speech Lang. Process., 30:495–507, 2022.
- [10] Alexandre Défossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio compression. Trans. Mach. Learn. Res., 2023, 2023.
- [11] Zhihao Du, Shiliang Zhang, Kai Hu, and Siqi Zheng. Funcodec: A fundamental, reproducible and integrable open-source toolkit for neural speech codec. In ICASSP, pages 591–595. IEEE, 2024.
- [12] Eugene Kharitonov, Damien Vincent, Zalán Borsos, Raphaël Marinier, Sertan Girgin, Olivier Pietquin, Matt Sharifi, Marco Tagliasacchi, and Neil Zeghidour. Speak, read and prompt: High-fidelity text-to-speech with minimal supervision. Trans. Assoc. Comput. Linguistics, 11:1703–1718, 2023.
- [13] Yakun Song, Zhuo Chen, Xiaofei Wang, Ziyang Ma, and Xie Chen. ELLA-V: stable neural codec language modeling with alignment-guided sequence reordering. CoRR, abs/2401.07333, 2024.
- [14] Chenpeng Du, Yiwei Guo, Hankun Wang, Yifan Yang, Zhikang Niu, Shuai Wang, Hui Zhang, Xie Chen, and Kai Yu. VALL-T: decoder-only generative transducer for robust and decoding-controllable text-to-speech. CoRR, abs/2401.14321, 2024.
- [15] Detai Xin, Xu Tan, Kai Shen, Zeqian Ju, Dongchao Yang, Yuancheng Wang, Shinnosuke Takamichi, Hiroshi Saruwatari, Shujie Liu, Jinyu Li, and Sheng Zhao. RALL-E: robust codec language modeling with chain-of-thought prompting for text-to-speech synthesis. CoRR, abs/2404.03204, 2024.
- [16] Sanyuan Chen, Shujie Liu, Long Zhou, Yanqing Liu, Xu Tan, Jinyu Li, Sheng Zhao, Yao Qian, and Furu Wei. VALL-E 2: Neural codec language models are human parity zero-shot text to speech synthesizers. CoRR, abs/2406.05370, 2024.
- [17] Bing Han, Long Zhou, Shujie Liu, Sanyuan Chen, Lingwei Meng, Yanming Qian, Yanqing Liu, Sheng Zhao, Jinyu Li, and Furu Wei. VALL-E R: robust and efficient zero-shot text-to-speech synthesis via monotonic alignment. CoRR, abs/2406.07855, 2024.
- [18] Yuancheng Wang, Haoyue Zhan, Liwei Liu, Ruihong Zeng, Haotian Guo, Jiachen Zheng, Qiang Zhang, Shunsi Zhang, and Zhizheng Wu. Maskgct: Zero-shot text-to-speech with masked generative codec transformer. CoRR, abs/2409.00750, 2024.
- [19] Takuma Okamoto, Haruki Yamashita, Yamato Ohtani, Tomoki Toda, and Hisashi Kawai. Wavenext: Convnext-based fast neural vocoder without ISTFT layer. In ASRU, pages 1–8. IEEE, 2023.
- [20] Hubert Siuzdak. Vocos: Closing the gap between time-domain and fourier-based neural vocoders for high-quality audio synthesis. In ICLR. OpenReview.net, 2024.
- [21] Lingwei Meng, Long Zhou, Shujie Liu, Sanyuan Chen, Bing Han, Shujie Hu, Yanqing Liu, Jinyu Li, Sheng Zhao, Xixin Wu, Helen Meng, and Furu Wei. Autoregressive speech synthesis without vector quantization. CoRR, abs/2407.08551, 2024.
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.
- [23] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR. OpenReview.net, 2021.

[24] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR. OpenReview.net, 2023.
- [25] Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, and Wei-Ning Hsu. Voicebox: Text-guided multilingual universal speech generation at scale. In NeurIPS, 2023.
- [26] Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Eric Liu, Yichong Leng, Kaitao Song, Siliang Tang, Zhizheng Wu, Tao Qin, Xiangyang Li, Wei Ye, Shikun Zhang, Jiang Bian, Lei He, Jinyu Li, and Sheng Zhao. Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models. In ICML. OpenReview.net, 2024.
- [27] Yiwei Guo, Chenpeng Du, Ziyang Ma, Xie Chen, and Kai Yu. Voiceflow: Efficient text-to-speech with rectified flow matching. In ICASSP, pages 11121–11125. IEEE, 2024.
- [28] Shivam Mehta, Ruibo Tu, Jonas Beskow, Éva Székely, and Gustav Eje Henter. Matcha-tts: A fast TTS architecture with conditional flow matching. In ICASSP, pages 11341–11345. IEEE, 2024.
- [29] Yuan Gao, Nobuyuki Morioka, Yu Zhang, and Nanxin Chen. E3 TTS: easy end-to-end diffusion-based text to speech. In ASRU, pages 1–8. IEEE, 2023.
- [30] Keon Lee, Dong Won Kim, Jaehyeon Kim, and Jaewoong Cho. Ditto-tts: Efficient and scalable zero-shot text-to-speech with diffusion transformer. CoRR, abs/2406.11427, 2024.
- [31] Sefik Emre Eskimez, Xiaofei Wang, Manthan Thakker, Canrun Li, Chung-Hsien Tsai, Zhen Xiao, Hemin Yang, Zirun Zhu, Min Tang, Xu Tan, Yanqing Liu, Sheng Zhao, and Naoyuki Kanda. E2 TTS: embarrassingly easy fully non-autoregressive zero-shot TTS. CoRR, abs/2406.18009, 2024.
- [32] Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, and Xie Chen. F5-TTS: A fairytaler that fakes fluent and faithful speech with flow matching. CoRR, abs/2410.06885, 2024.
- [33] Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, Mingqing Gong, Peisong Huang, Qingqing Huang, Zhiying Huang, Yuanyuan Huo, Dongya Jia, Chumin Li, Feiya Li, Hui Li, Jiaxin Li, Xiaoyang Li, Xingxing Li, Lin Liu, Shouda Liu, Sichao Liu, Xudong Liu, Yuchen Liu, Zhengxi Liu, Lu Lu, Junjie Pan, Xin Wang, Yuping Wang, Yuxuan Wang, Zhen Wei, Jian Wu, Chao Yao, Yifeng Yang, Yuanhao Yi, Junteng Zhang, Qidi Zhang, Shuo Zhang, Wenjie Zhang, Yang Zhang, Zilin Zhao, Dejian Zhong, and Xiaobin Zhuang. Seed-tts: A family of high-quality versatile speech generation models. CoRR, abs/2406.02430, 2024.
- [34] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, Zhifu Gao, and Zhijie Yan. Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens. CoRR, abs/2407.05407, 2024.
- [35] Haohan Guo, Kun Liu, Feiyu Shen, Yi-Chen Wu, Feng-Long Xie, Kun Xie, and Kaituo Xu. Fireredtts: A foundation text-to-speech framework for industry-level generative speech applications. CoRR, abs/2409.03283, 2024.
- [36] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [37] Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. In EMNLP (Findings), pages 15757–15773. Association for Computational Linguistics, 2023.
- [38] Trung Dang, David Aponte, Dung N. Tran, and Kazuhito Koishida. Livespeech: Low-latency zero-shot text-to-speech via autoregressive modeling of audio discrete codes. CoRR, abs/2406.02897, 2024.
- [39] Trung Dang, David Aponte, Dung N. Tran, Tianyi Chen, and Kazuhito Koishida. Zero-shot text-to-speech from continuous text streams. CoRR, abs/2410.00767, 2024.

[40] Mateusz Lajszczak, Guillermo Cámbara, Yang Li, Fatih Beyhan, Arent van Korlaar, Fan Yang, Arnaud Joly, Álvaro Martín-Cortinas, Ammar Abbas, Adam Michalski, Alexis Moinet, Sri Karlapati, Ewa Muszynska, Haohan Guo, Bartosz Putrycz, Soledad López Gambino, Kayeon Yoo, Elena Sokolova, and Thomas Drugman. BASE TTS: lessons from building a billion-parameter text-to-speech model on 100k hours of data. CoRR, abs/2402.08093, 2024.
- [41] Avihu Dekel, Slava Shechtman, Raul Fernandez, David Haws, Zvi Kons, and Ron Hoory. Speak while you think: Streaming speech synthesis during text generation. In ICASSP, pages 11931–11935. IEEE, 2024.
- [42] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: VQ-VAE made simple. In ICLR. OpenReview.net, 2024.
- [43] Tongyi Speech Team. Funaudiollm: Voice understanding and generation foundation models for natural interaction
between humans and llms. arxiv, 2024.
- [44] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [45] Qwen Team. Qwen2.5: A party of foundation models, September 2024.
- [46] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [47] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021.
- [48] Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, et al. Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36, 2024.
- [49] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS, 2023.
- [50] Zhifu Gao, Shiliang Zhang, Ian McLoughlin, and Zhijie Yan. Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition. In Interspeech, pages 2063–2067. ISCA, 2022.
- [51] Chenpeng Du, Yiwei Guo, Feiyu Shen, Zhijun Liu, Zheng Liang, Xie Chen, Shuai Wang, Hui Zhang, and Kai Yu. Unicats: A unified context-aware text-to-speech framework with contextual vq-diffusion and vocoding. In AAAI, pages 17924–17932. AAAI Press, 2024.
- [52] Yafeng Chen, Siqi Zheng, Hui Wang, Luyao Cheng, Qian Chen, and Jiajun Qi. An enhanced res2net with local and global feature fusion for speaker verification. In Interspeech. ISCA, 2023.
- [53] Chandan K. A. Reddy, Vishak Gopal, and Ross Cutler. Dnsmos P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In ICASSP, pages 886–890. IEEE, 2022.
- [54] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 28492–28518. PMLR, 2023.
- [55] Shu-wen Yang, Heng-Jui Chang, Zili Huang, Andy T Liu, Cheng-I Lai, Haibin Wu, Jiatong Shi, Xuankai Chang, Hsiang-Sheng Tsai, Wen-Chin Huang, et al. A large-scale evaluation of speech foundation models. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.
- [56] 2noise. Chattts. https://github.com/2noise/ChatTTS, 2024.
- [57] RVC-Boss. Gpt-sovits. https://github.com/RVC-Boss/GPT-SoVITS, 2024.

[58] Zengyi Qin, Wenliang Zhao, Xumin Yu, and Xin Sun. Openvoice: Versatile instant voice cloning. CoRR, abs/2312.01479, 2023.
- [59] Daniel Lyth and Simon King. Natural language guidance of high-fidelity text-to-speech with synthetic annotations. CoRR, abs/2402.01912, 2024.
- [60] Netease Youdao. Emotivoice. https://github.com/netease-youdao/EmotiVoice, 2024.