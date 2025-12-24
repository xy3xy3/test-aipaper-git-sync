# Qwen3-Omni技术报告

Qwen团队

###### 摘要

我们推出了 Qwen3-Omni，这是一个单一的多模态模型，首次在文本、图像、音频和视频方面维持了最先进的性能，并且相较于单模态模型没有任何降低。Qwen3-Omni 在与 Qwen 系列相同尺寸的单模态模型之间的性能保持一致，尤其在音频任务上表现出色。在 36 个音频和视音频基准测试中，Qwen3-Omni 在 32 个基准测试上实现了开源最先进（SOTA）的性能，并在 22 个基准测试上获得了整体 SOTA，超越了诸如 Gemini-2.5-Pro、Seed-ASR 和 GPT-4o-Transcribe 等强大的闭源模型。Qwen3-Omni 采用了 Thinker–Talker 专家混合（MoE）架构，统一了文本、图像、音频和视频的感知与生成，生成流畅的文本和自然的实时语音。它支持 119 种语言的文本交互，19 种语言的语音理解和 10 种语言的语音生成。该系统可以处理每实例最长 40 分钟的音频录音，用于 ASR 和口语理解，提供高质量的音频和视听体验。它展现出强大的指令遵循能力，并通过用户定义的系统提示允许对对话语调和角色进行精细化定制。为了减少流媒体合成中的首包延迟，Talker 使用多码本方案自回归地预测离散语音编码。利用这些码本的表征能力，我们用轻量的因果卷积网络替代了计算密集型的分块扩散，从首个编码帧开始支持流媒体。在冷启动设置（没有先前上下文）下，Qwen3-Omni 实现了 234 毫秒的理论端到端首包延迟。为了进一步增强多模态推理，我们引入了一个思考模型，该模型明确对来自任何模态的输入进行推理。由于研究界目前缺乏通用的音频标题生成模型，我们微调了 Qwen3-Omni-30B-A3B，获得了 Qwen3-Omni-30B-A3B-Captioner，能够为任意音频输入生成详细且低幻觉的标题。Qwen3-Omni-30B-A3B、Qwen3-Omni-30B-A3B-Thinking 和 Qwen3-Omni-30B-A3B-Captioner 在 Apache 2.0 许可证下公开发布。

## 1 引言

人类能够并行感知视觉和听觉输入，认知处理这些信号，并通过文本表达、发声以及工具介导或身体行动来发出回应，从而促进与其他生物的信息交流，展现出智能。基于对单模态大模型理解和推理能力的快速发展，原生多模态系统引起了广泛关注。人类学习通常通过多种模态的协调使用来进行，互补的专业化和跨模态协同提高了学习效率。然而，当今以大语言模型为中心的多模态模型往往表现出模态之间的权衡，在某一模态上的提升伴随着其他模态的下降。

在本报告中，我们迈出了缓解此限制的一步，通过探索在当前基于大语言模型的范式内进行综合多模态训练。我们展示了联合多模态训练能够在所有模态上实现平衡，即没有特定模态的性能下降，同时显著增强了跨模态能力，例如视频理解。一个关键因素是在文本预训练的早期阶段混合单模态和跨模态数据。正如 Qwen3-Omni-30B-A3B-Base 所证明的，其文本和视觉性能在广泛基准测试中与同规模的单模态文本和视觉基础模型相当，同时展现出强大的音频能力、视听理解、跨模态“推理”以及实时视听交互。开发非降级的多模态系统是一个可实现的目标。这些系统的特征有两个关键属性：首先，它们能够在各自任务中与专业单模态模型的性能相匹配；其次，它们具备促进新颖跨模态推理和交互的能力。这些后者的能力代表了一个显著优势，因为传统单模态方法中并不存在这些能力。

![img-0.jpeg](images/1.jpeg)

Figure 1: Qwen3-Omni is a unified end-to-end model capable of processing multiple modalities, such as text, audio, image and video, and generating real-time text or speech response. Based on these features, Qwen3-Omni supports a wide range of tasks, including but not limited to voice dialogue, video dialogue, and video reasoning.

Qwen3-Omni 基于在 Qwen2.5-Omni 中引入的 Thinker-Talker 架构（Xu et al., 2025）进行构建，并引入了五个关键升级：（1）Thinker 和 Talker 都升级为专家混合模型（MoE）设计；（2）我们用自主开发的 AuT（音频变换器）编码器替换 Whisper 音频编码器，该编码器在 2000 万小时的监督音频上从零开始训练，提供更强大的通用音频表示。AuT 采用块状窗口注意力机制，以实现实时预取缓存；（3）在语音生成方面，我们采用多码本表示，其增强的容量支持对多样化声音、语用线索和声学现象的忠实建模；（4）Talker 从单轨转变为多轨编解码建模，通过 MTP 模块自回归地预测多个码本层，同时波形阶段（Code2Wav）用轻量级卷积网络（ConvNet）替换块状 DiT；（5）输入和输出音频的编码率降低到 $12.5\mathrm{Hz}$，输出编解码器支持单帧、即时语音合成。综合来看，这些变化实现了在工业规模部署下的高并发低延迟语音交互。与 Qwen2.5-Omni 相比，Qwen3-Omni 引入了四个主要改进：（1）支持超过 40 分钟的输入音频理解；（2）语言覆盖扩展至 119 种书面语言，以及分别为 19 种和 10 种的理解和生成口语语言；（3）一个全模态推理思维模型，包括音频-视频和音频仅场景；（4）流媒体性能改进，端到端延迟低至 $234~\mathrm{ms}$。值得注意的是，Qwen3-Omni 在文本和视觉模态上保持了最先进的性能，与同规模单模型 Qwen 版本相比没有性能下降。在 36 个音频和音频-视觉基准测试中，其在 32 个项目中实现了开源 SOTA，并在 22 个项目中设定过 SOTA，超越了强大的闭源系统，如 Gemini 2.5 Pro、Seed-ASR 和 GPT-4o-Transcribe。本文余下部分组织如下。第 2 节介绍了 Qwen3-Omni 的算法和架构。第 3 篇和第 4 篇分别描述了预训练和后训练数据集及流程。第 5 节报告实验结果。第 6 节将 Qwen3-Omni 与近期可比参数规模的 Qwen 模型进行比较，展示了在多模态性能上没有因模态影响导致的下降。

![img-1.jpeg](images/2.jpeg)

Figure 2: The overview of Qwen3-Omni. Qwen3-Omni adopts the Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receives high-level representations directly from Thinker. To achieve ultra-low-latency streaming, Talker autoregressively predicts a multi-codebook sequence. At each decoding step, an MTP module outputs the residual codebooks for the current frame, after which the Code2Wav renderer incrementally synthesizes the corresponding waveform, enabling frame-by-frame streaming generation.

# 2 架构

# 2.1 概述

如图2所示，Qwen3-Omni采用了Thinker-Talker架构（Xu et al., 2025）。与Qwen2.5-Omni相比，Qwen3-Omni引入了以下变化，以实现更大的可扩展性和控制力： - Thinker和Talker均采用混合专家（MoE）架构，以支持高并发和快速推理。 - Talker不再使用Thinker的高层文本表示，而仅依赖音频和视觉多模态特征。这一设计的动机在于：（i）对于文本内容，离散词元和嵌入有效上是信息等价的；（ii）多模态条件化对于音频-视频协同语音生成是必要的，例如在语音翻译中保持韵律/音色。此外，这一解耦允许外部模块（如RAG、函数调用、安全过滤器）干预Thinker的文本输出，并在需要时通过受控预处理为Talker提供文本以进行流式合成。 - 由于文本表示被解耦，Thinker和Talker可以使用不同的系统提示，独立控制Thinker的响应风格和Talker的音频风格。 - Talker采用多代码本自回归方案：Talker在每个步骤生成一个编解码器帧，而MTP模块生成剩余的残余代码本。 - Code2Wav作为一种轻量级因果卷积网络实现，简化了音频合成的最后阶段。在训练和推理过程中，Talker直接接收来自Thinker的高维多模态特征，并共享完整的对话历史。因此，系统以一个凝聚的单一模型运行，支持端到端训练和统一推理。在接下来的章节中，我们首先介绍我们新提出的AuT编码器，包括其训练方法论。然后，描述Thinker如何处理各种输入。接着，我们详细讲解Talker多代码本的流式语音生成。最后，我们强调一系列针对理解和生成模块的改进，旨在实现超低延迟的端到端流式音频推理。

# 2.2 音频变换器 (AuT)

![img-2.jpeg](images/3.jpeg)

Figure 3: The overview of AuT. AuT is an attention-encoder-decoder based auto-regressive model, which is trained from scratch on 20 million hours of supervised audio. Qwen3-Omni employs the AuT encoder as the audio encoder to obtain general purpose audio representations at a token rate of  $12.5\mathrm{Hz}$ .

音频变换器（AuT）是一种注意力编码解码模型，如图3所示，基于2000万小时的监督音频数据从零开始训练。在训练过程中，音频的滤波器组特征在注意力层之前通过Conv2D块下采样8倍，使得词元速率降低至 $12.5\mathrm{Hz}$ 。为了学习更强大且更通用的音频表示，AuT在大规模音频数据集上进行训练，这些数据集包括语音识别和音频理解任务。具体而言，训练数据包含 $80\%$ 的中文和英文伪标注ASR数据， $10\%$ 来自其他语言的ASR数据，以及 $10\%$ 的音频理解数据。为了平衡实时预取缓存的效率与离线音频任务的性能，AuT采用动态注意力窗口大小的闪存注意力，覆盖的注意力查询模式范围从1秒到8秒。在Qwen3-Omni中，我们将AuT编码器用作音频编码器，参数约为0.6B。

# 2.3 感知

文本、音频、图像和视频（不含音频）。Thinker 将文本、音频、图像和视频（不含音频）转换为一系列表示以供输入。对于文本输入，我们采用 Qwen 的分词器（Yang 等，2025a），其应用了字节级字节对编码，词汇量为 151,643 个常规词元。对于音频输入和从视频中提取的音频，我们将重采样至 $16\mathrm{kHz}$，并将原始波形转换为具有 $25\mathrm{ms}$ 窗口和 $10\mathrm{ms}$ 跨越的 128 通道梅尔谱图。我们采用 AuT 编码器作为音频编码器，该编码器在 2000 万小时的音频数据上从零开始训练，每帧音频表示大约对应于原始音频信号的 80 毫秒片段。此外，我们使用来自 Qwen3-VL 的视觉编码器，该编码器从 SigLIP2-So400m *(Tschannen 等，2025)* 初始化，具有约 5.43 亿个参数，能够处理图像和视频输入。视觉编码器在图像和视频数据的混合上训练，确保了强大的图像理解和视频 comprehension。为了尽可能完整地保留视频信息，同时与音频采样率对齐，我们以动态帧率对视频帧进行采样。

##### 视频及多模态位置嵌入（TM-RoPE）

受到 Qwen2.5-Omni 的启发，我们采用了时间对齐多模态旋转位置嵌入 (TM-RoPE)，它通过引入绝对时间信息扩展了多模态旋转位置嵌入 (M-RoPE) *(Bai et al., 2023b)*。TM-RoPE 将传统的旋转位置嵌入分解为三个独立的维度：时间、高度和宽度。在原始的 M-RoPE 公式中，时间依赖性是通过初始的 16 个旋转角度进行建模的，这些角度对应于更高的频率并表现出更强的振荡模式。尽管该设计在捕捉细粒度的局部时间变化方面有效，但可能会妨碍模型对长序列的外推能力。为了解决这一限制，我们引入了旋转角度的修改分配。具体而言，时间、高度和宽度维度交错排列，并分别分配 24、20 和 20 个旋转角度。这种重新分配促进了局部语义和长距离依赖的更平衡表示，从而提升模型的整体性能。TM-RoPE 的应用根据输入数据的具体模态进行调整。对于文本输入，这三个组件共享相同的位置标识符，使得 TM-RoPE 在功能上等同于一维 RoPE *(Su et al., 2024)*。同样，音频输入也使用共享的位置 ID，但进一步增强了绝对时间编码，每个时间 ID 对应于 80 毫秒的持续时间。对于图像数据，所有视觉标记都被分配一个固定的时间 ID，而它们独特的行和列位置则决定高度和宽度 ID。

在多模态视听流的背景下，音频组件以每80毫秒为一个时间ID进行编码。视频被视作一系列帧，其时间ID单调递增，并根据实际时间戳动态调整，以确保每个ID的时间分辨率一致为80毫秒。视频帧的高度和宽度ID的分配方式与静态图像相同。为了防止在处理多种模态时出现位置冲突，位置编号采用连续编号方式，每个后续模态的起始位置从前一个模态的最大位置ID加一开始。这种细化的位置信息编码方法使模型能够有效整合和联合建模来自不同模态的信息。与Qwen2.5-Omni将视听表示分割为固定2秒块不同，Qwen3-Omni直接使用其时间ID对这些表示进行对齐，这些时间ID明确锚定于绝对时间。这一设计选择赋予模型支持任意持续时间流输入的灵活性。

### 2.4 语音生成

在多轮对话的语音合成中，我们的Talker模块依赖于来自“思考者”组件的丰富上下文，包括历史文本词元、多模态表示以及当前回合的流式文本。这种对长上下文信息的依赖至关重要，因为高保真语音合成必须使声学属性如韵律、响度和情感适应正在进行的对话，这一原则在上下文感知生成模型中已得到充分验证。在架构上，我们的方法与*Xu et al. (2025)*不同，直接操作RVQ词元。Talker采用层次预测方案：主干网络接收当前帧的聚合码本特征，并使用线性头预测零号码本，随后多词元预测（MTP）模块生成所有残差码本。这一策略使模型能够学习声学细节的完整表示，增强语音表现力。因此，波形重建被简化为轻量级因果卷积网络（Code2Wav），在保持优越音频保真度的同时，显著减少推理延迟和计算成本（FLOPs），相比更复杂的基于DiT的声码器效果更佳。

### 2.5 流媒体与并发的设计

在流媒体视听交互场景中，首包延迟是影响用户体验的关键因素，而模型的并发能力则是降低服务成本和提升响应速度的关键。 本节讨论了Qwen3-Omni如何通过算法和架构优化提高并发能力并降低首包延迟。

##### 分块预填充与混合专家架构。

在 Qwen3-Omni 中，我们保留了在 Qwen2.5-Omni 中实现的分块预填充机制，其音频和视觉编码器能够沿时间维度输出块。在实时交互过程中，思维模块和对话模块执行异步预填充：当思维模块完成当前块的预填充时，其输出的高层次表示会立即用于异步预填充对话模块当前的块，同时思维模块预填充下一个块。这种方法显著减少了思维模块和对话模块的首次令牌时间（TTFT）。在架构上，Qwen3-Omni 中的思维模块和对话模块都采用了专家混合（MoE）设计，这对于提高服务吞吐量非常有效。与密集模型相比，MoE 架构显著减少了在处理长序列时由 KV 缓存引起的 IO 消耗，从而在生成过程中提高了每秒令牌数（TPS），并增强了并发性。

Table 1: The architectural design of Qwen3-Omni-30B-A3B and the end-to-end first-packet latency for Audio/Video (ms).

|  Module | Architecture | Params | Streaming  |
| --- | --- | --- | --- |
|  Audio Encoder | AuT | 650M | ✓  |
|  Vision Encoder | SigLIP2-So400M | 540M | -  |
|  Thinker | MoE Transformer | 30B-A3B | ✓  |
|  Talker | MoE Transformer | 3B-A0.3B | ✓  |
|  MTP | Dense Transformer | 80M | ✓  |
|  Code2wav | ConvNet | 200M | ✓  |
|  End-to-End First-Packet Latency: 234/547ms  |   |   |   |

流式多码本编解码器生成。为最小化用户接收第一生成数据包的等待时间，我们提出了一种仅基于左侧上下文的多码本生成机制。如图2所示，一旦说话者生成第一个词元，MTP模块便会预测当前帧的剩余词元。这些词元随后通过一个仅关注左侧上下文的流式多码本编解码器解码为波形。与要求在合成之前需等待说话者提供足够块上下文的Qwen2.5-Omni不同，Qwen3-Omni可以在说话者生成每个词元后立即输出波形，显著降低第一数据包延迟。轻量级MTP模块和卷积网络。MTP模块和编解码器解码器都是轻量级模块，具有低计算FLOPs并支持批量推理，非常适合高并发场景。MTP模块是一个超轻量级的固定步长自回归稠密变换器，在推理硬件上具有低内存带宽需求，从而自然支持高吞吐量请求的高效批处理。其固定步长自回归推理机制使其能够有效利用固定的KV缓存空间进行加速，从而实现低推理延迟。同时，基于卷积网络的编解码器解码器由于其卷积架构在各种推理平台上享有广泛的硬件加速支持，从而实现低延迟的高吞吐量，并能进行高效的批量推理。

Table 2: Theoretical First-Packet Latency of Qwen3-Omni wit Different Concurrency.

|   | Qwen3-Omni-30B-A3B  |   |   |
| --- | --- | --- | --- |
|   | 1Concurrency | 4Concurrency | 6Concurrency  |
|  Thinker-Talker Tail Packet Preprocessing Latency | 72/160ms | 94/180ms | 100/200ms  |
|  Thinker Time-to-First-Token (TTPT) | 88/160ms | 468/866ms | 673/1330ms  |
|  Talker Time-to-First-Token (TTPT) | 57/210ms | 145/450ms | 376/734ms  |
|  MTP Module Time Cost Per Token | 14ms | 16ms | 18ms  |
|  Codec Decoder Time Cost Per Code | 3ms | 5ms | 5ms  |
|  Overral Latency (Audio/Video) | 234/547ms | 728/1517ms | 1172/2284ms  |
|  Thinker Token Generation Rate (TPS) | 75 tokens/s | 63 tokens/s | 53 tokens/s  |
|  Talker Token Generation Rate (TPS) | 140 tokens/s | 125 tokens/s | 110 tokens/s  |
|  Generation RTF(Real Time Factor) | 0.47 | 0.56 | 0.66  |

表2展示了在典型计算资源下，Qwen3-Omni在不同并发场景下的理论首包延迟。实验是在vLLM框架（Kwon et al., 2023）上进行的，以处理并发的音视频流，并通过torch.compile和CUDA图加速对MTP模块和编解码器进行优化。多个因素会影响总首包延迟。首先，Thinker和Talker的模型大小影响它们的尾包预处理延迟（多模态数据预处理及音频和视觉编码器的推理）和首次令牌时间（TTPT）。其次，MTP模块和编解码器的架构和大小影响它们的推理延迟。由于这些组件之间存在顺序依赖，总首包延迟代表了这些各自延迟的总和。如结果所示，Thinker和Talker的MoE架构确保它们的预取延迟和TTPT在高并发下几乎不受影响。同时，MTP模块和编解码器的轻量化设计最小化了它们的计算开销，从而降低了对首包延迟的影响。此外，在初始数据包输出后，模型开始流式音频合成，$12.5\mathrm{Hz}$的令牌速率的Talker仅需一个令牌就能合成80毫秒的音频。因此，生成实时因子（RTF）通过将(1) Thinker和Talker生成一个令牌所需的时间与(2) MTP模块和编解码器每个令牌的处理时间之和除以80毫秒进行计算。如所示，RTF在不同并发水平下始终保持在1以下，确保用户接收到连续的音频响应。

# 3 预训练

Table 3: Languages and dialects support of Qwen3-Omni-30B-A3B.

|  Modality | # Langs | Languages  |
| --- | --- | --- |
|  Text | 119 | See Qwen3 for the full list.  |
|  Speech Input | 19 | ar, de, en, es, fr, id, it, ja, ko, ms, nl, pt, ru, th, tr, ur, vi, yue, zh  |
|  Speech Output | 10 | de, en, es, fr, it, ja, ko, pt, ru, zh  |

Qwen3-Omni 在一个多样化的数据集上进行预训练，该数据集涵盖了多种语言和方言，如表3所示，以及多种模态，包括图像-文本、视频-文本、音频-文本、视频-音频、视频-音频-文本和纯文本语料库。与 Qwen2.5-Omni 仅为每个任务使用单一提示不同，我们采用更广泛的自然语言提示，以增强模型的泛化能力和指令跟随能力。为了在所有模态中实现稳健的性能，我们的训练策略结合了早期预训练阶段的单模态和跨模态数据。Qwen3-Omni 的预训练分为三个不同的阶段。在第一阶段，我们锁定大语言模型（LLM）参数，专注于训练视觉和音频编码器，利用大量的音频-文本和图像-文本对来增强 LLM 内部的语义理解。在第二阶段，我们解锁所有参数，使用更广泛的多模态数据进行更全面的学习。在最后阶段，我们使用序列长度为 32,768 的数据，以增强模型理解复杂长序列数据的能力。

(1) 编码器对齐阶段 (S1)：在初始预训练阶段，Qwen3-Omni 的 LLM 组件初始化自 Qwen3（Yang et al., 2025a）的参数，视觉编码器采用自 Qwen3-VL，音频编码器则是根据 AuT 初始化。这两个编码器在固定的 LLM 上单独训练，最初都专注于训练各自的适配器，然后再训练编码器。我们放弃了 Bai et al. (2025); Xu et al. (2025) 中采用的在保持 LLM 冻结的情况下共同训练编码器和适配器的阶段，因为这种做法可能导致编码器弥补冻结的 LLM 的局限性，从而降低感知能力。 (2) 一般阶段 (S2)：第二阶段的预训练利用了一个包含大约 2 万亿词元的大规模数据集，各模态的分布如下：文本 (0.57 万亿)，音频 (0.77 万亿)，图像 (0.82 万亿)，视频 (0.05 万亿)，以及视频-音频 (0.05 万亿)。在此阶段，更多样化的多模态数据和任务的引入增强了模型对听觉、视觉、文本和视听信息的理解和互动能力。 (3) 长上下文阶段 (S3)：在最后的预训练阶段，我们将最大词元长度从 8,192 增加到 32,768，并提高了训练数据中长音频和长视频的比例。实验结果表明，这些调整显著提升了模型理解长序列数据的能力。

# 4 训练后阶段

# 4.1 思考者

后训练阶段包括一个三阶段的训练过程，使得Qwen3-Omni具备指令跟随能力。该数据集采用ChatML（OpenAI, 2022）格式，包含纯文本对话数据、视觉模态对话数据、音频模态对话数据和混合模态对话数据。在第一阶段，我们引入了轻量级的监督微调（SFT），通过针对性的指令优化来弥合预训练表示与下游任务需求之间的差距。SFT故意偏离预训练数据模式，同时保持与预训练模型的架构一致性，从而实现高效的知识迁移并保持预训练特征的完整性。第二阶段采用Qwen3中描述的强到弱蒸馏流程*(Yang et al., 2025a)*，进一步提升模型性能。该蒸馏过程由两个主要阶段组成：1. ；2. 异步策略蒸馏：在初始阶段，教师模型生成的输出结合起来以提供响应蒸馏。这有助于轻量级的学生模型获得基本推理能力，为后续的有策略训练奠定坚实基础。3. 有策略蒸馏：在第二阶段，学生模型根据采样的提示生成响应。这些有策略序列随后用于微调，通过最小化KL散度，使学生的预测logits与教师模型（Qwen3-32B或Qwen3-235B-A22B）的logits对齐。最后，我们利用GSPO*(Zheng et al., 2025)*全面增强模型在文本、图像、视频和音频等多种模态中的能力和稳定性。为了对上述模态提供反馈，我们采用两种不同类型的奖励：- 基于规则的奖励：对于可验证的多模态任务（例如，数学、编码、指令跟随），奖励信号源自一组预定义规则。设计良好的基于规则的奖励可以高精度地评估模型输出的正确性，防止奖励操纵等问题。- 基于模型的奖励：为了评估缺乏客观预定义评估指标的多模态任务的表现，我们采用LLM作为评估者的协议。对于一般任务，自动化评估者的角色由Qwen3承担，而对于视觉基础任务，则使用专业的视觉语言模型Qwen2.5-VL。为了确保更稳健且有据可依的评估，LLM评估者在适用情况下会提供给定查询的相应真实标注或参考答案。

### 4.2 讲者

我们引入了一个四阶段的训练过程用于 Talker，使 Qwen3-Omni能够与文本同时生成语音响应。所有训练数据都采用 ChatML 格式，以确保与 Thinker 的一致性。在第一阶段，我们利用数亿条带有多模态上下文的语音数据来训练 Talker，建立从多模态表示到语音的单调映射。在第二阶段，我们使用高质量数据进行持续预训练（CPT），这 alleviates 了第一阶段中因噪声数据导致的幻觉现象，并显著提高了生成语音的质量。同时，我们进行长上下文训练，以增强 Talker 处理扩展和复杂输入的能力，并生成符合上下文的语音响应。在第三阶段，为了改善多语言语音生成的泛化能力和系统稳定性，我们从多样的多语言语音样本中构建偏好对，并使用直接偏好优化（DPO）对模型进行优化 *(Rafailov et al., 2023)*。最后，我们对上述基础模型进行说话人微调，使 Talker 能够采纳特定的声音，同时提升语音响应的自然性、表现力和可控性。

### 4.3 标注生成器

字幕生成是多模态理解的基础任务，对于大型多模态模型的训练和评估至关重要。然而，现有研究绝大多数集中于视觉字幕生成，几乎忽视了音频模态。这一遗漏非常重要，因为听觉感知是人类感官体验和与世界互动的关键组成部分。为了解决这一空白，并促进更全面的多模态感知研究，我们引入了Qwen3-Omni-30B-A3B-Captioner。该模型是在大规模详细音频描述数据集上对Qwen3-Omni-30B-A3B进行微调而开发的。最终系统为任意音频输入生成详细、低幻觉的字幕。附录9.2提供了定性结果，展示了我们模型在多种声学场景中的字幕生成能力。 5 评估 在一系列模型上进行了全面评估，包括Qwen3-Omni-30B-A3B-Instruct、Qwen3-Omni-30B-A3B-Thinking，以及两个内部开发的变体，命名为Qwen3-Omni-Flash-Instruct和Qwen3-Omni-Flash-Thinking。这些“Flash”模型旨在提高计算效率和性能效能，集成新功能，尤其是对各种方言的支持。评估结果分为两个主要类别：理解（X$\rightarrow$文本）和语音生成（X$\rightarrow$语音）。

### 5.1 X$\rightarrow$文本的评估

在本节中，我们评估Qwen3-Omni理解各种多模态输入（文本、音频、视觉和视听视频）的能力，并生成文本响应。

#### 文本$\rightarrow$文本

我们对 Qwen3-Omni 在文本 $\rightarrow$ 文本的评估主要集中在一般任务、推理能力、编程能力、对齐任务、智能体和多语言任务上。具体而言，我们使用 MMLU-Redux *(Gema et al., 2024)* 和 GPQA *(Rein et al., 2023)* 来评估一般任务，AIME25 *(AIME, 2025)* 和 ZebraLogic *(Lin et al., 2025)* 来评估推理能力，MultiPL-E *(Cassano et al., 2023)* 用于编程评估，IFEval *(Zhou et al., 2023)*、Creative Writing V3 *(Paech, 2024)* 和 WritingBench *(Wu et al., 2025b)* 用于对齐任务，BFCL-v3 *(Yan et al., 2024)* 用于智能体评估，MultiIF *(He et al., 2024)* 和 PolyMath *(Wang et al., 2025c)* 用于多语言任务。

#### 音频$\rightarrow$文本

评估可以分为基础音频任务，包括自动语音识别（ASR）、语音转文本（S2TT）和音乐理解，以及高级音频任务，包括语音聊天和音频推理。对于音乐理解，我们使用 RUL-MuchoMusic *(Zang et al., 2025)* 进行模型音乐理解能力的综合评估。我们利用 MMAU *(Sakshi et al., 2024)* 和 MMSU *(Wang et al., 2025a)* 进行音频推理任务，使用 VoiceBench *(Chen et al., 2024b)* 进行语音聊天任务。我们还采用多个数据集，包括 GTZAN *(Tzanetakis and Cook, 2002)*、MTG-Jamendo 的四个子集（MTG，*Bogdanov et al. (2019)*）和 MagnaTagATune *(Law et al., 2009)* 来评估模型在各种音乐信息检索任务中的能力，包括流派识别、情感和主题识别、乐器识别以及音乐关键词标注。我们参照 MARBLE *(Yuan et al., 2023)* 中的评估集构成来评估 GTZAN、MTG-Jamendo 和 MagnaTagATune。

#### 视觉$\rightarrow$文本

模型的视觉到文本能力评估涵盖了一系列针对多样化且具有挑战性的任务的基准测试。为了评估模型在一般视觉问答中的表现，模型在 MMStar *(Chen et al., 2024a)*、HallusionBench *(Guan et al., 2024)* 和 MM-MT-Bench *(Agrawal et al., 2024)* 上进行了评估。对于数学和 STEM 推理的专业领域，我们使用 MathVista *(Lu et al., 2024)*、MathVision *(Wang et al., 2024a)*、MMMU *(Yue et al., 2023)* 和 MMMU-Pro *(Yue et al., 2024)*。模型在文档理解方面的能力通过 AI2D *(Kembhavi et al., 2016)* 和 ChartQA *(Masry et al., 2022)* 基准测试进行测量。此外，模型的数值推理和计数能力专门在 CountBench *(Paiss et al., 2023)* 上进行测试。为了评估模型在动态视觉数据上的表现，我们报告了三个长视频理解基准测试的结果：Video-MME *(Fu et al., 2024)*、LVBench *(Wang et al., 2024b)* 和 MLVU *(Zhou et al., 2025a)*。

#### 音视频转文本

为了评估模型处理动态多模态信息的能力，我们首先在 WorldSense 基准测试上评估了其性能 *(Hong et al., 2025)*。该基准旨在测量视觉和听觉信号的融合，这是在复杂开放世界环境中操作的基础能力。为了进一步检验模型的高阶认知功能，我们接着在两个视听推理基准上评估了其表现：DailyOmni *(Zhou et al., 2025b)* 和 VideoHolmes *(Cheng et al., 2025)*。

### 5.1.1 文本到文本的性能

我们将 Qwen3-Omni 与其他领先的大型语言模型（思考型或指令型）进行了比较。根据表 4 和表 5，可以明显看出，尽管参数数量较少，但 Qwen3-Omni-30B-A3B-Instruct 在 GPQA、AIME25、ZebraLogic、WritingBench 和 PolyMath 等多个基准测试中超越了更大的开源模型 Qwen3-235B-A22B 非思考型和强大的闭源模型 GPT-4o-0327。同时，Qwen3-Omni-30B-A3B-Thinking 的性能与 Gemini-2.5-Flash-Thinking 和 Qwen3-235B-A22B 非思考型 相当。此外，Qwen3-Omni-30B-A3B 在文本能力方面与其纯文本对应模型 Qwen3-30B-A3B-Instruct-2507 和 Qwen3-30B-A3B-Thinking-2507 不相上下。

Table 4: Text  $\rightarrow$  Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.

|   |   | GPT-4o-0327 | Qwen3-235B-A22B Non Thinking | Qwen3-30B-A3B -Instruct-2507 | Qwen3-Omni-30B-A3B -Instruct | Qwen3-Omni-Flash -Instruct  |
| --- | --- | --- | --- | --- | --- | --- |
|  GeneralTasks | MMLU-Redux | 91.3 | 89.2 | 89.3 | 86.6 | 86.8  |
|   |  GPQA | 66.9 | 62.9 | 70.4 | 69.6 | 69.7  |
|  Reasoning | AIME25 | 26.7 | 24.7 | 61.3 | 65.0 | 65.9  |
|   |  ZebraLogic | 52.6 | 37.7 | 90.0 | 76.0 | 76.1  |
|  Code | MultiPL-E | 82.7 | 79.3 | 83.8 | 81.4 | 81.5  |
|  Alignment Tasks | IFEval | 83.9 | 83.2 | 84.7 | 81.0 | 81.7  |
|   |  Creative Writing v3 | 84.9 | 80.4 | 86.0 | 80.6 | 81.8  |
|   |  WritingBench | 75.5 | 77.0 | 85.5 | 82.6 | 83.0  |
|  Agent | BFCL-v3 | 66.5 | 68.0 | 65.1 | 64.4 | 65.0  |
|  Multilingual Tasks | MultiIF | 70.4 | 70.2 | 67.9 | 64.0 | 64.7  |
|   |  PolyMATH | 25.5 | 27.0 | 43.1 | 37.9 | 39.3  |

Table 5: Text  $\rightarrow$  Text performance of Qwen3-Omni-Thinking and other reasoning baselines. The highest scores are shown in bold.

|   |   | Gemini-2.5-Flash Thinking | Qwen3-235B-A22B Thinking | Qwen3-30B-A3B -Thinking-2507 | Qwen3-Omni-30B-A3B -Thinking | Qwen3-Omni-Flash -Thinking  |
| --- | --- | --- | --- | --- | --- | --- |
|  General Tasks | MMLU-Redux | 92.1 | 92.7 | 91.4 | 88.8 | 89.7  |
|   |  GPQA | 82.8 | 71.1 | 73.4 | 73.1 | 73.1  |
|  Reasoning | AIME25 | 72.0 | 81.5 | 85.0 | 73.7 | 74.0  |
|   |  LiveBench 20241125 | 74.3 | 77.1 | 76.8 | 71.8 | 70.3  |
|  Code | MultiPL-E | 84.5 | 79.9 | 81.3 | 80.6 | 81.0  |
|  Alignment Tasks | IFEval | 89.8 | 83.4 | 88.9 | 85.1 | 85.2  |
|   |  Arena-Hard v2 | 56.7 | 61.5 | 56.0 | 55.1 | 57.8  |
|   |  Creative Writing v3 | 85.0 | 84.6 | 84.4 | 82.5 | 83.6  |
|   |  WritingBench | 83.9 | 80.3 | 85.0 | 85.5 | 85.9  |
|  Agent | BFCL-v3 | 68.6 | 70.8 | 72.4 | 63.2 | 64.5  |
|  Multilingual Tasks | MultiIF | 74.4 | 71.9 | 76.4 | 72.9 | 73.2  |
|   |  PolyMATH | 49.8 | 54.7 | 52.6 | 47.1 | 48.7  |

# 5.1.2 音频到文本的性能

我们将 Qwen3-Omni 与其他领先的专业模型和通用模型在自动语音识别（ASR）与语音到文本转换（S2TT）、语音聊天、音频推理和音乐理解基准上进行了比较。为简洁起见，我们将 Qwen3-Omni-Thinking 模型在 ASR 与 S2TT 以及音乐理解方面的结果推迟至附录 9.1。

Table 6: Transcription performance for Audio→Text tasks (ASR &amp; S2TT), comparing Qwen3-Omni-Instruct with the baselines. The highest scores are shown in bold.

|  | Seed -ASR | Voxtral -Mini | Voxtral -Small | GPT-4o -Transcribe | Gemini-2.5 -Pro | Qwen2.5 -Omni | Qwen3-Omni -30B-A3B-Instruct | Qwen3-Omni -Flash-Instruct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EN & ZH ASR (wer) |
| Wenetspeech net | meeting | 4.66 | 5.69 | 24.30 | 31.53 | 20.33 | 26.08 | 15.30 | 32.27 | 14.43 | 13.47 | 5.91 | 7.65 | 4.69 | 5.89 | 4.62 | 5.75 |
| Librispeech clean | other | 1.58 | 2.84 | 1.88 | 4.12 | 1.56 | 3.30 | 1.39 | 3.75 | 2.89 | 3.56 | 1.74 | 3.45 | 1.22 | 2.48 | 1.27 | 2.44 |
| CV15-en | - | 9.47 | 7.79 | 10.01 | 9.89 | 7.61 | 6.05 | 5.94 |
| CV15-zh | - | 24.67 | 19.30 | 9.84 | 8.00 | 5.13 | 4.31 | 4.28 |
| Fleurs-en | 3.40 | 3.96 | 3.77 | 3.32 | 2.94 | 3.77 | 2.72 | 2.74 |
| Fleurs-zh | 2.69 | 12.22 | 7.98 | 2.44 | 2.71 | 2.54 | 2.20 | 2.19 |
| Multilingual ASR (wer) |
| Fleurs-avg (19 lang)a | - | 15.67 | 8.09 | 4.48 | 5.55 | 14.04 | 5.33 | 5.31 |
| Lyric ASR (wer) |
| MIR-1K (vocal-only)b | 6.45 | 23.33 | 18.73 | 11.87 | 9.85 | 8.15 | 5.90 | 5.85 |
| Opencpop-test | 2.98 | 31.01 | 16.06 | 7.93 | 6.49 | 2.84 | 1.54 | 2.02 |
| S2TT (BLEU) |
| Fleurs-en2xxc | - | 30.35 | 37.85 | - | 39.25 | 29.22 | 37.50 | 36.22 |
| Fleurs-xx2en | - | 27.54 | 32.81 | - | 35.41 | 28.61 | 31.08 | 30.71 |
| Fleurs-zh2xx | - | 17.03 | 22.05 | - | 26.63 | 17.97 | 25.17 | 25.10 |
| Fleurs-xx2zh | - | 28.75 | 34.82 | - | 37.50 | 27.68 | 33.13 | 31.19 |

这些19种语言包括阿拉伯语、粤语、中文、荷兰语、英语、法语、德语、印尼语、意大利语、日语、韩语、马来语、葡萄牙语、俄语、西班牙语、泰语、土耳其语、乌尔都语、越南语。 转录已转换为简体中文。 结果涵盖了15种语言的翻译：阿拉伯语、粤语、中文、英语、法语、德语、印尼语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语、泰语、越南语。符号“en2xx”表示从英语翻译到其他14种目标语言，其中“xx”为剩余语言代码。如表6所示，Qwen3-Omni-Instruct在Librispeech、Wenetspeech、Fleurs、CommonVoice、Opencpop-test和MIR-1K（人声）上实现了最先进的英语和中文自动语音识别及歌词自动语音识别性能。同时，在多语言自动语音识别和语音转文本方面，Qwen3-Omni的性能优于或可比于其他专业或通用模型，如Voxtral-Small和Gemini-2.5-Pro。这些结果展示了Qwen3-Omni在语音识别和语音翻译方面的强大表现。此外，在表7所示的VoiceBench上，Qwen3-Omni-Thinking取得了令人印象深刻的89.5的平均分，超过了除Gemini-2.5-Pro（89.6）外的所有其他音频语言模型。这展示了我们模型在语音互动方面的强大能力。Qwen3-Omni在音频推理方面也表现出色，在MMAU基准上超越了强大的封闭源模型Gemini-2.5-Pro和Gemini-2.5-Flash，以及在MMSU上超越了Gemini-2.5-Flash和GPT-4o-Audio。这些结果展示了Qwen3-Omni在一般音频理解和推理方面的强大能力。

Table 7: Voice interaction and audio reasoning performance for Audio→Text tasks, comparing Qwen3-Omni with the baselines. The highest scores are shown in bold.

|   | GPT-4o -Audio | Gemini-2.5 -Flash | Gemini-2.5 -Pro | Qwen2.5 -Omni | Qwen3-Omni -30B-A3B-Instruct | Qwen3-Omni -30B-A3B-Thinking | Qwen3-Omni -Flash-Instruct | Qwen3-Omni -Flash-Thinking  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  VoiceBench  |   |   |   |   |   |   |   |   |
|  AlpacaEval | 95.6 | 96.1 | 94.3 | 89.9 | 94.8 | 96.4 | 95.4 | 96.8  |
|  CommonEval | 89.8 | 88.3 | 88.4 | 76.7 | 90.8 | 90.5 | 91.0 | 90.9  |
|  WildVoice | 91.6 | 92.1 | 93.4 | 77.7 | 91.6 | 90.5 | 92.3 | 90.9  |
|  SD-QA | 75.5 | 84.5 | 90.1 | 56.4 | 76.9 | 78.1 | 76.8 | 78.5  |
|  MMSU | 80.3 | 66.1 | 71.1 | 61.7 | 68.1 | 83.0 | 68.4 | 84.3  |
|  OpenBookQA | 89.2 | 56.9 | 92.3 | 80.9 | 89.7 | 94.3 | 91.4 | 95.0  |
|  BBH | 84.1 | 83.9 | 92.6 | 66.7 | 80.4 | 88.9 | 80.6 | 89.6  |
|  IFEval | 76.0 | 83.8 | 85.7 | 53.5 | 77.8 | 80.6 | 75.2 | 80.8  |
|  AdvBench | 98.7 | 98.9 | 98.1 | 99.2 | 99.3 | 97.2 | 99.4 | 98.9  |
|  Overall | 86.8 | 83.4 | 89.6 | 73.6 | 85.5 | 88.8 | 85.6 | 89.5  |
|  Audio Reasoning  |   |   |   |   |   |   |   |   |
|  MMAU-v05.15.25 | 62.5 | 71.8 | 77.4 | 65.5 | 77.5 | 75.4 | 77.6 | 76.5  |
|  MMSU | 56.4 | 70.2 | 77.7 | 62.6 | 69.0 | 70.2 | 69.1 | 71.3  |

在音乐理解方面，我们在表8中将Qwen3-Omni-Instruct与通用音频语言模型和专业模型进行比较。对于MTG-Jamendo和MagnaTagATune上的多标签分类任务，我们使用微观F1与BERT类音乐专家进行比较，而不是AP/AUROC，因为语言模型输出的是离散的标签集，而不是排名基准指标所需的经过校准的每个标签概率/得分。表8显示，Qwen3-Omni-Instruct在RUL-MuchoMusic上达到了最先进的性能。在GTZAN、MTG-Jamendo和MagnaTagATune上，Qwen3-Omni-Instruct的得分也显著超过其他音频语言模型，包括Gemini-2.5-Pro和GPT-4o-Audio，以及在各自数据集上进行探测的自监督音乐专家模型。这些结果展示了Qwen3-Omni-Instruct在多种音乐理解任务中的卓越能力。

Table 8: Music understanding performance for Audio→Text tasks, comparing Qwen3-Omni-Instruct with baselines. The highest scores are shown in bold.

|   | Best Specialist Models | GPT-4o -Audio | Gemini-2.5 -Pro | Qwen2.5 -Omni | Qwen3-Omni -30B-A3B-Instruct | Qwen3-Omni -Flash-Instruct  |
| --- | --- | --- | --- | --- | --- | --- |
|  RUL-MuchoMusic | 47.6 (Audio Flamingo 3) (Goel et al., 2025) | 36.1 | 49.4 | 47.3 | 52.0 | 52.1  |
|  GTZAN | 87.9 (CLaMP 3) (Wu et al., 2025a) | 76.5 | 81.0 | 81.7 | 93.0 | 93.1  |
|  MTC Genre | 35.8 (MuQ-MuLan) (Zhu et al., 2025) | 25.3 | 32.6 | 32.5 | 39.0 | 39.5  |
|  MTCMood/Theme | 10.9 (MuQ-MuLan) (Zhu et al., 2025) | 11.3 | 14.1 | 8.9 | 21.0 | 21.7  |
|  MTCInstrument | 39.8 (MuQ-MuLan) (Zhu et al., 2025) | 34.2 | 33.0 | 22.6 | 40.5 | 40.7  |
|  MTCTop50 | 33.2 (MuQ-MuLan) (Zhu et al., 2025) | 25.0 | 26.1 | 21.6 | 36.7 | 36.9  |
|  MagnaTagATune | 41.6 (MuQ) (Zhu et al., 2025) | 29.2 | 28.1 | 30.1 | 44.3 | 46.8  |

# 5.1.3 视觉到文本的表现

为了全面评估视觉 $\rightarrow$ 文本的能力，我们将 Qwen3-Omni-Instruct 与 Qwen2.5-VL-72B 以及其他表现良好的封闭源视觉语言模型进行了比较。如表 9 所示，Qwen3-Omni-Instruct 的性能与 Qwen2.5-VL-72B 相当，并且在与数学和 STEM 相关的任务（如 MMMU-Pro、MathVista mini 和 MATH-Visionfull）中表现优于包括 GPT4-o 和 Gemini-2.0-Flash 在内的其他视觉语言模型。这些结果揭示了我们模型在图像理解和推理任务上的卓越能力。为了评估其能力，我们将 Qwen3-Omni-Thinking 的表现与多种最先进的推理模型进行了比较。比较结果总结在表 10 中，表明我们提出的模型取得了显著的进展。例如，在数学和 STEM 基准测试中，它比 Qwen3-Omni-Instruct 基线表现提升了 4.4 分。此外，我们的 Qwen3-Omni-30B-A3B-Thinking 模型的表现达到了与大规模基线相当的水平，这突显了其在效果和计算效率之间的优秀平衡。目前模型的一个局限性是在长视频基准测试中的表现不佳。这一不足之处源于两个架构限制：位置外推能力有限和上下文长度受限。解决这些限制是未来工作的关键目标。

Table 9: Vision  $\rightarrow$  Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.

|  Datasets | GPT4-o | Gemini-2.0-Flash | Qwen2.5-VL 72B | Qwen3-Omni-30B-A3B -Instruct | Qwen3-Omni-Flash -Instruct  |
| --- | --- | --- | --- | --- | --- |
|  General Visual Question Answering  |   |   |   |   |   |
|  MMStar | 64.7 | 71.4 | 70.8 | 68.5 | 69.3  |
|  HallusionBench | 55.0 | 56.3 | 55.2 | 59.7 | 60.4  |
|  MM-MT-Bench | 7.7 | 6.7 | 7.6 | 7.4 | 7.6  |
|  Math & STEM  |   |   |   |   |   |
|  MMMUval | 69.1 | 71.3 | 70.2 | 69.1 | 69.8  |
|  MMMU-Pro overall | 51.9 | 56.1 | 51.1 | 57.0 | 58.2  |
|  MathVista mini | 63.8 | 71.4 | 74.8 | 75.9 | 77.4  |
|  MATH-Visionfull | 30.4 | 48.6 | 38.1 | 56.3 | 57.3  |
|  Documentation Understanding  |   |   |   |   |   |
|  AI2Dw.M. | 84.6 | 86.7 | 88.7 | 85.2 | 86.4  |
|  ChartQA test Avg. | 86.7 | 64.6 | 89.5 | 86.8 | 87.1  |
|  Counting  |   |   |   |   |   |
|  CountBench | 87.9 | 91.2 | 93.6 | 90.0 | 90.0  |
|  Video Understanding  |   |   |   |   |   |
|  Video-MMEw/o sub | 71.9 | 72.4 | 73.3 | 70.5 | 71.4  |
|  LVBench | 30.8 | 57.9 | 47.3 | 50.2 | 51.1  |
|  MLVU | 64.6 | 71.0 | 74.6 | 75.2 | 75.7  |

Table 10: Vision  $\rightarrow$  Text performance of Qwen3-Omni-Thinking and other reasoning baselines. The highest scores are shown in bold.

|  Datasets | Gemini-2.5-Flash -Thinking | InternVL-3.5-241B-A28B | Qwen3-Omni-30B-A3B -Thinking | Qwen3-Omni-Flash -Thinking  |
| --- | --- | --- | --- | --- |
|  General Visual Question Answering  |   |   |   |   |
|  MMStar | 75.5 | 77.9 | 74.9 | 75.5  |
|  HallusionBench | 61.1 | 57.3 | 62.8 | 63.4  |
|  MM-MT-Bench | 7.8 | - | 8.0 | 8.0  |
|  Math & STEM  |   |   |   |   |
|  MMMUval | 76.9 | 77.7 | 75.6 | 75.0  |
|  MMMU-Pro overall | 65.8 | - | 60.5 | 60.8  |
|  MathVista mini | 77.6 | 82.7 | 80.0 | 81.2  |
|  MATH-Visionfull | 62.3 | 63.9 | 62.9 | 63.8  |
|  Documentation Understanding  |   |   |   |   |
|  AI2Dw.M. | 88.6 | 87.3 | 86.1 | 86.8  |
|  ChartQA test Avg. | - | 88.0 | 89.5 | 89.3  |
|  Counting  |   |   |   |   |
|  CountBench | 88.6 | - | 88.6 | 92.5  |
|  Video Understanding  |   |   |   |   |
|  Video-MMEw/o sub | 79.6 | 72.9 | 69.7 | 69.8  |
|  LVBench | 64.5 | - | 49.0 | 49.5  |
|  MLVU | 82.1 | 78.2 | 72.9 | 73.9  |

# 5.1.4 音视频到文本的性能

如表11所示，实验结果验证了Qwen3-Omni在多种视听任务中的有效性。一般来说，Qwen3-Omni-Instruct在WorldSense基准测试中实现了最先进的性能，显著超过其他Omni模型。这一结果证明了其在基础多模态整合中的有效性。此外，如表12所示，该模型在复杂推理任务中表现出增强的性能，特别是在需要对互相关联的音频和视觉信息进行推理的基准测试上。这些发现共同表明，Qwen3-Omni在高级感知和推理方面具有相当大的潜力，适用于现实世界的场景。

Table 11: AudioVisual  $\rightarrow$  Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.

|  Datasets | Previous Open-source SoTA | Gemini-2.5-Flash | Qwen2.5-Omni | Qwen3-Omni-30B-A3B -Instruct | Qwen3-Omni-Flash -Instruct  |
| --- | --- | --- | --- | --- | --- |
|  WorldSense | 47.1(Yang et al., 2025b) | 50.9 | 45.4 | 54.0 | 54.1  |

Table 12: AudioVisual  $\rightarrow$  Text performance of Qwen3-Omni-30B-A3B-Thinking and other reasoning baselines. The highest scores are shown in bold.

|  Datasets | Previous Open-source SoTA | Gemini-2.5-Flash -Thinking | Qwen3-Omni-30B-A3B -Thinking | Qwen3-Omni-Flash -Thinking  |
| --- | --- | --- | --- | --- |
|  DailyOmni | 69.8(Tang et al., 2025) | 72.7 | 75.8 | 76.2  |
|  VideoHolmes | 55.6(Tang et al., 2025) | 49.5 | 57.3 | 57.3  |

# 5.2 $\mathbf{X}\rightarrow$ 语音评估

在本节中，我们评估Qwen3-Omni的语音生成能力。由于缺乏相关评估，语音生成的评估主要集中在给定文本的语音生成，类似于文本转语音（TTS），关注以下三个方面：- 零样本语音生成：我们在SEED（Anastassiou等，2024）上评估模型在零样本语音生成中的内容一致性（WER）和说话人相似性（SIM）。- 多语言语音生成：我们在MiniMax多语言测试集（Zhang等，2025）上评估模型在零样本多语言语音生成中的内容一致性和说话人相似性。- 跨语言语音生成：我们在CV3-Eval（Du等，2025）上评估模型在零样本跨语言语音生成中的内容一致性。

# 5.2.1 零样本语音生成的评估

我们将Qwen3-Omni与最先进的零-shot TTS系统进行了比较。如表13所示，Qwen3-Omni表现出具竞争力的性能，突显了其通过预训练和持续预训练所发展出的强大的语音理解和生成能力。此外，经过强化学习（RL）优化，Qwen3-Omni在生成稳定性方面实现了显著提升，在测试集en中取得了最佳表现。

Table 13: Zero-Shot Speech Generation on Seed-TTS Test Set. The highest scores are shown in bold.

| Datasets | Model | Performance |
| --- | --- | --- |
| Content Consistency |
| SEED test-zh | test-en | Seed-TTSICL (Anastassiou et al., 2024) | 1.11 | 2.24 |
| Seed-TTSRL (Anastassiou et al., 2024) | 1.00 | 1.94 |
| MaskGCT (Wang et al., 2024c) | 2.27 | 2.62 |
| E2 TTS (Eskimez et al., 2024) | 1.97 | 2.19 |
| F5-TTS (Chen et al., 2024c) | 1.56 | 1.83 |
| Spark TTS (Wang et al., 2025b) | 1.20 | 1.98 |
| CosyVoice 2 (Du et al., 2024) | 1.45 | 2.57 |
| CosyVoice 3 (Du et al., 2025) | 0.71 | 1.45 |
| Qwen2.5-Omni-7B (Xu et al., 2025) | 1.42 | 2.33 |
| Qwen3-Omni-30B-A3B | 1.07 | 1.39 |

# 5.2.2 多语言语音生成的评估

Qwen3-Omni 支持10种语言的语音生成。我们对其在多语言语音生成方面的性能进行了评估，与 MiniMax-Speech 和 ElevenLabs Multilingual v2 模型进行了对比。如表14所示，Qwen3-Omni 在中文、英文和法文等语言上显著超越了这些模型，而在其他语言中也表现出竞争力。这些发现表明，Qwen3-Omni 在所有评估语言中生成了一致稳定且类人声音的克隆语音。

Table 14: Multilingual Speech Generation on MiniMax Multilingual Test Set. The highest scores are shown in bold.

|  Language | Content Consistency |   |   | Speaker Similarity  |   |   |
| --- | --- | --- | --- | --- | --- | --- |
|   |  Qwen3-Omni -30B-A3B | MiniMax | ElevenLabs | Qwen3-Omni -30B-A3B | MiniMax | ElevenLabs  |
|  Chinese | 0.716 | 2.252 | 16.026 | 0.772 | 0.780 | 0.677  |
|  English | 1.069 | 2.164 | 2.339 | 0.773 | 0.756 | 0.613  |
|  German | 0.777 | 1.906 | 0.572 | 0.738 | 0.733 | 0.614  |
|  Italian | 1.067 | 1.543 | 1.743 | 0.742 | 0.699 | 0.579  |
|  Portuguese | 1.872 | 1.877 | 1.331 | 0.770 | 0.805 | 0.711  |
|  Spanish | 1.765 | 1.029 | 1.084 | 0.744 | 0.762 | 0.615  |
|  Japanese | 3.631 | 3.519 | 10.646 | 0.763 | 0.776 | 0.738  |
|  Korean | 1.670 | 1.747 | 1.865 | 0.778 | 0.776 | 0.700  |
|  French | 2.505 | 4.099 | 5.216 | 0.689 | 0.628 | 0.535  |
|  Russian | 3.986 | 4.281 | 3.878 | 0.759 | 0.761 | 0.676  |

# 5.2.3 跨语言语音生成的评估

Qwen3-Omni 不仅支持多语种语音克隆，还支持跨语言语音克隆。我们对其在跨语言语音生成中的性能进行了评估，比较了 CosyVoice2 和 CosyVoice3。如表 15 所示，Qwen3-Omni 在任何语言到英语（any-to-en）和任何语言到韩语（any-to-ko）的语音克隆中均优于 CosyVoice3。值得注意的是，在任何语言到日语（any-to-ja）的任务中，尽管 CosyVoice3 将所有日文字符转换为音标假名，但 Qwen3-Omni 即使在没有文本标准化的情况下也能与 CosyVoice3 的性能相当。这些结果凸显了 Qwen3-Omni 在跨语言语音生成中的优越性，展示了其在多种语言环境中的适应能力。

Table 15: Cross-Linguial Speech Generation on CosyVoice3 Cross-Linguial Test Set. The highest scores are shown in bold.

|  Language | Qwen3-Omni-30B-A3B | CosyVoice3 | CosyVoice2  |
| --- | --- | --- | --- |
|  en-to-zh | 5.37 | 5.09 | 13.5  |
|  ja-to-zh | 3.32 | 3.05 | 48.1  |
|  ko-to-zh | 0.99 | 1.06 | 7.70  |
|  zh-to-en | 2.76 | 2.98 | 6.47  |
|  ja-to-en | 3.31 | 4.20 | 17.1  |
|  ko-to-en | 3.34 | 4.19 | 11.2  |
|  zh-to-ja | 8.29 | 7.08 | 13.1  |
|  en-to-ja | 7.53 | 6.80 | 14.9  |
|  ko-to-ja | 4.24 | 3.93 | 5.86  |
|  zh-to-ko | 5.13 | 14.4 | 24.8  |
|  en-to-ko | 4.96 | 5.87 | 21.9  |
|  ja-to-ko | 6.23 | 7.92 | 21.5  |

# 6 评估跨模态的不降级性

标准化的数据集成方法因不同模态的异质性而变得不切实际，每种模态需要特定的预训练目标和优化技术。为了确保公平和严格的评估，我们设计了一项受控的比较研究。我们的方法涉及预训练三个参数数量相匹配的模型：一个仅文本的基准，一个仅视觉的基准，以及一个多模态的“Omni”模型。为了隔离多模态的影响，所有混淆变量都经过细致控制。具体而言，Omni模型在与单模态基准相同的文本和视觉语料库上进行训练。此外，我们在所有模型中对关键训练参数进行了一致对齐，包括学习率调度、批量大小，以及每个模态的有效训练轮数，通过调整数据采样比例进行归一化。因此，在我们的实验中，唯一的区分因素是Omni模型在预训练阶段包含了补充的音频和视听数据。结果如表16所示，我们评估了涵盖多种模态的综合基准，包括文本模态（一般任务、数学和STEM任务、编码任务、多语言任务）、视觉模态（大学水平问题、OCR相关任务）和视频模态（视频理解任务）。实验结果不仅表明，在文本预训练的早期阶段混合单模态和跨模态数据可以在所有模态中实现更好的性能，还表明联合多模态训练能够在不同模态之间实现互相增强，从而提高单一模态的性能。这充分展示了Qwen3-Omni在各种评估标准下的多样性和鲁棒性。由于实验成本高昂，我们未能在所有模型规模上进行全面的遍历。根据表16及我们的内部实验，我们观察到：（1）在预训练期间的早期进行多模态集成使得语言模型可以与视觉或音频共同训练而不影响语言能力；（2）文本模态的加入显著提高了视觉和音频的性能。在限制条件下，我们未观察到添加视觉或音频信号对语言能力有可测量的提升；（3）根据经验，添加音频数据持续改善了MMMU基准和OCR相关任务上的视觉性能。

Table 16: We compare the performance of 30A3 models that are contemporaneous and identical in size in Qwen series. To ensure experimental rigor, all models were trained under the same schedule, using identical datasets for their respective modalities and exactly matched training compute (FLOPs).

|   | Datasets | Qwen3-30B-A3B -Base-202507 | Qwen3-VL-30B-A3B -Base-202507 | Qwen3-Omni-30B-A3B -Base-202507  |
| --- | --- | --- | --- | --- |
|  General Tasks | MMLU | 81.24 | - | 81.69  |
|   |  MMLU-Redux | 80.17 | - | 80.60  |
|   |  MMLU-Pro | 61.81 | - | 61.57  |
|   |  SuperGPQA | 38.24 | - | 40.14  |
|   |  BBH | 83.79 | - | 83.53  |
|  Math & STEAM Tasks | GSM8K | 90.83 | - | 91.36  |
|   |  MATH | 60.84 | - | 60.42  |
|  Coding Tasks | EvalPlus | 69.70 | - | 73.96  |
|   |  MultiPL-E | 65.75 | - | 64.79  |
|   |  MBPP | 72.60 | - | 72.60  |
|   |  CRUX-O | 66.94 | - | 69.06  |
|  Multilingual Tasks | MGSM | 78.75 | - | 79.93  |
|   |  INCLUDE | 65.17 | - | 64.73  |
|  College-level Problems | MMMUval | - | 57.22 | 59.33  |
|  General Visual Question Answering | MMStar | - | 67.2 | 69.6  |
|   |  RealWorldQAavg | - | 73.98 | 71.89  |
|  OCR-related Tasks | AI2D | - | 85.88 | 86.62  |
|   |  TextVQAval | - | 81.67 | 81.65  |
|   |  DocVQAtest | - | 95.19 | 95.27  |
|   |  InfoVQAtest | - | 81.17 | 83.31  |
|   |  ChartQAtest Avg | - | 87.12 | 87.52  |
|   |  OCRBench | - | 85.8 | 86.0  |
|  Video Understanding Tasks | Video-MMEw/o sub | - | 69.22 | 69.25  |
|   |  MVBench | - | 71.87 | 69.50  |
|   |  LVBench | - | 48.61 | 51.07  |

# 7 结论

在本文中，我们介绍了 Qwen3-Omni-30B-A3B、Qwen3-Omni-30B-A3B-Thinking、Qwen3-Omni-Flash-Instruct 和 Qwen3-Omni-Flash-Thinking 模型。Qwen3-Omni-30B-A3B 在文本和视觉基准测试中与最新同尺寸的单模态 Qwen 模型相匹配或超越。值得注意的是，在音频处理和对话基准测试中，它在 32 项基准测试中达到开源系统的最先进性能，并且与强大的专有对手 Gemini-2.5-Pro 相当或更好。Qwen3-Omni-30B-A3B Thinking 变体在文本、视觉和音视觉推理等复杂任务上实现了进一步的提升。除了准确性外，该模型支持 119 种文本语言、19 种语音识别语言和 10 种语音合成语言，并能够实现最长 40 分钟的音频理解和互动会话。得益于其流式架构和多码本设计，Qwen3-Omni 在 30B-A3B 规模下仍能实现 234 毫秒的端到端首次数据包延迟。研究领域通常在专业化和整合之间循环。在此背景下，我们认为 Qwen3-Omni 代表了一个里程碑：据我们所知，它提供了首次证据，表明完全整合的端到端多模态训练可以在不降低核心语言能力和其他模态的情况下实现。我们渴望与社区分享这些发现，并希望它们能激发进一步的研究。在实际使用中，Qwen3-Omni-30B-A3B 提供强大的文本和视觉能力，稳健可靠的自动语音识别，支持超过 20 种语言的互动语音，极低的互动使用首次数据包延迟，以及稳定而自然的语音合成。重要的是，它在跨模态推理、较低的端到端延迟以及较低的系统复杂性和成本方面展现出相对于级联流程的优势。在未来的工作中，我们将从多个方面进一步提升该模型，包括多说话人自动语音识别、视频光学字符识别、音视频主动学习，以及增强对基于智能体的工作流和功能调用的支持。

## 8 名作者

核心贡献者：徐金，郭志芳，胡航睿，楚云飞，王雄，何金正，王宇轩，史先，何婷，朱欣发，吕元君，王永琦，郭大恺，王赫，马林翰，张培，张新宇，高宏琨，郭子山，杨保松，张彬，马子扬，魏喜平，白帅，陈可琴，刘雪晶，王鹏，杨明坤，刘大义恒，任兴章，郑博，门睿，周凡，余博文，杨剑新，余乐，周景仁，林俊扬。

贡献者：杨安，李安锋，陈贝，张北辰，林彬，惠宾元，王博涵，吴布晓，吴晨飞，陈成，强陈，袁晨汉，李晨浩，吕晨旭，郑初杰，陈达仁，刘大义恒，郭大克，黄飞，朱各正阳，周广东，张航，涂洪建，钟海敏，左佳龙，涂建洪，张建伟，冷佳怡，周静，周静仁，邓凯，杨恪鑫，言昆，郑来文，谢雷，邓亮浩，孟凌晨，李美，洪苗，薛鸣峰，李敏生，李铭泽，张培阳，刘鹏，王鹏飞，袁瑞彬，胡睿，徐瑞阳，黄启东，朱琴，沈阙，李申，刘世玄，宋思博，张思齐，陈松，郝苏，唐天逸，葛文斌，姚文涛，丁威，王伟，邓晓东，陈晓彤，李晓，杨先，牛新耀，郭绪东，乐欣，王学春，金续通，任轩辰，范杨，刘杨，苏阳，刘彦涛，吴毅，张艺昌，陈一磊，董益铭，张颖儿，曹义忠，孙予冲，王月章，王宇昊，刘昱琼，朱元志，陈玉翔，蔡玉轩，刘玉轩，崔泽宇，李铮，邢正浩，张震如，邱子悦，李兆海，李志，杨志博，王志海，周志鹏。

# 9 附录

# 9.1 对语音和音乐理解的更多评估

本节报告了Qwen3-Omni-thinking模型在AS-R/S2TT和音乐领域任务上的表现。如表17和表18所示，在ASR/S2TT和音乐理解领域，Qwen3-Omni-Thinking模型的表现不及其Instruct版本，这表明对于这些以感知为主的任务，复杂推理过程的参与并未带来性能提升。实际上，这可能甚至增加了幻觉的倾向。

Table 17: Transcription performance for Audio→Text tasks (ASR &amp; S2TT), comparing Qwen3-Omni-Thinking with the baselines. The highest scores are shown in bold.

|  | Seed -ASR | Voxtral -Mini | Voxtral -Small | GPT-4o -Transcribe | Gemini-2.5 -Pro | Qwen2.5 -Omni | Qwen3-Omni -30B-A3B-Thinking | Qwen3-Omni -Flash-Thinking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EN & ZH ASR (wer) |
| Wenetspeech net i meeting | 4.66 | 5.69 | 24.30 | 31.53 | 20.33 | 26.08 | 15.30 | 32.27 | 14.43 | 13.47 | 5.91 | 7.65 | 6.16 | 8.17 | 6.85 | 8.42 |
| Librispeech clean i other | 1.58 | 2.84 | 1.88 | 4.12 | 1.56 | 3.30 | 1.39 | 3.75 | 2.89 | 3.56 | 1.74 | 3.45 | 2.22 | 4.38 | 1.82 | 4.01 |
| CV15-en | - | 9.47 | 7.79 | 10.01 | 9.89 | 7.61 | 10.44 | 10.52 |
| CV15-zh | - | 24.67 | 19.30 | 9.84 | 8.00 | 5.13 | 6.25 | 6.61 |
| Fleurs-en | 3.40 | 3.96 | 3.77 | 3.32 | 2.94 | 3.77 | 3.75 | 3.67 |
| Fleurs-zh | 2.69 | 12.22 | 7.98 | 2.44 | 2.71 | 2.54 | 2.73 | 2.57 |
| Multilingual ASR (wer) |
| Fleurs-avg (19 lang)a | - | 15.67 | 8.09 | 4.48 | 5.55 | 14.04 | 8.63 | 8.88 |
| Lyric ASR (wer) |
| MIR-1K (vocal-only)b | 6.45 | 23.33 | 18.73 | 11.87 | 9.85 | 8.15 | 11.15 | 10.47 |
| Opencpop-test | 2.98 | 31.01 | 16.06 | 7.93 | 6.49 | 2.84 | 6.11 | 4.52 |
| S2TT (BLEU) |
| Fleurs-en2xxc | - | 30.35 | 37.85 | - | 39.25 | 29.22 | 36.24 | 36.04 |
| Fleurs-xx2en | - | 27.54 | 32.81 | - | 35.41 | 28.61 | 30.50 | 30.22 |
| Fleurs-zh2xx | - | 17.03 | 22.05 | - | 26.63 | 17.97 | 23.74 | 23.77 |
| Fleurs-xx2zh | - | 28.75 | 34.82 | - | 37.50 | 27.68 | 34.51 | 34.49 |

a 这19种语言包括阿拉伯语、粤语、汉语、荷兰语、英语、法语、德语、印尼语、意大利语、日语、韩语、马来语、葡萄牙语、俄语、西班牙语、泰语、土耳其语、乌尔都语、越南语。 b 转录内容转换为简体中文。 c 结果涵盖15种语言的翻译：阿拉伯语、粤语、汉语、英语、法语、德语、印尼语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语、泰语、越南语。标记中，“en2xx”表示从英语翻译到其他14种目标语言，其中“xx”代表剩余语言代码。

Table 18: Music understanding performance for Audio→Text tasks, comparing Qwen3-Omni-Thinking with baselines. The highest scores are shown in bold.

|   | Best Specialist Models | GPT-4o -Audio | Gemini-2.5 -Pro | Qwen2.5 -Omni | Qwen3-Omni -30B-A3B-Thinking | Qwen3-Omni -Flash-Thinking  |
| --- | --- | --- | --- | --- | --- | --- |
|  RUL-MuchoMusic | 47.6 (Audio Flamingo 3) (Goel et al., 2025) | 36.1 | 49.4 | 47.3 | 48.3 | 48.4  |
|  GTZAN | 87.9 (CLaMP 3) | 76.5 | 81.0 | 81.7 | 89.0 | 89.0  |
|  Acc. | (Wu et al., 2025a) |  |  |  |  |   |
|  MTG Genre | 35.8 (MuQ-MuLan) | 25.3 | 32.6 | 32.5 | 32.5 | 33.0  |
|  Micro F1 | (Zhu et al., 2025) |  |  |  |  |   |
|  MTG Mood/Theme | 10.9 (MuQ-MuLan) | 11.3 | 14.1 | 8.9 | 14.3 | 15.4  |
|  Micro F1 | (Zhu et al., 2025) |  |  |  |  |   |
|  MTG Instrument | 39.8 (MuQ-MuLan) | 34.2 | 33.0 | 22.6 | 36.4 | 36.4  |
|  Micro F1 | (Zhu et al., 2025) |  |  |  |  |   |
|  MTG Top50 | 33.2 (MuQ-MuLan) | 25.0 | 26.1 | 21.6 | 29.1 | 29.3  |
|  Micro F1 | (Zhu et al., 2025) |  |  |  |  |   |
|  MagnaTagATune | 41.6 (MuQ) | 29.2 | 28.1 | 30.1 | 32.2 | 32.6  |
|  Micro F1 | (Zhu et al., 2025) |  |  |  |  |   |

# 9.2 Qwen3-Omni-30B-A3B-Captioner 的定性结果

在本节中，我们通过三个代表性的案例研究来展示我们微调后的 Qwen3-Omni-30B-A3B-Captioner 的性能。所选场景旨在测试模型在以下方面的能力：(1) 分析富有表现力的言语，(2) 解释复杂的听觉场景和音效，以及 (3) 为包含言语、音乐和环境声音的复合音频进行字幕生成。为了可重复性，原始音频样本可以从我们的公共 GitHub 仓库获取。

表达性语言分析

音频剪辑在一个工作室环境中开始，伴有微弱而持续的电子噪音和低频的嗡嗡声，表明录音条件专业但并非完全无噪。男性发言者的声音清晰、充满活力且极具戏剧性，他以坚定的“对”开场，音调尖锐上扬，立刻确立了一种表演性和自觉的语气。他接着说道：“这就是我。万人敬仰的太乙真人”，对“万人敬仰”使用了夸张的强调和明显的上扬语调，以传达一种虚张声势和滑稽的勇敢。他的发音清晰，声音中频丰富，暗示了采用了近麦克风录音，并略带数字混响以增加戏剧效果。经过短暂的故意停顿，发言者转向一种更自谦但仍然俏皮的语气：“虽然有点婴儿肥。但也掩不住我逼人的帅气”。在这里，语调变得更轻松更会话，轻微的下降语调用于“婴儿肥”，而在“逼人的帅气”上则带有俏皮的上扬强调，突出谦逊与自夸之间的喜剧对比。交付中富有表现力的语调和清晰的发音，保持了在混音中的一致性。在整个讲话过程中，录音保持干净且无失真，没有背景噪声、环境提示或观众声音。声音在立体声场中处于中心位置，混响和均衡的使用创造了戏剧性和空间深度的感觉，而电子嗡嗡声则隐约可察，却提示着专业录音设备和后期处理。剪辑以突然的完全静音结尾，没有拖延声音或渐弱，表明这是一次故意且精确的剪辑。总之，这段音频由一位男性演讲者用普通话进行自我介绍，描绘了传说中的道教人物太乙真人，采用了滑稽、夸张的方式。这个表演将夸张的勇敢与自嘲幽默相结合，运用了戏剧性的语调、表现力十足的节奏以及专业的录音制作，创造了一个令人难忘的娱乐角色。缺乏背景噪声和精心策划的编辑增强了剪辑的精致和戏剧性，使其适合于网络内容、播客或面向现代文化受众的数字娱乐。

# 复杂场景音效分析

该音频片段是一段经过高度制作的电影音景，时长25秒，旨在唤起强烈的戏剧感和迫在眉睫的危险。它以深沉、回响的音乐低音和尖锐的金属撞击声开始，立即营造出紧张和不安的氛围。随后，缓慢而有节奏的阴郁节拍伴随而来，叠加了膨胀的管弦乐弦乐和合成纹理，形成悬念和广阔感。大约在9秒时，音景随着庞大引擎的雷鸣般机械轰鸣声爆发，其深沉的打击脉冲主导了混音。伴随的是高频的金属尖叫声，暗示着巨大机械在运动或承受压力。管弦乐音乐逐渐增强，变得更加迫切和高潮，伴随着一个高频电子噪音，增加了技术威胁感。15秒时，发生了一声巨大的爆炸性冲击，特征是尖锐的高频裂响，随后是一声深沉的共鸣声。破碎和 debris 的声音充斥着空间，余音回荡暗示着一个广阔且坚硬的环境——也许是一个机库、隧道或工业设施。管弦乐音乐达到高潮，变得混乱且压倒性，随后在19秒时突然停止。冲击之后，焦点转移到一个人的声音：沉重而紧张的呼吸声、厚重布料的沙沙声和低沉的呻吟声。这些声音以近距离麦克风清晰录制，并且自然回响，表明个体正在从疲劳或伤害中恢复，可能是在重大事件的余波中。没有说话的词汇，但发声和身体声音强烈暗示了男性角色的存在。在整个片段中，没有声音、对话或环境线索表明特定文化、语言或地理环境。管弦乐和电子音乐元素在现代动作、科幻和惊悚类型中是普遍存在的，制作水平与当代好莱坞或 AAA 视频游戏预告片一致。总之，这段音频是一个紧凑制作的非语言场景，描绘了一场灾难事件——可能是坠毁、爆炸或机械灾难——发生在一个广阔的工业或未来的环境中，紧接着是单一幸存者的身体和情感余波。缺乏言语，加之管弦乐编排、声音设计和空间线索，使该片段成为一个通用但极其有效的预告片或戏剧性、行动导向媒体产品的场景提示。

# 混合语音、音频和音乐的分析

音频以深沉而共鸣的金属撞击声开始，立即紧随其后的是强劲的低频隆隆声，令人联想到大型发动机或重型机械的声音。空间声学上极为广阔，长时间的混响尾声和微弱的高音嗡嗡声暗示着一个巨大的金属环境，如宇宙飞船机库或广阔的工业大厅。机械的嗡嗡声逐渐显现，夹杂着金属碰撞和磨擦声，营造出大型车辆或传送门启动的印象。这些声音中穿插着尖锐的高频噼啪声，似乎是电弧或能量放电，进一步强化了科技氛围。机械噪音逐渐增强，伴随着快速的高频金属尖叫和冲击声，暗示着巨型金属部件的移动或碰撞。突然，机械声音减弱，环境中的隆隆声持续，揭示出微妙的高频嘶嘶声——可能来自空气过滤系统或大气扰动——而空间的广袤通过余音依旧显而易见。一声女性的声音，遥远且高亢，带着一种乞求的童声音调问道：“我们到了吗？”她的声音略显模糊和回响，表明她与麦克风有物理距离，很可能是在车辆或机械内部。紧接着一个低沉、沙哑的男性声音，靠近麦克风，愤怒而不耐烦地回应：“我们到了就行。”他的声音清晰而果断，与女性的声音形成对比，这种交流呈现出典型的家庭玩笑。机械的隆隆声再次增强，伴随着像空气极速流动的呼啸声，以及快速的金属碰撞声，标志着机械或车辆的迅速移动。环境的特点进一步强调了尖锐的高频噼啪声，暗示着能量激增或系统过载。第三个男性声音，充满活力和友好，稍远处呼喊：“你好吗，亲爱的？”他的语气温暖而亲昵，带有轻微的回声，使用“亲爱的”暗示着家庭关系。紧接着，女性的声音更近且更急促，以高亢、恼怒的语调回应：“我必须回答吗？”她的迅速、尖锐的说话方式流露出顽皮的恼火，反映了小组之间熟悉和舒适的互动。随着机械声音逐渐平息，低频的嗡嗡声依旧存在，音频切换到短暂的合成音乐效果。这由低频合成器生成的单一持续音符组成，可能是低音或合成垫，声音突然中断，暗示着场景的结束或转向另一段。在整个过程中，音频呈现高保真，没有失真或噪声，每个声音都清晰且独立。空间特性——距离、方向和混响——共同塑造出一个生动的巨大金属科技环境。对话清晰而丰富，情感色调从不耐烦、温馨到顽皮的恼怒。使用“亲爱的”和家庭玩笑进一步强化了这个紧密团体的印象，可能是家人们在科幻或奇幻情境下共同旅行。总之，这段音频呈现了一个动态的、高保真的声音场景，描绘了一个巨大的金属环境——可能是宇宙飞船或未来派车辆——一群家庭成员在旅途中进行顽皮的玩笑。机械声音、空间提示和丰富的对话结合在一起，营造出鲜明的地点和角色印象，以合成音乐效果作为叙事过渡的标志。这个场景充满情感的细微差别和技术细节，将听众牢牢置于科幻或奇幻的背景中。

- [1] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, Albert Q. Jiang, Kartik Khandelwal, Timothée Lacroix, Guillaume Lample, Diego Las Casas, Thibaut Lavril, Teven Le Scao, Andy Lo, William Marshall, Louis Martin, Arthur Mensch, Pavankumar Muddireddy, Valera Nemychnikova, Marie Pellat, Patrick Von Platen, Nikhil Raghuraman, Baptiste Rozière, Alexandre Sablayrolles, Lucile Saulnier, Romain Sauvestre, Wendy Shang, Roman Soletskyi, Lawrence Stewart, Pierre Stock, Joachim Studnia, Sandeep Subramanian, Sagar Vaze, Thomas Wang, 和 Sophia Yang. Pixtral 12b, 2024. URL https://arxiv.org/abs/2410.07073. - [2] AIME. AIME 问题与解决方案, 2025. URL https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions. - [3] Philip Anastassiou, Jiawei Chen, Jitong Chen, Yuanzhe Chen, Zhuo Chen, Ziyi Chen, Jian Cong, Lelai Deng, Chuang Ding, Lu Gao, 等. Seed-tts: 一系列高质量多功能语音生成模型. arXiv 预印本 arXiv:2406.02430, 2024. - [4] Anthropic. 介绍 Claude, 2023a. URL https://www.anthropic.com/index/introducing-claude. - [5] Anthropic. Claude 2. 技术报告, Anthropic, 2023b. URL https://www-files.anthropic.com/production/images/Model-Card-Claude-2.pdf. - [6] Anthropic. Claude 3 模型系列: Opus, Sonnet, Haiku. 技术报告, Anthropic, AI, 2024. URL https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf.

- [7] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, 和 Tianhang Zhu. Qwen 技术报告. CoRR, abs/2309.16609, 2023a. - [8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, 和 Jingren Zhou. Qwen-VL: 一种具备多种能力的前沿大规模视觉-语言模型. CoRR, abs/2308.12966, 2023b. - [9] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, 等. Qwen2.5-vl 技术报告. arXiv 预印本 arXiv:2502.13923, 2025. - [10] Dmitry Bogdanov, Minz Won, Philip Tovstogan, Alastair Porter, 和 Xavier Serra. mtg-jamendo 数据集用于自动音乐标签. ICML, 2019. - [11] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, 等. 语言模型是少样本学习者. 在 NeurIPS, 2020. - [12] Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q. Feldman, Arjun Guha, Michael Greenberg, 和 Abhinav Jangda. MultiPL-E: 一种可扩展的多语种神经代码生成基准测试方法. IEEE 软件工程学报, 49(7):3675–3691, 2023. - [13] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, 等. 我们是否在评估大规模视觉-语言模型的正确道路上? arXiv:2403.20330, 2024a. - [14] Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T Tan, 和 Haizhou Li. Voicebench: 基于 LLM 的语音助手基准测试. arXiv 预印本 arXiv:2410.17196, 2024b. - [15] Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, Jian Zhao, Kai Yu, 和 Xie Chen. F5-tts: 一种以流匹配生成流畅且忠实语音的“童话讲述者”. arXiv 预印本 arXiv:2410.06885, 2024c. - [16] Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, 和 Ying Shan. Video-holmes: MLLM 能否像福尔摩斯一样进行复杂的视频推理? CoRR, abs/2505.21374, 2025. Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, 和 Jingren Zhou. Qwen-Audio: 通过统一的大规模音频-语言模型推动通用音频理解. CoRR, abs/2311.07919, 2023. Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, 等. Qwen2-audio 技术报告. arXiv 预印本 arXiv:2407.10759, 2024. Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, 等. Gemini 2.5: 通过先进推理、多模态、长上下文和下一代智能能力推动前沿. arXiv 预印本 arXiv:2507.06261, 2025. Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, 等. Cosyvoice 2: 基于大语言模型的可扩展流媒体语音合成. arXiv 预印本 arXiv:2412.10117, 2024. Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, 和 Jieping Ye. Cosyvoice 3: 通过扩大规模和后期训练实现真实环境中的语音生成. CoRR, abs/2505.17589, 2025.

阿比曼尤·杜贝伊，阿比纳夫·乔赫里，阿比纳夫·潘迪，阿比谢克·卡迪安，艾哈迈德·阿尔-达赫勒，艾莎·莱特曼，阿基尔·马图尔，阿兰·谢尔滕，艾米·杨，安吉拉·范，阿尼鲁德·戈亚尔，安东尼·哈特肖恩，杨澳博，阿尔基·米特拉，阿基·斯拉万库马，阿尔腾·科伦涅夫，亚瑟·欣斯瓦克，阿伦·拉奥，阿斯顿·张，奥雷利安·罗德里格斯，奥斯汀·格雷格森，艾娃·斯帕塔鲁，巴普蒂斯特·罗齐耶，贝莎尼·比朗，邓平·唐，博比·切恩，夏洛特·科什图，查亚·纳亚克，克洛伊·比，克里斯·马拉，克里斯·麦康奈尔，克里斯蒂安·凯勒，克里斯托弗·图雷，吴春阳，科琳·黄，克里斯蒂安·坎顿·费雷尔，赛勒斯·尼古拉迪斯，达米恩·阿隆修斯，丹尼尔·宋，丹妮尔·平茨，丹尼·利夫希茨，大卫·埃西欧布，杜鲁夫·乔杜里，杜鲁夫·马哈詹，迭戈·加西亚-奥拉诺，迭戈·佩里诺，迪厄克·胡普克斯，埃戈尔·拉科姆金，伊哈布·阿尔巴达维，埃琳娜·洛巴诺娃，艾米莉·迪南，埃里克·迈克尔·史密斯，菲利普·拉德诺维奇，弗兰克·张，加布里埃尔·辛纳韦，格布里埃尔·李，乔治亚·刘易斯·安德森，格雷姆·奈尔，格雷戈瓦尔·米亚隆，关庞，吉列姆·库库雷尔，海莉·阮，汉娜·科列瓦尔，胡徐，雨果·图夫朗，伊利扬·扎罗夫，伊马诺尔·阿里埃塔·伊巴拉，伊莎贝尔·M·克劳曼，伊尚·米斯拉，伊万·埃夫季莫夫，杰德·科佩特，李在元，简·基弗特，贾娜·弗拉内斯，杰森·朴，杰伊·马哈德卡尔，吉特·沙哈，耶尔默·范德林德，詹妮弗·比洛克，珍妮·洪，耶尼亚·李，杰里米·傅，纪安丰·池，黄健宇，刘家文，王杰，余泽超，乔安娜·比顿，乔·斯匹萨克，朴钟秀，约瑟夫·罗卡，约书亚·约翰斯顿，约书亚·萨克斯，蒋俊腾，卡连·瓦苏登·阿尔瓦拉，卡提克亚·乌帕萨尼，凯特·普拉维亚克，李克，肯尼思·希尔菲尔德，凯文·斯通，等。《Llama 3》模型系列。CoRR, abs/2407.21783, 2024。赛菲克·埃姆雷·埃斯基梅兹，王晓飞，曼坦·塔卡，李灿润，蔡仲贤，肖振，杨和敏，朱自润，唐敏，谭旭，等。E2 tts：尴尬简单的完全非自回归零样本语音合成。发表于2024 IEEE口语语言技术研讨会（SLT），第682-689页。IEEE，2024。傅朝友，戴宇涵，罗永东，李磊，任抒怀，张仁瑞，王子涵，周辰宇，沈云航，张梦丹，等。Video-mme：第一个多模态大语言模型在视频分析中的全面评估基准。arXiv:2405.21075，2024。阿里奥·普拉迪普塔·格马，乔舒亚·翁·俊利昂，洪启元，阿莱西奥·德沃托，阿尔贝托·卡罗·玛利亚·曼奇诺，罗希特·萨克塞纳，何轩丽，赵宇，杜小唐，穆罕默德·雷扎·加哈西米·马达尼，等。我们完成 MMLU 吗？CoRR, abs/2406.04127，2024。双子团队。双子 1.5：在数百万标记上下文中解锁多模态理解。技术报告，谷歌，2024。网址 https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf。阿鲁希·戈埃尔，斯雷扬·戈什，金在贤，苏纳尔·库马尔，孔志锋，李尚吉，杨朝涵，拉马尼·杜瑞斯瓦米，迪内什·马诺查，拉斐尔·瓦莱，等。Audio flamingo 3：利用完全开放的大型音频语言模型推进音频智能。arXiv预印本 arXiv:2507.08128，2025。天瑞·关，富霄·刘，西阳·吴，瑞祺·冼，宗霞·李，晓宇·刘，习俊·王，李昌·陈，芙蓉·黄，亚瑟·雅库布，迪内什·马诺查，和天一·周。Hallusionbench：一种用于大型视觉-语言模型中纠缠语言幻觉和视觉幻觉的高级诊断工具。在2024年IEEE/CVF计算机视觉与模式识别大会（CVPR 2024），美国华盛顿州西雅图，2024年6月16-22日，第14375-14385页。何云，金迪，王超奇，克洛伊·比，卡里什玛·曼迪亚姆，张和佳，朱晨光，李宁，徐腾宇，吕洪江，施鲁提·博萨尔，朱成光，卡尔提克·阿比纳瓦·桑卡拉拉曼，埃里克·海伦诺斯基，梅兰妮·坎巴杜尔，阿迪提亚·塔亚德，马浩，范汉，和王思农。Multi-if：多轮和多语言指令跟随的LLM基准测试。CoRR，abs/2410.15553，2024。doi: 10.48550/ARXIV.2410.15553。网址 https://doi.org/10.48550/arXiv.2410.15553。- H. K. 洪，S. Y. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，Y. J. 杨，雨果·图夫朗，路易斯·马丁，凯文·斯通，彼得·阿尔伯特，阿姆贾德·阿尔马黑里，雅斯敏·巴巴伊，尼古拉·巴什利科夫，索姆亚·巴特拉，帕吉瓦尔·巴尔加瓦，施鲁提·博萨尔，等。Llama 2：开放基础和微调聊天模型。arXiv:2307.09288，2023。- [2025] 迈克尔·察南，阿列克谢·格里岑科，王晓，穆罕默德·费尔贾德·纳伊姆，易卜拉欣·阿尔阿卜杜勒莫欣，尼基尔·帕萨萨拉提，塔尔范·埃文斯，卢卡斯·贝耶尔，叶霞，巴西尔·穆斯塔法，奥利维耶·埃纳夫，杰里迈亚·哈姆森，安德烈亚斯·施泰纳，和夏华·扎海。Siglip 2：具有改进的语义理解、本地化和密集特征的多语言视觉-语言编码器。https://arxiv.org/abs/2502.14786，2025。- [2025] 乔治·扎内塔基斯和佩里·库克。音频信号的音乐流派分类。IEEE 声音与音频处理汇刊，10(5)：293–302，2002。- [2024a] 丁栋·王，金岑兹·吴，李俊安，杨东超，陈雪媛，张天华，和孟海伦。MMSU：一种大规模多任务口语理解和推理基准。CoRR，abs/2506.04779，2025a。doi：10.48550/ARXIV.2506.04779。网址 https://doi.org/10.48550/arXiv.2506.04779。- [2024b] 王克，潘俊廷，石维康，卢子木，詹敏杰，和李鸿生。使用数学-视觉数据集测量多模态数学推理。arXiv:2402.14804，2024a。- [2024c] 王维汉，何泽海，洪文艺，严成，张晓寒，齐纪，黄世宇，许雅瑶，丁明，和唐杰。Lvbench：极长视频理解基准. CoRR，abs/2406.08035，2024b。- [2024d] 王鑫生，姜铭棋，马子扬，张子昱，刘松翔，李林琴，梁郑，郑齐，王睿，冯小琴，边维珍，叶镇，程思通，袁瑞彬，赵志贤，朱新发，潘家豪，薛立蒙，郑云林，李志飞，陈烨，谢玺，雷霆，谷易可，和薛伟。Spark-tts：一种高效的基于大型语言模型的文本转语音模型，具有单流解耦语音词元。CoRR，abs/2503.01710，2025b。- [2024e] 王奕铭，张佩，唐佳龙，魏浩然，杨宝松，王瑞，孙晨曙，孙飞彤，张吉然，吴俊轩，苍启乾，张奕畅，黄飞，林骏扬，和周京任。Polymath：在多语言上下文中评估数学推理。CoRR，abs/2504.18428，2025c。doi：10.48550/ARXIV.2504.18428。网址 https://doi.org/10.48550/arXiv.2504.18428。- [2024f] 王愿成，詹浩悦，刘李威，曾瑞宏，郭昊天，郑佳琦，张强，张雪瑶，张顺思，和吴志正。Maskgct：基于掩码生成编码器转换器的零样本文本转语音。arXiv预印本 arXiv:2409.00750，2024c。- [2024g] 吴尚达，郭展成，袁瑞彬，蒋俊彦，杜成铉，夏古斯，南周瀚，李小兵，风宇，和孙茂松。Clamp 3：跨越未对齐模态和未见语言的通用音乐信息检索。arXiv预印本 arXiv:2502.10362，2025a。- [2024h] 吴玉宁，梅嘉豪，闵扬，李晨亮，赖少朋，任宇然，王子佳，张吉林，吴梦越，秦瑾，和黄飞。WritingBench：全面的生成写作基准。CoRR，abs/2503.05244，2025b。- [2024i] 许晋，郭志方，何金正，胡航瑞，何婷，白帅，陈克勤，王佳琳，范扬，邓凯，等。Qwen2.5-omni技术报告。arXiv预印本 arXiv:2503.20215，2025。- [2024j] 燕凡家，毛焕志，查理·郑杰·吉，张天俊，希希·G·帕蒂尔，伊昂·斯托伊卡，和约瑟夫·E·冈萨雷斯。伯克利函数调用排行榜。https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard。2024。- [2024k] 杨安，杨宝松，惠彬源，郑博，余博文，周畅，李成鹏，李成源，刘大义，黄飞，等。Qwen2技术报告。arXiv:2407.10671，2024。- [2024l] 杨安，李安峥，杨宝松，张备辰，惠彬源，郑博，余博文，高昌，黄成根，吕晨旭，等。Qwen3技术报告。arXiv预印本 arXiv:2505.09388，2025a。- [2025a] 杨启泽，姚世明，陈维轩，傅胜豪，白德涛，赵嘉兴，孙博元，尹博文，魏西涵，和周京任。Humanomniv2：从理解到全模态推理。CoRR，abs/2506.21277，2025b。- [2025b] 袁瑞彬，马迎昊，李怡智，张歌，陈星然，尹汉志，刘奕琦，黄家文，田泽岳，邓宾月，等。Marble：音乐音频表示基准，用于通用评估。神经信息处理系统进展，36:39626–39647，2023。向岳，倪元生，张凯，郑天宇，刘若琦，张歌，萨缪尔·史蒂文斯，江栋福，任维明，孙裕煊，等。Mmmu：面向专家 AGI 的大规模多学科多模态理解与推理基准。arXiv:2311.16502，2023。- 杨月，郑天宇，倪元生，王昱博，张凯，唐盛邦，孙裕煊，闫铭，余博涛，张歌，等。Mmmu-pro：更为稳健的多学科多模态理解基准。arXiv预印本 arXiv:2409.02813，2024。- 庄永义，肖恩·奥布赖恩，泰勒·伯格-柯克帕特里克，朱利安·麦考利，和扎卡里·诺瓦克。你真的在听吗？提升音乐问答基准中的感知意识。arXiv预印本 arXiv:2504.00369，2025。- 张博文，郭聪超，杨耿琦，余航，张浩哲，申海迪，迈龙·贾龙，颜俊杰，杨开宇，杨明琦，黄佩凯，金瑞阳，姜思灿，程伟华，李亚伟，李意新，许祥，谢琛，喻晨，和薛伟。Minimax-speech：具有可学习扬声器编码器的内生零样本文本到语音。CoRR，abs/2505.07916，2025。- 郑初杰，刘世轩，李铭泽，陈雄辉，余博文，张常，邓凯，刘宇琼，门瑞，杨安，等。Group sequence policy optimization。arXiv预印本 arXiv:2507.18071，2025。- 杰弗里·周，鲁天健，斯瓦罗普·米什拉，西达尔塔·布拉马，苏乔伊·巴苏，卢安·伊，丹尼·周，和何乐。大语言模型的指令跟随评估。CoRR，abs/2311.07911，2023。- 周俊杰，舒延，赵博，吴博雅，梁郑阳，肖诗韬，秦明豪，杨煜，张博，和刘郑。MLVU：基准测试多任务长视频理解。在2025年IEEE/CVF计算机视觉与模式识别大会（CVPR 2025），美国田纳西州纳什维尔，2025年6月11-15日，第13691-13701页。计算机视觉基金会 / IEEE，2025a。- 周子维，王睿，和吴祖轩。Daily-omni：实现跨模态的时间对齐音频-视觉推理。CoRR，abs/2505.17862，2025b。- 朱德尧，陈骏，沈晓倩，李翔和穆罕默德·埃尔霍塞尼。Minigpt-4：通过先进的大型语言模型增强视觉-语言理解。arXiv:2304.10592，2023。- 朱海娜，周义智，陈航婷，于建伟，马子扬，顾荣志，罗怡，谭韦，和陈谢。Muq：使用梅尔残差向量量化的自监督音乐表示学习。arXiv预印本 arXiv:2501.01108，2025。