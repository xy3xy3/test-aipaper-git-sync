# 1. 论文基本信息

## 1.1. 标题
Qwen3-Omni Technical Report

## 1.2. 作者
核心贡献者包括：Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, Junyang Lin 等。

## 1.3. 发表期刊/会议
本报告为技术报告 (Technical Report)，未明确提及已发表在特定期刊或会议，通常此类报告在预印本平台（如 arXiv）发布后可能会在后续提交给相关会议或期刊。

## 1.4. 发表年份
2025年

## 1.5. 摘要
本文介绍了 Qwen3-Omni，这是一个单一的多模态模型，首次在文本、图像、音频和视频等多种模态上保持了最先进 (state-of-the-art, SOTA) 的性能，且相对于单模态对应模型没有任何性能下降。Qwen3-Omni 在 Qwen 系列中与同尺寸的单模态模型性能持平，尤其在音频任务上表现出色。在36个音频和视听基准测试中，Qwen3-Omni 在32个基准上实现了开源 SOTA，并在22个基准上达到了整体 SOTA，超越了强大的闭源模型，如 Gemini-2.5-Pro, Seed-ASR 和 GPT-4o-Transcribe。

Qwen3-Omni 采用了一种 Thinker-Talker 混合专家 (Mixture-of-Experts, MoE) 架构，统一了文本、图像、音频和视频的感知与生成，从而产生流畅的文本和自然的实时语音。它支持119种语言的文本交互，19种语言的语音理解和10种语言的语音生成。为了减少流式合成中的首包延迟 (first-packet latency)，Talker 模块采用多码本 (multi-codebook) 方案自回归地预测离散语音编码器 (discrete speech codecs)。利用这些码本的表示能力，模型用一个轻量级的因果卷积网络 (causal ConvNet) 取代了计算密集型分块扩散 (block-wise diffusion)，实现了从第一个编码帧开始的流式传输。在冷启动 (cold-start) 设置下，Qwen3-Omni 的理论端到端首包延迟为234毫秒。为了进一步加强多模态推理能力，模型引入了一个 Thinking 模型，显式地对来自任何模态的输入进行推理。

由于研究社区目前缺乏通用的音频字幕生成模型，团队通过对 Qwen3-Omni-30B-A3B 进行微调，得到了 Qwen3-Omni-30B-A3B-Captioner，该模型能够为任意音频输入生成详细且幻觉 (hallucination) 较少的字幕。Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking 和 Qwen3-Omni-30B-A3B-Captioner 均以 Apache 2.0 许可证公开发布。

## 1.6. 原文链接
原文链接: https://arxiv.org/abs/2509.17765v1
PDF 链接: https://arxiv.org/pdf/2509.17765v1.pdf
发布于 (UTC): 2025-09-22T13:26:24.000Z

# 2. 整体概括

## 2.1. 研究背景与动机
**论文试图解决的核心问题是什么？**
当前的语言模型 (large language models, LLMs) 虽然在单模态理解和推理方面取得了快速进展，但多模态模型往往面临一个关键限制：模态间的性能权衡 (modality trade-offs)，即在一部分模态上取得性能提升的同时，可能导致其他模态的性能下降。这使得构建一个能在所有模态上同时保持最先进性能且无损耗的统一多模态模型成为一项挑战。

**为什么这个问题在当前领域是重要的？现有研究存在哪些具体的挑战或空白？**
人类对世界的感知是多模态的，通过并行处理视觉和听觉输入，进行认知处理，并通过文本、语音等方式进行响应。因此，开发能够模仿这种多模态感知和交互能力的系统，是实现通用人工智能 (Artificial General Intelligence, AGI) 的关键一步。现有挑战包括：
1.  **模态间性能不一致：** 许多多模态模型难以在所有模态上同时达到单模态专业模型的性能水平。
2.  **跨模态推理能力不足：** 传统模型在整合不同模态信息进行复杂推理方面存在局限。
3.  **实时交互延迟：** 语音生成等任务对延迟要求极高，现有方法在保证质量的同时难以实现超低延迟。
4.  **缺乏通用音频字幕模型：** 相比视觉模态，音频模态的通用字幕生成模型仍是空白。

**这篇论文的切入点或创新思路是什么？**
本文的创新思路在于：
1.  **端到端集成多模态训练：** 探索在统一的 LLM 范式下进行集成多模态训练，旨在消除模态间的性能退化。
2.  **Thinker-Talker MoE 架构：** 采用升级版的 Thinker-Talker 混合专家 (MoE) 架构，将感知（Thinker）和生成（Talker）分离并优化，以支持高并发和快速推理，特别是对语音生成进行深度优化。
3.  **专门的音频编码器和语音生成机制：** 引入从头训练的 AuT 音频编码器，并为 Talker 模块设计多码本自回归预测和轻量级因果卷积网络 (causal ConvNet) 的语音合成流程，以实现超低延迟。
4.  **显式多模态推理模型：** 引入 `Thinking` 模型以加强跨模态推理。
5.  **填补音频字幕空白：** 通过微调，推出了通用的音频字幕模型。

## 2.2. 核心贡献/主要发现
**论文最主要的贡献是什么？**
1.  **首个无性能下降的统一多模态模型：** Qwen3-Omni 首次证明了单一多模态模型可以在文本、图像、音频和视频所有模态上，与同尺寸的单模态专业模型性能持平甚至超越，而没有任何模态特异性性能下降。
2.  **卓越的音频性能：** 在36个音频和视听基准测试中，Qwen3-Omni 在32个上实现了开源 SOTA，并在22个上达到整体 SOTA，超越了 Gemini-2.5-Pro 和 GPT-4o-Transcribe 等强大闭源模型。
3.  **升级的 Thinker-Talker MoE 架构：** Thinker 和 Talker 模块均升级为 MoE 架构，提高了可扩展性和并发推理能力。Talker 进一步解耦，仅依赖多模态特征，实现了 Thinker 和 Talker 的独立系统提示控制。
4.  **创新的低延迟语音合成：** 引入多码本自回归预测和轻量级因果 ConvNet (Code2Wav) 的 Talker 模块，实现了理论端到端234毫秒的超低首包延迟。
5.  **强大的 AuT 音频编码器：** 引入从头训练的 AuT（Audio Transformer）编码器，在2000万小时监督音频数据上训练，生成更强的通用音频表示，并支持长达40分钟的音频处理。
6.  **显式多模态推理能力：** 引入了一个 `Thinking` 模型，能够对来自任何模态的输入进行显式推理，进一步增强了多模态理解。
7.  **通用音频字幕生成模型：** 首次提出了一个通用的音频字幕生成模型 Qwen3-Omni-30B-A3B-Captioner，填补了该领域研究空白。
8.  **广泛的语言支持：** 支持119种文本语言交互，19种语音理解语言和10种语音生成语言。

**论文得出了哪些关键的结论或发现？这些发现解决了什么具体问题？**
1.  **多模态训练不再需要性能权衡：** 证明了通过精心设计的集成多模态训练，可以实现各模态性能的协同提升而非相互妥协。解决了多模态模型在不同模态间性能不平衡的问题。
2.  **实时、自然的多模态交互成为可能：** Qwen3-Omni 的低延迟语音合成能力和多语言支持，使得其能够提供高度流畅和自然的实时语音交互体验。解决了多模态对话系统在延迟和语音自然度方面的瓶颈。
3.  **跨模态能力显著增强：** 通过 Thinker-Talker 架构、TM-RoPE 位置编码以及 Thinking 模型，模型能有效整合不同模态信息进行复杂推理，尤其在视听理解等任务上表现突出。解决了现有模型在深层跨模态推理方面的不足。
4.  **填补了音频领域的特定空白：** Qwen3-Omni-30B-A3B-Captioner 的发布，为音频研究社区提供了一个重要的基础工具。解决了通用音频字幕模型的缺失问题。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解 Qwen3-Omni，需要了解以下核心概念：

*   <strong>多模态模型 (Multimodal Models)</strong>：能够处理并理解来自多种信息源（如文本、图像、音频、视频）的数据的模型。与单模态模型（只处理一种数据）不同，多模态模型旨在融合不同模态的信息，以获得更全面的理解和更强大的功能。

*   <strong>状态空间模型 (State-of-the-Art, SOTA)</strong>：指在某个特定任务或数据集上，目前表现最好的模型或方法。Qwen3-Omni 在多个基准测试上实现了 SOTA，意味着它在这些任务上达到了当前的最高水平。

*   <strong>混合专家 (Mixture-of-Experts, MoE) 架构</strong>：一种深度学习架构，其中模型由多个“专家网络”组成，并通过一个“门控网络”来决定哪些专家处理特定的输入。这种架构可以显著增加模型的参数量（提高模型容量），同时在推理时只激活一部分专家，从而降低计算成本，提高效率和吞吐量，特别适用于处理大规模和多样化的数据。

*   <strong>编码器-解码器 (Encoder-Decoder) 架构</strong>：一种常见的神经网络结构，编码器将输入数据（如音频、图像）转换成一种紧凑的中间表示（也称作特征向量或嵌入），解码器则根据这个中间表示生成输出（如文本、语音）。

*   <strong>自回归 (Autoregressive)</strong> 模型：指模型在生成序列时，当前步的输出依赖于前一步的输出。在语音合成中，这意味着模型会逐帧或逐码本生成语音，每生成一部分都会作为下一部分生成的输入。

*   <strong>离散语音编码器 (Discrete Speech Codecs)</strong>：将连续的语音信号量化为离散的数字序列（码本），从而实现语音的压缩和高效传输。在语音合成中，模型直接预测这些离散码本，而不是连续的声学特征，可以简化生成过程并提高实时性。

*   <strong>码本 (Codebook)</strong>：离散语音编码中的一组预定义的向量（或编码），用于表示语音的不同片段。多码本方案则使用多组码本，以更精细地捕捉语音的细节。

*   <strong>因果卷积网络 (Causal ConvNet)</strong>：一种卷积神经网络，其输出仅依赖于当前及过去的输入，而不依赖未来的输入。这对于流式数据处理（如实时语音合成）至关重要，因为它允许模型在接收到输入的同时进行生成，而无需等待整个输入序列完成。

*   <strong>首包延迟 (First-Packet Latency)</strong>：在流式数据传输或实时交互系统中，从系统接收到请求到生成并发送第一个响应数据包所需的时间。对于语音对话系统，低首包延迟对于提升用户体验至关重要。

*   <strong>冷启动 (Cold-Start)</strong>：在没有历史上下文或预热数据的情况下启动系统。在语音交互中，通常指首次进行对话时的延迟。

*   <strong>幻觉 (Hallucination)</strong>：生成式模型（尤其是大语言模型）在生成内容时，出现看似合理但实际上是虚假、不准确或与输入不符的信息的现象。

*   <strong>时间对齐多模态旋转位置嵌入 (Time-aligned Multimodal Rotary Position Embedding, TM-RoPE)</strong>：一种在多模态模型中使用的位置编码技术。它基于旋转位置嵌入 (Rotary Position Embedding, RoPE) 扩展而来，不仅考虑了文本的序列位置，还引入了时间、高度、宽度的维度，并能将不同模态（如视频帧和音频）的时序信息对齐，从而更好地处理多模态数据的时空关系。

*   <strong>持续预训练 (Continual Pretraining, CPT)</strong>：在模型已完成初始预训练后，继续使用新的、通常更高质量或特定领域的数据进行训练，以进一步提升模型性能、适应新数据分布或缓解初始训练中引入的某些问题（如幻觉）。

*   <strong>直接偏好优化 (Direct Preference Optimization, DPO)</strong>：一种强化学习微调 (Reinforcement Learning from Human Feedback, RLHF) 的替代方法，用于对生成模型进行对齐。它直接优化模型的策略，使其更倾向于生成人类偏好的响应，而无需复杂的奖励模型训练。

## 3.2. 前人工作
论文在引言和背景中提及了大量前人工作，主要分为以下几类：

1.  <strong>单模态大模型 (Unimodal Large Models)</strong>：
    *   <strong>大语言模型 (LLMs)</strong>：如 GPT 系列 (OpenAI, 2023; Anthropic, 2023a, b, 2024)、Gemini 系列 (Gemini Team, 2024)、Llama 系列 (Touvron et al., 2023; Dubey et al., 2024) 和 Qwen 系列 (Bai et al., 2023a, b, 2025; Chu et al., 2023, 2024; Yang et al., 2024, 2025a)。这些模型在理解和生成文本方面取得了显著成功，为多模态模型提供了强大的文本主干网络。
    *   <strong>大型视觉模型 (Large Vision Models)</strong>：如 CLIP (Radford et al., 2021)、SigLIP2 (Tschannen et al., 2025)。这些模型在图像理解和视觉-语言对齐方面奠定了基础。
    *   <strong>大型音频模型 (Large Audio Models)</strong>：如 Whisper (Radford et al., 2023)。这些模型在语音识别和音频理解方面取得了突破。

2.  <strong>多模态模型 (Multimodal Models)</strong>：
    *   **LLM-centric 多模态模型**：如 GPT-4o (OpenAI, 2024)、Gemini 2.5 (Comanici et al., 2025)。这些模型尝试将多种模态集成到以 LLM 为中心的框架中。
    *   **Qwen 系列的多模态模型**：如 Qwen-VL (Bai et al., 2023b)、Qwen2.5-Omni (Xu et al., 2025)。Qwen3-Omni 正是在 Qwen2.5-Omni 的 Thinker-Talker 架构基础上进行升级的。

3.  <strong>语音合成技术 (Speech Synthesis Technologies)</strong>：
    *   <strong>基于扩散模型 (Diffusion-based models)</strong>：如 DiT (Peebles et al., 2023)。这些模型在生成高质量语音方面表现出色，但通常计算密集。
    *   <strong>零样本语音合成 (Zero-shot Speech Generation)</strong>：如 Seed-TTS (Anastassiou et al., 2024)、CosyVoice (Du et al., 2024, 2025)。

4.  <strong>评估基准 (Evaluation Benchmarks)</strong>：
    *   文本：MMLU-Redux (Gema et al., 2024)、GPQA (Rein et al., 2023)、AIME25 (AIME, 2025) 等。
    *   音频：VoiceBench (Chen et al., 2024b)、MMAU (Sakshi et al., 2024)、MMSU (Wang et al., 2025a)、RUL-MuchoMusic (Zang et al., 2025) 等。
    *   视觉：MMStar (Chen et al., 2024a)、MathVista (Lu et al., 2024)、MMMU (Yue et al., 2023) 等。
    *   视听：WorldSense (Hong et al., 2025)、DailyOmni (Zhou et al., 2025b)、VideoHolmes (Cheng et al., 2025) 等。

## 3.3. 技术演进
多模态技术从早期独立的单模态研究（如计算机视觉的图像识别、自然语言处理的文本理解、语音识别的声学模型）发展而来。随着 Transformer (Vaswani et al., 2017) 架构的出现和预训练大模型的兴起，多模态领域开始尝试将不同模态的数据统一到一个框架中。

早期多模态模型通常采用**多编码器-单解码器**架构，即为每个模态使用单独的编码器（如 CNN 用于图像，Transformer 用于文本），然后将提取出的特征拼接或融合后输入到一个共享的解码器（通常是 LLM）进行生成。然而，这种方法容易导致模态间性能权衡，且难以实现真正的端到端联合训练。

Qwen 系列的多模态模型（如 Qwen-VL, Qwen2.5-Omni）代表了这一领域的持续演进。Qwen2.5-Omni 引入了 **Thinker-Talker 架构**，将模型分为一个负责高级推理和文本生成的 `Thinker` 模块，以及一个负责语音生成的 `Talker` 模块。这种分离允许对不同任务进行优化，并为实时交互提供了基础。

Qwen3-Omni 在此基础上实现了进一步的飞跃。它解决了 Qwen2.5-Omni 中的一些限制，例如：
1.  **架构升级**：将 Thinker 和 Talker 都升级为 **MoE 架构**，提高了可扩展性和推理效率。
2.  **音频编码器强化**：用从头训练的 **AuT 编码器**取代了通用编码器 (如 Whisper)，获得了更强的通用音频表示。
3.  **语音合成优化**：Talker 从单码本升级为**多码本**，并通过轻量级 **ConvNet (Code2Wav)** 取代了计算量大的扩散模型，显著降低了首包延迟。
4.  **语言覆盖扩展**：支持更多的文本和语音语言。
5.  **引入 Thinking 模型**：显式地增强了多模态推理能力。
6.  **音频字幕新任务**：扩展了模型能力，能生成高质量的音频字幕。

    总的来说，Qwen3-Omni 代表了从**多编码器-单解码器**到<strong>分离感知-生成（Thinker-Talker），再到全模态性能无损耗、低延迟、强推理</strong>的统一多模态模型的发展趋势。

## 3.4. 差异化分析
Qwen3-Omni 与相关工作中的主要方法相比，核心区别和创新点在于：

1.  <strong>无损性能统一 (Non-Degrading Performance Unification)</strong>：
    *   **差异点**：大多数现有多模态模型，包括一些强大的闭源模型，在追求多模态能力时，往往会出现“模态权衡”，即在一个模态上表现出色，可能导致其他模态性能下降。Qwen3-Omni 首次宣称实现了在文本、图像、音频、视频等所有模态上，与同尺寸的单模态专业模型持平甚至超越，且无任何性能下降。
    *   **创新点**：这得益于其独特的训练策略，即在文本预训练的早期阶段混合单模态和跨模态数据，并通过精心设计的架构和训练流程实现模态间的协同增强。

2.  <strong>升级的 Thinker-Talker MoE 架构 (Upgraded Thinker-Talker MoE Architecture)</strong>：
    *   **差异点**：Qwen2.5-Omni 已经引入了 Thinker-Talker 架构。Qwen3-Omni 将这两个关键组件都升级为 **MoE 架构**。
    *   **创新点**：MoE 架构显著增加了模型的容量，同时通过稀疏激活 (sparse activation) 实现了更高的推理效率和并发能力。此外，Talker 模块不再直接消耗 Thinker 的文本表示，而是仅依赖音频和视觉多模态特征，这种解耦允许 Thinker 和 Talker 使用独立的系统提示，实现更灵活的控制（例如，Thinker 控制回应风格，Talker 控制语音风格），并支持外部模块干预 Thinker 的文本输出。

3.  <strong>超低延迟流式语音合成 (Ultra-Low-Latency Streaming Speech Synthesis)</strong>：
    *   **差异点**：现有的高质量语音合成系统（特别是基于扩散模型的）通常存在较高的延迟，难以满足实时交互需求。Qwen2.5-Omni 仍需等待 Talker 提供足够的块上下文 (block-context) 才能开始合成。
    *   **创新点**：Qwen3-Omni 采用**多码本自回归预测**结合<strong>轻量级因果 ConvNet (Code2Wav)</strong> 的 Talker 模块。它能做到每生成一个码本帧就立即合成音频，无需等待整个块。同时，输入和输出音频码率降至 $12.5 \mathrm{Hz}$，使得理论端到端首包延迟低至234毫秒，显著优于现有系统，实现了真正的即时流式语音生成。

4.  <strong>强大的通用音频编码器 AuT (Powerful General-Purpose Audio Encoder AuT)</strong>：
    *   **差异点**：许多多模态模型使用现成的通用音频编码器（如 Whisper）。
    *   **创新点**：Qwen3-Omni 引入了团队从头训练的 <strong>AuT (Audio Transformer) 编码器</strong>，并在2000万小时监督音频数据上进行训练，能够产生更强大和通用的音频表示。AuT 还采用了分块窗口注意力 (block-wise window attention) 以实现实时预取缓存 (real-time prefetch caching)。

5.  <strong>显式多模态推理的 Thinking 模型 (Explicit Multimodal Reasoning with Thinking Model)</strong>：
    *   **差异点**：许多多模态 LLM 依赖隐式的融合来执行多模态推理。
    *   **创新点**：Qwen3-Omni 引入了一个独立的 **Thinking 模型**，能够显式地对来自任意模态的输入进行推理，这尤其增强了模型在音频-视频和纯音频场景下的复杂推理能力。

6.  <strong>填补音频字幕任务空白 (Addressing the Audio Captioning Gap)</strong>：
    *   **差异点**：现有研究主要集中在视觉字幕生成，缺乏通用的音频字幕模型。
    *   **创新点**：通过对 Qwen3-Omni-30B-A3B 进行微调，开发了 **Qwen3-Omni-30B-A3B-Captioner**，能够为任意音频输入生成详细、低幻觉的字幕。

        这些差异化和创新点共同使得 Qwen3-Omni 在性能、效率、功能和交互体验上超越了现有的大多数多模态模型。

# 4. 方法论

## 4.1. 方法原理
Qwen3-Omni 的核心思想是构建一个统一的、端到端的多模态模型，该模型能够处理和生成文本、图像、音频和视频数据，并且在所有模态上都能保持或超越单模态模型的性能，同时实现低延迟的实时交互。其原理基于以下几个核心组件和设计理念：

1.  **Thinker-Talker 架构**：将复杂的认知和生成过程解耦为两个主要模块：
    *   <strong>Thinker (思考者)</strong>：负责处理多模态输入（文本、音频、图像、视频），进行高级理解、推理和生成文本响应。
    *   <strong>Talker (说话者)</strong>：接收 Thinker 生成的高级表示（通常是文本），并结合多模态特征，实时生成自然的语音。
2.  **MoE 架构的应用**：为了处理大规模数据、提高模型容量并保持推理效率，Thinker 和 Talker 模块都采用了混合专家 (Mixture-of-Experts, MoE) 架构。这使得模型可以在参数量巨大的同时，只激活部分专家进行推理，从而降低计算成本和提高吞吐量。
3.  **模态专用编码器与统一表示**：
    *   为每种模态（音频、视觉）设计或选择高性能的编码器，将原始数据转换成统一的嵌入表示，以便 Thinker 进行跨模态融合和理解。
    *   通过时间对齐多模态旋转位置嵌入 (TM-RoPE) 机制，有效地整合不同模态的时序和空间信息，即使在处理长序列的多模态输入时也能保持一致性。
4.  **低延迟流式语音合成**：Talker 模块的设计专注于实时性和自然度。通过自回归预测多码本离散语音编码器 (discrete speech codecs)，并使用轻量级因果卷积网络 (causal ConvNet) 进行波形生成，实现超低的首包延迟。
5.  **强化多模态推理**：引入一个独立的 `Thinking` 模型，旨在显式地进行更深层次的多模态推理，提升模型在复杂场景下的决策和理解能力。
6.  **分阶段预训练**：采用多阶段预训练策略，逐步对齐编码器、进行大规模通用多模态学习，并针对长上下文能力进行优化，以确保模型在不同模态和任务上的鲁棒性能。

## 4.2. 核心方法详解

Qwen3-Omni 的架构如图 Figure 2 所示，采用 Thinker-Talker 架构。

![img-1.jpeg](images/2.jpeg)
*该图像是Qwen3-Omni的架构示意图，展示了其Thinker-Talker MoE架构。其中，Thinker负责文本生成，Talker通过MTP模块和Streaming Codec Decoder实现多码本序列的流式语音生成，支持多模态输入处理。*

Figure 2: The overview of Qwen3-Omni. Qwen3-Omni adopts the Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receives high-level representations directly from Thinker. To achieve ultra-low-latency streaming, Talker autoregressively predicts a multi-codebook sequence. At each decoding step, an MTP module outputs the residual codebooks for the current frame, after which the Code2Wav renderer incrementally synthesizes the corresponding waveform, enabling frame-by-frame streaming generation.

### 4.2.1. 架构总览 (Overview)
Qwen3-Omni 沿用了 Qwen2.5-Omni 的 Thinker-Talker 架构，但引入了五项关键升级和四项主要改进。

**关键升级：**
1.  <strong>Thinker 和 Talker 升级为 MoE 架构 (Thinker and Talker are upgraded to Mixture-of-Experts (MoE) designs)</strong>：提高了模型的容量和推理效率。
2.  <strong>音频编码器升级为 AuT (Audio Transformer) (We replace Whisper audio encoder with our AuT (Audio Transformer) encoder)</strong>：从头在2000万小时监督音频数据上训练，生成更强的通用音频表示。AuT 采用分块窗口注意力 (block-wise window attention) 以实现实时预取缓存。
3.  <strong>语音生成采用多码本表示 (On the speech generation side, we adopt a multi-codebook representation)</strong>：增加了容量，支持更忠实地建模多样化的声音、副语言线索 (paralinguistic cues) 和声学现象。
4.  <strong>Talker 采用多轨编码器建模 (The Talker shifts from single-track to multi-track codec modeling)</strong>：通过多令牌预测 (Multi-Token Prediction, MTP) 模块自回归地预测多个码本层，同时波形生成阶段 (Code2Wav) 用轻量级卷积网络 (ConvNet) 取代了分块扩散 (block-wise DiT)。
5.  <strong>降低输入输出音频码率和支持单帧合成 (The input and output audio code rates are reduced to $12.5\mathrm{Hz}$, with the output codec enabling single-frame, immediate speech synthesis)</strong>：实现了低延迟语音交互。

**主要改进：**
1.  支持超过40分钟的音频理解。
2.  语言覆盖扩展到119种书面语言，19种语音理解语言和10种语音生成语言。
3.  引入 Thinking 模型，实现全模态推理，包括纯音频和视听场景。
4.  流式性能改进，端到端延迟低至234毫秒。

### 4.2.2. AuT 音频编码器 (Audio Transformer (AuT))
AuT 是一个注意力编码器-解码器 (attention-encoder-decoder) 架构的自回归 (auto-regressive) 模型，从头开始在2000万小时的监督音频数据上训练。

![img-2.jpeg](images/3.jpeg)
*该图像是图表，展示了AuT模型架构的结构示意图。AuT包含编码器和解码器，编码器由32层自注意力层和3层降采样卷积组成，解码器进行8次交叉注意力和自注意力计算，实现从Fbank特征到文本的自回归转换。*

Figure 3: The overview of AuT. AuT is an attention-encoder-decoder based auto-regressive model, which is trained from scratch on 20 million hours of supervised audio. Qwen3-Omni employs the AuT encoder as the audio encoder to obtain general purpose audio representations at a token rate of $12.5\mathrm{Hz}$.

**训练过程：**
*   **输入处理**：音频的滤波器组特征 (filter bank features) 经过 Conv2D 块进行8倍下采样，将词元 (token) 速率降低到 $12.5\mathrm{Hz}$。
*   **目标**：学习更强大、更通用的音频表示。
*   **训练数据**：包括80%的中文和英文伪标注 (pseudo-labeled) 自动语音识别 (Automatic Speech Recognition, ASR) 数据，10%的其他语言 ASR 数据，以及10%的音频理解数据。
*   **注意力机制**：为平衡实时预取缓存 (real-time prefetch caching) 的效率和离线音频任务的性能，AuT 采用带有动态注意力窗口大小的闪存注意力 (flash attention)，覆盖1到8秒的注意力查询模式。
*   **Qwen3-Omni 中的应用**：AuT 编码器作为 Qwen3-Omni 的音频编码器，包含约0.6B参数，生成 $12.5\mathrm{Hz}$ 词元速率的通用音频表示。

### 4.2.3. 感知模块 (Perception)
Thinker 模块将文本、音频、图像和视频（不含音频）转换为一系列表示作为输入。

1.  <strong>文本 (Text)</strong>：
    *   使用 Qwen 的分词器 (tokenizer)，基于字节级字节对编码 (byte-level byte-pair encoding)，词汇量为151,643个常规词元。

2.  <strong>音频与视频中的音频 (Audio and Audio extracted from Video)</strong>：
    *   重采样至 $16\mathrm{kHz}$。
    *   将原始波形转换为128通道的梅尔谱图 (mel-spectrogram)，窗口大小为 $25\mathrm{ms}$，步长为 $10\mathrm{ms}$。
    *   采用上述的 **AuT 编码器** 作为音频编码器，每帧音频表示对应约 $80\mathrm{ms}$ 的原始音频信号。

3.  <strong>图像与视频 (Image and Video)</strong>：
    *   采用 Qwen3-VL 的视觉编码器，该编码器初始化自 SigLIP2-So400m (Tschannen et al., 2025)，包含约5.43亿参数。
    *   视觉编码器在图像和视频混合数据上训练，以确保强大的图像和视频理解能力。
    *   为了在保留视频信息完整性的同时与音频采样率对齐，视频帧以动态帧率进行采样。

#### 4.2.3.1. 视频和多模态位置嵌入 (Video and Multimodal Position Embedding, TM-RoPE)
Qwen3-Omni 借鉴 Qwen2.5-Omni，使用时间对齐多模态旋转位置嵌入 (Time-aligned Multimodal Rotary Position Embedding, TM-RoPE)。TM-RoPE 扩展了多模态旋转位置嵌入 (Multimodal Rotary Position Embedding, M-RoPE) (Bai et al., 2023b)，通过集成绝对时间信息。

**TM-RoPE 的工作原理：**
*   **分解**：TM-RoPE 将传统的旋转位置嵌入 (rotary position embedding) 分解为三个维度：<strong>时间 (temporal)</strong>、<strong>高度 (height)</strong> 和 <strong>宽度 (width)</strong>。
*   **角度分配调整**：与原始 M-RoPE 中时间依赖性由前16个旋转角（对应更高频率）建模不同，TM-RoPE 调整了旋转角的分配。具体来说，时间、高度和宽度维度分别分配24、20和20个旋转角，彼此交错。这种重新分配旨在平衡局部语义和长程依赖的表示，从而增强模型整体性能。
*   **模态特定应用**：
    *   **文本输入**：三个组件（时间、高度、宽度）共享相同的定位符，TM-RoPE 此时功能上等同于一维 RoPE (Su et al., 2024)。
    *   **音频输入**：使用共享的位置 ID，但通过绝对时间编码进一步增强，每个时间 ID 对应 $80\mathrm{ms}$ 的持续时间。
    *   **图像数据**：所有视觉词元分配一个常量时间 ID，其独特的行和列位置决定了高度和宽度 ID。
    *   **多模态视听流**：音频组件以 $80\mathrm{ms}$ 为单位进行时间 ID 编码。视频被视为一个具有单调递增时间 ID 的帧序列，这些 ID 根据其实际时间戳动态调整，以确保每 $80\mathrm{ms}$ 有一个一致的时间分辨率。视频帧的高度和宽度 ID 分配方式与静态图像相同。
*   **连续位置编号**：为避免在处理多个模态时出现位置冲突，位置编号是连续的，每个后续模态都从前一个模态的最大位置 ID 加一开始。
*   **长序列处理**：与 Qwen2.5-Omni 将视听表示分割成固定的2秒块不同，Qwen3-Omni 使用其明确锚定到绝对时间的时间 ID 直接对齐这些表示。这种设计允许模型灵活支持任意持续时间的流式输入。

### 4.2.4. 语音生成 (Speech Generation)
Talker 模块在多轮对话中进行语音合成，其条件信息包括来自 Thinker 的丰富上下文，如历史文本词元、多模态表示以及当前轮次的流式文本。

**Talker 的架构和工作流程：**
1.  **直接操作 RVQ 词元**： Talker 直接操作 RVQ (Residual Vector Quantization) 词元，这与 *Xu et al. (2025)* 的方法不同。
2.  **分层预测方案**：
    *   <strong>主干网络 (Backbone)</strong>： Talker 的主干网络接收当前帧的聚合码本特征。
    *   <strong>线性头 (Linear Head)</strong>：使用一个线性头来预测第零个码本 (zeroth codebook)。
    *   <strong>多令牌预测模块 (Multi-Token Prediction, MTP) (Multi-Token Prediction (MTP) module)</strong>：在预测完第零个码本后，MTP 模块生成所有剩余的码本 (residual codebooks)。这种策略使得模型能够学习完整的声学细节表示，增强语音表现力。
3.  <strong>波形重建 (Waveform Reconstruction)</strong>：
    *   **Code2Wav**：波形重建简化为一个轻量级的因果卷积网络 (causal ConvNet)，命名为 Code2Wav。
    *   **优势**：与更复杂的基于 DiT (Diffusion Transformer) 的声码器 (vocoders) 相比，Code2Wav 显著降低了推理延迟和计算成本 (FLOPs)，同时实现了卓越的音频保真度。

### 4.2.5. 流式传输和并发设计 (Designs for Streaming and Concurrency)
Qwen3-Omni 通过算法和架构优化来增强并发能力并降低首包延迟。

#### 4.2.5.1. 分块预填充和 MoE 架构 (Chunked Prefilling and MoE Architecture)
*   <strong>分块预填充 (Chunked Prefilling)</strong>：沿用了 Qwen2.5-Omni 的机制，其音频和视觉编码器能够沿时间维度输出分块。
    *   **异步预填充**：在实时交互中，Thinker 和 Talker 模块异步地执行预填充：当 Thinker 完成当前块的预填充后，其输出的高级表示会立即被用于异步预填充 Talker 的当前块，而 Thinker 同时开始预填充其下一个块。这显著减少了 Thinker 和 Talker 的首词元时间 (Time-To-First-Token, TTFT)。
*   <strong>MoE 架构 (MoE Architecture)</strong>：Qwen3-Omni 中的 Thinker 和 Talker 均采用 MoE 设计。
    *   **提高服务吞吐量**：MoE 架构在处理长序列时显著减少了键值缓存 (KV cache) 的 I/O 消耗，从而增加了生成期间的每秒词元数 (tokens per second, TPS) 并提高了并发性。

#### 4.2.5.2. 流式多码本编码器生成 (Streaming Multi-Codebook Codec Generation)
*   **低延迟机制**：为了最小化用户接收第一个生成数据包的等待时间，提出了一种仅限左上下文 (left context only) 的多码本生成机制。
*   **实时合成**：如 Figure 2 所示，一旦 Talker 生成了第一个词元，MTP 模块就会预测当前帧的其余词元。这些词元随后由一个流式多码本编码器解码器解码成波形，该解码器只关注左侧上下文。与 Qwen2.5-Omni 需要等待 Talker 提供足够的块上下文才能合成不同，Qwen3-Omni 可以在 Talker 生成每个词元后立即输出波形，显著降低了首包延迟。

#### 4.2.5.3. 轻量级 MTP 模块和 ConvNet (Lightweight MTP module and ConvNet)
*   **MTP 模块**：一个超轻量级的固定步长自回归密集 Transformer。它对推理硬件的内存带宽要求低，天然适用于高吞吐量请求的批处理 (batched inference)。其固定步长的自回归推理机制使其能有效利用固定的 KV 缓存内存空间进行加速，从而实现低推理延迟。
*   <strong>Codec Decoder (编码器解码器)</strong>：作为 ConvNet-based 的解码器，也以低延迟实现了高吞吐量。其卷积架构在各种推理平台都享有广泛的硬件加速支持，并能实现高效的批处理推理。

    Table 1: The architectural design of Qwen3-Omni-30B-A3B and the end-to-end first-packet latency for Audio/Video (ms).

    | Module          | Architecture       | Params   | Streaming |
    |-----------------|--------------------|----------|-----------|
    | Audio Encoder   | AuT                | 650M     | ✓         |
    | Vision Encoder  | SigLIP2-So400M     | 540M     | -         |
    | Thinker         | MoE Transformer    | 30B-A3B  | ✓         |
    | Talker          | MoE Transformer    | 3B-A0.3B | ✓         |
    | MTP             | Dense Transformer  | 80M      | ✓         |
    | Code2wav        | ConvNet            | 200M     | ✓         |
    | End-to-End First-Packet Latency: 234/547ms | | | |

Table 1 展示了 Qwen3-Omni-30B-A3B 的架构设计及其在音频/视频场景下的端到端首包延迟。可以看到各个模块的参数量以及是否支持流式传输。

Table 2: Theoretical First-Packet Latency of Qwen3-Omni wit Different Concurrency.

| | Qwen3-Omni-30B-A3B | | |
|---|---|---|---|
| | 1Concurrency | 4Concurrency | 6Concurrency |
| Thinker-Talker Tail Packet Preprocessing Latency | 72/160ms | 94/180ms | 100/200ms |
| Thinker Time-to-First-Token (TTPT) | 88/160ms | 468/866ms | 673/1330ms |
| Talker Time-to-First-Token (TTPT) | 57/210ms | 145/450ms | 376/734ms |
| MTP Module Time Cost Per Token | 14ms | 16ms | 18ms |
| Codec Decoder Time Cost Per Code | 3ms | 5ms | 5ms |
| Overral Latency (Audio/Video) | 234/547ms | 728/1517ms | 1172/2284ms |
| Thinker Token Generation Rate (TPS) | 75 tokens/s | 63 tokens/s | 53 tokens/s |
| Talker Token Generation Rate (TPS) | 140 tokens/s | 125 tokens/s | 110 tokens/s |
| Generation RTF(Real Time Factor) | 0.47 | 0.56 | 0.66 |

Table 2 展示了 Qwen3-Omni 在不同并发 (concurrency) 情景下的理论首包延迟。实验在 vLLM 框架上进行，并通过 `torch.compile` 和 CUDA Graph 加速 MTP 模块和编码器解码器。总首包延迟是各个组件延迟之和。结果表明，Thinker 和 Talker 的 MoE 架构确保了其预取延迟和 TTFT 在高并发下基本不受影响。轻量级的 MTP 模块和编码器解码器最大限度地减少了计算开销，对首包延迟影响较小。此外，生成实时因子 (Real Time Factor, RTF) 计算为 Thinker 和 Talker 生成一个词元的时间加上 MTP 模块和编码器解码器每词元的处理时间，再除以 $80\mathrm{ms}$ (因为 $12.5\mathrm{Hz}$ 词元率意味着一个词元对应 $80\mathrm{ms}$ 音频)。RTF 始终低于1，确保了用户可以接收到连续流式音频响应。

### 4.2.6. 预训练 (Pretraining)
Qwen3-Omni 的预训练分为三个阶段：

1.  <strong>编码器对齐阶段 (Encoder Alignment Stage, S1)</strong>：
    *   **LLM 初始化**：LLM 组件使用 Qwen3 (Yang et al., 2025a) 的参数初始化。
    *   **视觉编码器**：采用 Qwen3-VL 的视觉编码器。
    *   **音频编码器**：使用 AuT 初始化。
    *   **训练策略**：LLM 参数被锁定。视觉和音频编码器独立训练，首先训练各自的适配器 (adapters)，然后训练编码器。
    *   **与前人工作对比**：与 Bai et al. (2025); Xu et al. (2025) 不同，该阶段放弃了编码器和适配器在 LLM 冻结情况下共同训练的方法，因为那可能导致编码器补偿冻结 LLM 的限制，从而降低感知能力。

2.  <strong>通用阶段 (General Stage, S2)</strong>：
    *   **参数解冻**：所有参数都被解冻。
    *   **数据量**：使用一个包含约2万亿词元 (tokens) 的大规模数据集。
    *   **数据分布**：文本 (0.57万亿)、音频 (0.77万亿)、图像 (0.82万亿)、视频 (0.05万亿) 和视听 (0.05万亿)。
    *   **目标**：引入更多样化的多模态数据和任务，增强模型在听觉、视觉、文本和视听信息方面的理解和交互能力。

3.  <strong>长上下文阶段 (Long Context Stage, S3)</strong>：
    *   **最大词元长度**：将最大词元长度从8,192增加到32,768。
    *   **数据比例**：提高了训练数据中长音频和长视频的比例。
    *   **目标**：显著提升模型理解复杂长序列数据的能力。

        Table 3: Languages and dialects support of Qwen3-Omni-30B-A3B.

        | Modality    | # Langs | Languages                                                               |
        |-------------|---------|-------------------------------------------------------------------------|
        | Text        | 119     | See Qwen3 for the full list.                                            |
        | Speech Input| 19      | ar, de, en, es, fr, id, it, ja, ko, ms, nl, pt, ru, th, tr, ur, vi, yue, zh |
        | Speech Output| 10     | de, en, es, fr, it, ja, ko, pt, ru, zh                                  |

Table 3 列出了 Qwen3-Omni-30B-A3B 支持的语言和方言。

### 4.2.7. 后训练 (Post-training)

#### 4.2.7.1. Thinker
Thinker 的后训练包含一个三阶段训练过程，使其具备指令遵循 (instruction-following) 能力。数据集采用 ChatML (OpenAI, 2022) 格式，包括纯文本对话、视觉模态对话、音频模态对话和混合模态对话数据。

1.  <strong>监督微调 (Supervised Fine-Tuning, SFT)</strong>：
    *   **目的**：通过有针对性的指令优化，弥合预训练表示与下游任务要求之间的差距。
    *   **策略**：SFT 有意偏离预训练数据模式，但保持与预训练模型架构的一致性，以实现高效知识迁移并保留预训练特征的完整性。

2.  <strong>蒸馏 (Distillation)</strong>：
    *   采用 Qwen3 (Yang et al., 2025a) 中描述的“强到弱蒸馏 (Strong-to-Weak Distillation)”管线，进一步提高模型性能。
    *   **两阶段蒸馏**：
        *   <strong>离策略蒸馏 (Off-policy Distillation)</strong>：初始阶段，结合教师模型 (teacher models) 生成的输出进行响应蒸馏。这有助于轻量级学生模型 (student models) 获得基本推理能力，为后续的在策略训练奠定基础。
        *   <strong>在策略蒸馏 (On-policy Distillation)</strong>：第二阶段，学生模型根据采样提示 (sampled prompts) 生成响应。这些在策略序列用于微调，通过最小化 Kullback-Leibler (KL) 散度 (KL divergence)，使学生模型的预测 logits 与教师模型 (Qwen3-32B 或 Qwen3-235B-A22B) 的 logits 对齐。
            *   <strong>KL 散度 (Kullback-Leibler Divergence)</strong>：在概率论或信息论中，KL 散度是衡量两个概率分布之间差异的非对称度量。对于两个概率分布 $P$ 和 $Q$，其 KL 散度定义为：
                $$
                D_{KL}(P || Q) = \sum_{x \in X} P(x) \log\left(\frac{P(x)}{Q(x)}\right)
                $$
                *   `P(x)`：真实的概率分布。
                *   `Q(x)`：模型预测的概率分布。
                *   $x$：事件或类别。
                *   $\sum_{x \in X}$：对所有可能的事件或类别求和。
                *   $\log$：通常取自然对数。
                *   **目的**：当 $D_{KL}(P || Q)$ 越接近0，表示 $Q$ 越接近 $P$。在蒸馏中，我们希望学生模型的输出分布 $Q$ 尽可能地接近教师模型的输出分布 $P$，因此最小化 KL 散度可以促使学生模型学习教师模型的知识。

3.  **GSPO (Group Sequence Policy Optimization)**：
    *   最后，利用 GSPO (Zheng et al., 2025) 全面增强模型在文本、图像、视频和音频等各种模态下的能力和稳定性。
    *   <strong>反馈奖励 (Feedback Rewards)</strong>：
        *   <strong>基于规则的奖励 (Rule-based Reward)</strong>：对于可验证的多模态任务（例如，数学、编码、指令遵循），奖励信号来自一组预定义的规则。精心设计的基于规则的奖励可以高精度地评估模型输出的正确性，防止奖励操纵 (reward hacking) 等问题。
        *   <strong>基于模型的奖励 (Model-based Reward)</strong>：对于缺乏客观、预定义评估指标的多模态任务，采用“LLM 作为评判者 (LLM-as-a-judge)”协议。自动化评估器由 Qwen3 担任通用任务的评判者，而专业视觉-语言模型 Qwen2.5-VL 用于视觉相关任务。为确保更稳健和有依据的评估，LLM 评估器在适用情况下会提供相应查询的真实标注数据 (ground-truth) 或参考答案。

#### 4.2.7.2. Talker
Talker 的训练也分为四阶段，使其能够在文本生成的同时生成语音响应。所有训练数据同样采用 ChatML 格式。

1.  **多模态上下文下的语音数据训练**：
    *   **目的**：建立从多模态表示到语音的单调映射。
    *   **数据**：利用数亿语音数据与多模态上下文进行训练。

2.  <strong>持续预训练 (Continual Pretraining, CPT)</strong>：
    *   **目的**：使用高质量数据进行 CPT，缓解第一阶段中噪声数据引起的幻觉，显著提高生成语音的质量。
    *   **长上下文训练**：同时进行长上下文训练，增强 Talker 处理扩展和复杂输入并生成上下文适宜的语音响应的能力。

3.  <strong>直接偏好优化 (Direct Preference Optimization, DPO)</strong>：
    *   **目的**：为提高多语言语音生成和系统稳定性，从多样化的多语言语音样本构建偏好对 (preference pairs)，并使用 DPO (Rafailov et al., 2023) 优化模型。
    *   **DPO 介绍**：DPO 是一种通过直接优化策略以匹配偏好数据的强化学习算法，避免了传统 RLHF 中复杂的奖励模型训练。它直接利用人类偏好数据（例如，“回应 A 比回应 B 更好”）来调整策略模型，使其生成更受偏好的结果。其核心思想是，人类对模型输出的偏好实际上编码了潜在的奖励函数，DPO 尝试直接找到一个策略，该策略根据这个隐式奖励函数表现最优。
        *   **基本原理**：DPO 将策略优化的目标直接与偏好数据联系起来，通常通过一个简单的损失函数实现。对于一对偏好样本 $(x, y_w, y_l)$，其中 $x$ 是提示，$y_w$ 是被偏好的回应，$y_l$ 是未被偏好的回应，DPO 旨在最大化 $P(y_w|x)$ 相对于 $P(y_l|x)$ 的似然比 (likelihood ratio)。这通常通过一个二元交叉熵损失 (binary cross-entropy loss) 实现。
        *   **损失函数**：DPO 的损失函数通常可以写为：
            $$
            L_{DPO}(\pi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \left( r_\pi(x, y_w) - r_\pi(x, y_l) \right) \right) \right]
            $$
            其中 $r_\pi(x,y)$ 是从策略 $\pi$ 和参考策略 $\pi_{ref}$ 导出的隐式奖励函数，通常由以下形式给出：
            $$
            r_\pi(x,y) = \beta^{-1} \log \frac{\pi(y|x)}{\pi_{ref}(y|x)}
            $$
            *   $\pi$：当前要优化的策略模型。
            *   $\pi_{ref}$：参考策略模型（通常是 SFT 后的模型，固定不变）。
            *   $\mathcal{D}$：人类偏好数据集。
            *   $\sigma(\cdot)$：Sigmoid 函数。
            *   $\beta$：超参数，控制奖励差异的尺度。
            *   **目的**：最小化这个损失函数会促使 $\pi(y_w|x)$ 的概率相对于 $\pi(y_l|x)$ 提高，从而使得模型更倾向于生成人类偏好的响应。

4.  <strong>说话人微调 (Speaker Fine-tuning)</strong>：
    *   在上述基础模型上应用说话人微调，使 Talker 能够采用特定声音 (specific voices)，同时细化其语音响应的自然度、表现力和可控性。

#### 4.2.7.3. 字幕生成器 (Captioner)
Qwen3-Omni-30B-A3B-Captioner 是通过对 Qwen3-Omni-30B-A3B 在大规模详细音频描述数据集上进行微调而开发的。其目的是解决研究社区目前缺乏通用音频字幕生成模型的问题，并促进多模态感知的全面研究。该模型能够为任意音频输入生成详细且幻觉较少的字幕。

# 5. 实验设置

## 5.1. 数据集
Qwen3-Omni 的实验使用了涵盖文本、音频、视觉和视听模态的广泛数据集。

**预训练阶段使用的数据集**：
*   <strong>音频-文本 (Audio-Text)</strong>：用于 AuT 编码器训练，包括2000万小时的监督音频数据，其中80%是中文和英文的伪标注 ASR 数据，10%是其他语言的 ASR 数据，10%是音频理解数据。
*   <strong>图像-文本 (Image-Text)</strong>：用于视觉编码器和通用阶段训练。
*   <strong>视频-文本 (Video-Text)</strong>：用于通用阶段训练。
*   <strong>视频-音频 (Video-Audio)</strong>：用于通用阶段训练。
*   <strong>视频-音频-文本 (Video-Audio-Text)</strong>：用于通用阶段训练。
*   <strong>纯文本 (Pure Text Corpora)</strong>：用于通用阶段训练。

    <strong>后训练阶段（Thinker 和 Talker）使用的数据集</strong>：
*   **ChatML 格式数据**：所有训练数据都以 ChatML (OpenAI, 2022) 格式构建，确保与 Thinker 和 Talker 的输入一致性。
    *   **Thinker 数据**：包括纯文本对话、视觉模态对话、音频模态对话和混合模态对话数据。
    *   **Talker 数据**：数亿带有多模态上下文的语音数据，以及高质量的数据用于持续预训练。为 DPO 优化构建多样化的多语言语音样本偏好对。

        <strong>字幕生成器 (Captioner) 使用的数据集</strong>：
*   <strong>大规模详细音频描述数据集 (Large-scale dataset of detailed audio descriptions)</strong>：用于微调 Qwen3-Omni-30B-A3B 以获得 Qwen3-Omni-30B-A3B-Captioner。

    **评估阶段使用的基准数据集**：

*   <strong>文本 $\rightarrow$ 文本 (Text $\rightarrow$ Text)</strong>：
    *   <strong>通用任务 (General Tasks)</strong>：
        *   MMLU-Redux (Gema et al., 2024)：多学科多选知识问答。
        *   GPQA (Rein et al., 2023)：高难度通用问答。
    *   <strong>推理能力 (Reasoning Ability)</strong>：
        *   AIME25 (AIME, 2025)：美国数学邀请赛题目。
        *   ZebraLogic (Lin et al., 2025)：逻辑推理。
    *   <strong>编码能力 (Coding Ability)</strong>：
        *   MultiPL-E (Cassano et al., 2023)：多语言代码生成。
    *   <strong>对齐任务 (Alignment Tasks)</strong>：
        *   IFEval (Zhou et al., 2023)：指令遵循评估。
        *   Creative Writing V3 (Paech, 2024)：创意写作。
        *   WritingBench (Wu et al., 2025b)：写作能力评估。
    *   <strong>智能体 (Agent)</strong>：
        *   BFCL-v3 (Yan et al., 2024)：Berkeley 函数调用排行榜。
    *   <strong>多语言任务 (Multilingual Tasks)</strong>：
        *   MultiIF (He et al., 2024)：多语言指令遵循。
        *   PolyMath (Wang et al., 2025c)：多语言数学推理。

*   <strong>音频 $\rightarrow$ 文本 (Audio $\rightarrow$ Text)</strong>：
    *   **基本音频任务**：
        *   ASR (Automatic Speech Recognition) & S2TT (Speech-to-Text Translation)：
            *   Wenetspeech net & meeting：中文和英文语音识别。
            *   Librispeech clean & other：英文语音识别。
            *   CV15-en & CV15-zh：CommonVoice 15 英文和中文语音识别。
            *   Fleurs-en & Fleurs-zh：Fleurs 英文和中文语音识别。
            *   Fleurs-avg (19 lang)：19种语言的平均语音识别。
            *   MIR-1K (vocal-only)：歌词 ASR。
            *   Opencpop-test：中文流行歌曲 ASR。
            *   Fleurs-en2xx, Fleurs-xx2en, Fleurs-zh2xx, Fleurs-xx2zh：Fleurs 语音到文本翻译基准。
    *   **高级音频任务**：
        *   语音聊天 (Voice Chatting)：
            *   VoiceBench (Chen et al., 2024b)：评估 LLM 驱动的语音助手。
        *   音频推理 (Audio Reasoning)：
            *   MMAU (Sakshi et al., 2024)。
            *   MMSU (Wang et al., 2025a)。
        *   音乐理解 (Music Understanding)：
            *   RUL-MuchoMusic (Zang et al., 2025)：综合音乐理解评估。
            *   GTZAN (Tzanetakis and Cook, 2002)：音乐流派识别。
            *   MTG-Jamendo (Bogdanov et al., 2019)：音乐标签、情绪、主题、乐器识别。
            *   MagnaTagATune (Law et al., 2009)：音乐关键词标注。

*   <strong>视觉 $\rightarrow$ 文本 (Vision $\rightarrow$ Text)</strong>：
    *   <strong>通用视觉问答 (General Visual Question Answering)</strong>：
        *   MMStar (Chen et al., 2024a)。
        *   HallusionBench (Guan et al., 2024)：幻觉和视觉错觉诊断。
        *   MM-MT-Bench (Agrawal et al., 2024)。
    *   <strong>数学与 STEM 推理 (Mathematical and STEM Reasoning)</strong>：
        *   MathVista (Lu et al., 2024)。
        *   MathVision (Wang et al., 2024a)。
        *   MMMU (Yue et al., 2023) 和 MMMU-Pro (Yue et al., 2024)。
    *   <strong>文档理解 (Document Understanding)</strong>：
        *   AI2D (Kembhavi et al., 2016)。
        *   ChartQA (Masry et al., 2022)。
    *   <strong>数值推理与计数 (Numerical Reasoning and Counting)</strong>：
        *   CountBench (Paiss et al., 2023)。
    *   <strong>长视频理解 (Long Video Understanding)</strong>：
        *   Video-MME (Fu et al., 2024)。
        *   LVBench (Wang et al., 2024b)。
        *   MLVU (Zhou et al., 2025a)。

*   <strong>视听视频 $\rightarrow$ 文本 (AudioVisual Video $\rightarrow$ Text)</strong>：
    *   **通用理解**：
        *   WorldSense (Hong et al., 2025)：视觉和听觉信号集成。
    *   **高级认知功能**：
        *   DailyOmni (Zhou et al., 2025b)。
        *   VideoHolmes (Cheng et al., 2025)。

*   <strong>X $\rightarrow$ 语音 (X $\rightarrow$ Speech)</strong>：
    *   <strong>零样本语音生成 (Zero-Shot Speech Generation)</strong>：
        *   SEED (Anastassiou et al., 2024)：评估内容一致性 (WER) 和说话人相似度 (SIM)。
    *   <strong>多语言语音生成 (Multilingual Speech Generation)</strong>：
        *   MiniMax 多语言测试集 (Zhang et al., 2025)。
    *   <strong>跨语言语音生成 (Cross-Linguial Speech Generation)</strong>：
        *   CV3-Eval (Du et al., 2025)。

## 5.2. 评估指标
论文中使用了多种评估指标，涵盖了文本、音频、视觉、视听的理解和生成任务。

### 5.2.1. 词错误率 (Word Error Rate, WER)
*   **概念定义**：词错误率 (WER) 是衡量语音识别或语音转文本系统性能的常用指标。它计算识别出的词序列与参考词序列之间需要进行替换、插入或删除操作的最小次数，然后除以参考词序列的总词数。WER 越低表示系统性能越好。
*   **数学公式**：
    $$
    \text{WER} = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}
    $$
*   **符号解释**：
    *   $S$：替换 (substitutions) 的数量，即识别出的词与参考词不匹配。
    *   $D$：删除 (deletions) 的数量，即参考词中存在但在识别结果中缺失的词。
    *   $I$：插入 (insertions) 的数量，即识别结果中存在但在参考词中不存在的词。
    *   $N$：参考词序列的总词数，通常等于 $S + D + C$。
    *   $C$：正确识别的词数。

### 5.2.2. BLEU 分数 (Bilingual Evaluation Understudy, BLEU)
*   **概念定义**：BLEU 是一种用于评估机器翻译文本质量的算法。它通过比较机器翻译文本与高质量参考译文（通常由人类翻译）之间的 n-gram (连续的 n 个词组成的序列) 匹配程度来衡量翻译的准确性和流畅性。BLEU 分数介于0到1之间，分数越高表示翻译质量越好。
*   **数学公式**：
    $$
    \text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
    $$
    其中，$\text{BP}$ (Brevity Penalty) 是简洁惩罚，用于惩罚过短的机器翻译：
    $$
    \text{BP} = \begin{cases}
    1 & \text{if } c > r \\
    e^{(1 - r/c)} & \text{if } c \le r
    \end{cases}
    $$
    $p_n$ 是修改后的 n-gram 精度 (precision)：
    $$
    p_n = \frac{\sum_{\text{sentence} \in \text{MT}} \sum_{n\text{-gram} \in \text{sentence}} \min(\text{count}(\text{n-gram}), \text{max\_ref\_count}(\text{n-gram}))}{\sum_{\text{sentence} \in \text{MT}} \sum_{n\text{-gram} \in \text{sentence}} \text{count}(\text{n-gram})}
    $$
*   **符号解释**：
    *   $N$：最大 n-gram 长度，通常取4。
    *   $w_n$：n-gram 权值，通常设为 $1/N$。
    *   $p_n$：n-gram 精度。
    *   $c$：机器翻译文本的总词数。
    *   $r$：参考翻译文本中最接近机器翻译文本长度的参考译文的词数。
    *   $\text{count}(\text{n-gram})$：n-gram 在机器翻译中出现的次数。
    *   $\text{max\_ref\_count}(\text{n-gram})$：n-gram 在所有参考译文中出现的最大次数。
    *   $\sum_{\text{sentence} \in \text{MT}}$：对机器翻译中的所有句子求和。

### 5.2.3. 微 F1 分数 (Micro F1 Score)
*   **概念定义**：F1 分数是精确率 (Precision) 和召回率 (Recall) 的调和平均值，常用于评估分类模型的性能，尤其是在类别不平衡的情况下。微 F1 分数 (Micro F1) 是通过对所有类别中的真阳性 (True Positives, TP)、假阳性 (False Positives, FP) 和假阴性 (False Negatives, FN) 进行累加，然后计算总体的精确率和召回率，最后计算 F1 分数。它给予每个样本相同的权重，更关注样本数量多的类别。
*   **数学公式**：
    首先计算总体的真阳性 (TP)、假阳性 (FP) 和假阴性 (FN)：
    $$
    \text{Micro-TP} = \sum_{c=1}^{C} \text{TP}_c \\
    \text{Micro-FP} = \sum_{c=1}^{C} \text{FP}_c \\
    \text{Micro-FN} = \sum_{c=1}^{C} \text{FN}_c
    $$
    然后计算微精确率 (Micro Precision) 和微召回率 (Micro Recall)：
    $$
    \text{Micro-Precision} = \frac{\text{Micro-TP}}{\text{Micro-TP} + \text{Micro-FP}} \\
    \text{Micro-Recall} = \frac{\text{Micro-TP}}{\text{Micro-TP} + \text{Micro-FN}}
    $$
    最后计算微 F1 分数：
    $$
    \text{Micro-F1} = 2 \cdot \frac{\text{Micro-Precision} \cdot \text{Micro-Recall}}{\text{Micro-Precision} + \text{Micro-Recall}}
    $$
*   **符号解释**：
    *   $C$：类别的总数。
    *   $\text{TP}_c$：类别 $c$ 的真阳性数量。
    *   $\text{FP}_c$：类别 $c$ 的假阳性数量。
    *   $\text{FN}_c$：类别 $c$ 的假阴性数量。

### 5.2.4. 阿尔帕卡评估分数 (AlpacaEval)
*   **概念定义**：AlpacaEval 是一种基于 LLM 自动评估的指标，用于衡量模型在遵循指令和生成高质量响应方面的能力。它通常涉及一个强大的 LLM（例如 GPT-4）作为评估者，比较不同模型对同一指令的响应质量。分数越高，表示模型表现越好。

### 5.2.5. 内容一致性 (Content Consistency)
*   **概念定义**：在语音生成任务中，内容一致性衡量生成语音的语义内容与输入文本的匹配程度。通常通过计算生成语音的 ASR 结果与原始文本之间的 WER 来量化。WER 越低，内容一致性越高。

### 5.2.6. 说话人相似度 (Speaker Similarity, SIM)
*   **概念定义**：在语音克隆或多说话人语音生成任务中，说话人相似度 (SIM) 衡量生成语音的音色、语调等说话人特征与目标说话人（通常通过参考音频提供）的匹配程度。高 SIM 分数表示生成语音更像目标说话人。通常使用预训练的说话人嵌入模型（如 ECAPA-TDNN）来提取嵌入向量，然后计算生成语音和参考语音嵌入向量之间的余弦相似度 (cosine similarity)。

### 5.2.7. 精度 (Accuracy, Acc.)
*   **概念定义**：精度是分类任务中最直观的指标，表示模型正确预测的样本数量占总样本数量的比例。
*   **数学公式**：
    $$
    \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
    $$
*   **符号解释**：
    *   Number of Correct Predictions：模型正确分类的样本数量。
    *   Total Number of Predictions：总的预测样本数量。

### 5.2.8. 实时因子 (Real Time Factor, RTF)
*   **概念定义**：实时因子 (RTF) 衡量语音生成或处理系统相对于音频实际持续时间的处理速度。RTF 小于1表示系统处理速度快于实时，可以进行实时流式处理；RTF 大于1则表示处理速度慢于实时。
*   **数学公式**：
    $$
    \text{RTF} = \frac{\text{Time taken to process audio}}{\text{Duration of audio}}
    $$
*   **符号解释**：
    *   Time taken to process audio：系统处理一段音频所需的时间。
    *   Duration of audio：这段音频的实际持续时间。

## 5.3. 对比基线
论文将 Qwen3-Omni 与以下主要基线模型进行了比较：

*   <strong>文本 $\rightarrow$ 文本 (Text $\rightarrow$ Text)</strong>：
    *   **闭源 SOTA 模型**：GPT-4o-0327 (OpenAI)、Gemini-2.5-Flash-Thinking (Google)。
    *   **Qwen 系列内部基线**：
        *   Qwen3-235B-A22B Non-Thinking
        *   Qwen3-235B-A22B Thinking
        *   Qwen3-30B-A3B-Instruct-2507 (文本专用模型)
        *   Qwen3-30B-A3B-Thinking-2507 (文本专用模型)

*   <strong>音频 $\rightarrow$ 文本 (Audio $\rightarrow$ Text)</strong>：
    *   **专业 ASR 模型**：Seed-ASR (Anastassiou et al., 2024)、Voxtral-Mini、Voxtral-Small。
    *   **通用多模态模型**：GPT-4o-Transcribe (OpenAI)、Gemini-2.5-Pro (Google)、Gemini-2.5-Flash (Google)、GPT-4o-Audio (OpenAI)。
    *   **Qwen 系列内部基线**：Qwen2.5-Omni。
    *   **音乐专业模型**：Audio Flamingo 3 (Goel et al., 2025)、CLaMP 3 (Wu et al., 2025a)、MuQ-MuLan (Zhu et al., 2025)、MuQ (Zhu et al., 2025)。

*   <strong>视觉 $\rightarrow$ 文本 (Vision $\rightarrow$ Text)</strong>：
    *   **闭源 SOTA 模型**：GPT4-o (OpenAI)、Gemini-2.0-Flash (Google)、Gemini-2.5-Flash-Thinking (Google)。
    *   **Qwen 系列内部基线**：Qwen2.5-VL-72B、InternVL-3.5-241B-A28B。

*   <strong>视听视频 $\rightarrow$ 文本 (AudioVisual Video $\rightarrow$ Text)</strong>：
    *   **闭源 SOTA 模型**：Gemini-2.5-Flash (Google)、Gemini-2.5-Flash-Thinking (Google)。
    *   **Qwen 系列内部基线**：Qwen2.5-Omni。
    *   **其他开源 SOTA 模型**：WorldSense (Yang et al., 2025b)、DailyOmni (Tang et al., 2025)、VideoHolmes (Tang et al., 2025)。

*   <strong>X $\rightarrow$ 语音 (X $\rightarrow$ Speech)</strong>：
    *   **零样本 TTS 模型**：Seed-TTSICL (Anastassiou et al., 2024)、Seed-TTSRL (Anastassiou et al., 2024)、MaskGCT (Wang et al., 2024c)、E2 TTS (Eskimez et al., 2024)、F5-TTS (Chen et al., 2024c)、Spark TTS (Wang et al., 2025b)、CosyVoice 2 (Du et al., 2024)、CosyVoice 3 (Du et al., 2025)。
    *   **多语言 TTS 模型**：MiniMax-Speech (Zhang et al., 2025)、ElevenLabs Multilingual v2。
    *   **跨语言 TTS 模型**：CosyVoice2 (Du et al., 2024)、CosyVoice3 (Du et2 al., 2025)。
    *   **Qwen 系列内部基线**：Qwen2.5-Omni-7B (Xu et al., 2025)。

        这些基线模型涵盖了各种规模、模态能力（单模态专家、多模态通用模型）和技术类型（开源、闭源），为 Qwen3-Omni 的性能提供了全面的比较参照。

# 6. 实验结果与分析

## 6.1. 核心结果分析
Qwen3-Omni 的实验结果表明其在多模态任务上表现出色，尤其在音频领域达到领先水平，并成功实现了各模态间性能的非退化。

### 6.1.1. 文本 $\rightarrow$ 文本 (Text $\rightarrow$ Text)
以下是原文 Table 4 的结果：

<table>
<thead>
<tr>
<th></th>
<th></th>
<th>GPT-4o-0327</th>
<th>Qwen3-235B-A22B Non Thinking</th>
<th>Qwen3-30B-A3B -Instruct-2507</th>
<th>Qwen3-Omni-30B-A3B -Instruct</th>
<th>Qwen3-Omni-Flash -Instruct</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">GeneralTasks</td>
<td>MMLU-Redux</td>
<td>91.3</td>
<td>89.2</td>
<td>89.3</td>
<td>86.6</td>
<td>86.8</td>
</tr>
<tr>
<td>GPQA</td>
<td>66.9</td>
<td>62.9</td>
<td>70.4</td>
<td>69.6</td>
<td>69.7</td>
</tr>
<tr>
<td rowspan="2">Reasoning</td>
<td>AIME25</td>
<td>26.7</td>
<td>24.7</td>
<td>61.3</td>
<td>65.0</td>
<td>65.9</td>
</tr>
<tr>
<td>ZebraLogic</td>
<td>52.6</td>
<td>37.7</td>
<td>90.0</td>
<td>76.0</td>
<td>76.1</td>
</tr>
<tr>
<td>Code</td>
<td>MultiPL-E</td>
<td>82.7</td>
<td>79.3</td>
<td>83.8</td>
<td>81.4</td>
<td>81.5</td>
</tr>
<tr>
<td rowspan="3">Alignment Tasks</td>
<td>IFEval</td>
<td>83.9</td>
<td>83.2</td>
<td>84.7</td>
<td>81.0</td>
<td>81.7</td>
</tr>
<tr>
<td>Creative Writing v3</td>
<td>84.9</td>
<td>80.4</td>
<td>86.0</td>
<td>80.6</td>
<td>81.8</td>
</tr>
<tr>
<td>WritingBench</td>
<td>75.5</td>
<td>77.0</td>
<td>85.5</td>
<td>82.6</td>
<td>83.0</td>
</tr>
<tr>
<td>Agent</td>
<td>BFCL-v3</td>
<td>66.5</td>
<td>68.0</td>
<td>65.1</td>
<td>64.4</td>
<td>65.0</td>
</tr>
<tr>
<td rowspan="2">Multilingual Tasks</td>
<td>MultiIF</td>
<td>70.4</td>
<td>70.2</td>
<td>67.9</td>
<td>64.0</td>
<td>64.7</td>
</tr>
<tr>
<td>PolyMATH</td>
<td>25.5</td>
<td>27.0</td>
<td>43.1</td>
<td>37.9</td>
<td>39.3</td>
</tr>
</tbody>
</table>

<p>Table 4: Text $\rightarrow$ Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.</p>

以下是原文 Table 5 的结果：

<table>
<thead>
<tr>
<th></th>
<th></th>
<th>Gemini-2.5-Flash Thinking</th>
<th>Qwen3-235B-A22B Thinking</th>
<th>Qwen3-30B-A3B -Thinking-2507</th>
<th>Qwen3-Omni-30B-A3B -Thinking</th>
<th>Qwen3-Omni-Flash -Thinking</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">General Tasks</td>
<td>MMLU-Redux</td>
<td>92.1</td>
<td>92.7</td>
<td>91.4</td>
<td>88.8</td>
<td>89.7</td>
</tr>
<tr>
<td>GPQA</td>
<td>82.8</td>
<td>71.1</td>
<td>73.4</td>
<td>73.1</td>
<td>73.1</td>
</tr>
<tr>
<td rowspan="2">Reasoning</td>
<td>AIME25</td>
<td>72.0</td>
<td>81.5</td>
<td>85.0</td>
<td>73.7</td>
<td>74.0</td>
</tr>
<tr>
<td>LiveBench 20241125</td>
<td>74.3</td>
<td>77.1</td>
<td>76.8</td>
<td>71.8</td>
<td>70.3</td>
</tr>
<tr>
<td>Code</td>
<td>MultiPL-E</td>
<td>84.5</td>
<td>79.9</td>
<td>81.3</td>
<td>80.6</td>
<td>81.0</td>
</tr>
<tr>
<td rowspan="4">Alignment Tasks</td>
<td>IFEval</td>
<td>89.8</td>
<td>83.4</td>
<td>88.9</td>
<td>85.1</td>
<td>85.2</td>
</tr>
<tr>
<td>Arena-Hard v2</td>
<td>56.7</td>
<td>61.5</td>
<td>56.0</td>
<td>55.1</td>
<td>57.8</td>
</tr>
<tr>
<td>Creative Writing v3</td>
<td>85.0</td>
<td>84.6</td>
<td>84.4</td>
<td>82.5</td>
<td>83.6</td>
</tr>
<tr>
<td>WritingBench</td>
<td>83.9</td>
<td>80.3</td>
<td>85.0</td>
<td>85.5</td>
<td>85.9</td>
</tr>
<tr>
<td>Agent</td>
<td>BFCL-v3</td>
<td>68.6</td>
<td>70.8</td>
<td>72.4</td>
<td>63.2</td>
<td>64.5</td>
</tr>
<tr>
<td rowspan="2">Multilingual Tasks</td>
<td>MultiIF</td>
<td>74.4</td>
<td>71.9</td>
<td>76.4</td>
<td>72.9</td>
<td>73.2</td>
</tr>
<tr>
<td>PolyMATH</td>
<td>49.8</td>
<td>54.7</td>
<td>52.6</td>
<td>47.1</td>
<td>48.7</td>
</tr>
</tbody>
</table>

<p>Table 5: Text $\rightarrow$ Text performance of Qwen3-Omni-Thinking and other reasoning baselines. The highest scores are shown in bold.</p>

*   **Instruct 模型**：Qwen3-Omni-30B-A3B-Instruct 尽管参数量较小，但在 GPQA, AIME25, ZebraLogic, WritingBench 和 PolyMath 等多个基准测试中超越了更大的开源模型 Qwen3-235B-A22B Non-Thinking 和强大的闭源模型 GPT-4o-0327。这表明 Qwen3-Omni 在文本任务上的指令遵循能力和推理能力非常强大。
*   **Thinking 模型**：Qwen3-Omni-30B-A3B-Thinking 的性能与 Gemini-2.5-Flash-Thinking 和 Qwen3-235B-A22B Non-Thinking 相当，表明其在文本推理方面具备与大型模型竞争的能力。
*   **与文本专用模型持平**：Qwen3-Omni-30B-A3B 在文本能力上与同尺寸的文本专用模型（Qwen3-30B-A3B-Instruct-2507 和 Qwen3-30B-A3B-Thinking-2507）持平，证明了其在集成多模态能力的同时，没有牺牲核心的文本性能。

### 6.1.2. 音频 $\rightarrow$ 文本 (Audio $\rightarrow$ Text)
以下是原文 Table 6 的结果：

<table>
<thead>
<tr>
<th></th>
<th></th>
<th>Seed -ASR</th>
<th>Voxtral -Mini</th>
<th>Voxtral -Small</th>
<th>GPT-4o -Transcribe</th>
<th>Gemini-2.5 -Pro</th>
<th>Qwen2.5 -Omni</th>
<th>Qwen3-Omni -30B-A3B-Instruct</th>
<th>Qwen3-Omni -Flash-Instruct</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">EN &amp; ZH ASR (wer)</td>
<td>Wenetspeech net</td>
<td>4.66</td>
<td>5.69</td>
<td>24.30</td>
<td>31.53</td>
<td>20.33</td>
<td>26.08</td>
<td>15.30</td>
<td>32.27</td>
</tr>
<tr>
<td>meeting</td>
<td>14.43</td>
<td>13.47</td>
<td>5.91</td>
<td>7.65</td>
<td>4.69</td>
<td>5.89</td>
<td>4.62</td>
<td>5.75</td>
</tr>
<tr>
<td>Librispeech clean</td>
<td>1.58</td>
<td>2.84</td>
<td>1.88</td>
<td>4.12</td>
<td>1.56</td>
<td>3.30</td>
<td>1.39</td>
<td>3.75</td>
</tr>
<tr>
<td>other</td>
<td>2.89</td>
<td>3.56</td>
<td>1.74</td>
<td>3.45</td>
<td>1.22</td>
<td>2.48</td>
<td>1.27</td>
<td>2.44</td>
</tr>
<tr>
<td rowspan="2"></td>
<td>CV15-en</td>
<td>-</td>
<td>9.47</td>
<td>7.79</td>
<td>10.01</td>
<td>9.89</td>
<td>7.61</td>
<td>6.05</td>
<td>5.94</td>
</tr>
<tr>
<td>CV15-zh</td>
<td>-</td>
<td>24.67</td>
<td>19.30</td>
<td>9.84</td>
<td>8.00</td>
<td>5.13</td>
<td>4.31</td>
<td>4.28</td>
</tr>
<tr>
<td rowspan="2"></td>
<td>Fleurs-en</td>
<td>3.40</td>
<td>3.96</td>
<td>3.77</td>
<td>3.32</td>
<td>2.94</td>
<td>3.77</td>
<td>2.72</td>
<td>2.74</td>
</tr>
<tr>
<td>Fleurs-zh</td>
<td>2.69</td>
<td>12.22</td>
<td>7.98</td>
<td>2.44</td>
<td>2.71</td>
<td>2.54</td>
<td>2.20</td>
<td>2.19</td>
</tr>
<tr>
<td>Multilingual ASR (wer)</td>
<td>Fleurs-avg (19 lang)<sup>a</sup></td>
<td>-</td>
<td>15.67</td>
<td>8.09</td>
<td>4.48</td>
<td>5.55</td>
<td>14.04</td>
<td>5.33</td>
<td>5.31</td>
</tr>
<tr>
<td rowspan="2">Lyric ASR (wer)</td>
<td>MIR-1K (vocal-only)<sup>b</sup></td>
<td>6.45</td>
<td>23.33</td>
<td>18.73</td>
<td>11.87</td>
<td>9.85</td>
<td>8.15</td>
<td>5.90</td>
<td>5.85</td>
</tr>
<tr>
<td>Opencpop-test</td>
<td>2.98</td>
<td>31.01</td>
<td>16.06</td>
<td>7.93</td>
<td>6.49</td>
<td>2.84</td>
<td>1.54</td>
<td>2.02</td>
</tr>
<tr>
<td rowspan="4">S2TT (BLEU)</td>
<td>Fleurs-en2xx<sup>c</sup></td>
<td>-</td>
<td>30.35</td>
<td>37.85</td>
<td>-</td>
<td>39.25</td>
<td>29.22</td>
<td>37.50</td>
<td>36.22</td>
</tr>
<tr>
<td>Fleurs-xx2en</td>
<td>-</td>
<td>27.54</td>
<td>32.81</td>
<td>-</td>
<td>35.41</td>
<td>28.61</td>
<td>31.08</td>
<td>30.71</td>
</tr>
<tr>
<td>Fleurs-zh2xx</td>
<td>-</td>
<td>17.03</td>
<td>22.05</td>
<td>-</td>
<td>26.63</td>
<td>17.97</td>
<td>25.17</td>
<td>25.10</td>
</tr>
<tr>
<td>Fleurs-xx2zh</td>
<td>-</td>
<td>28.75</td>
<td>34.82</td>
<td>-</td>
<td>37.50</td>
<td>27.68</td>
<td>33.13</td>
<td>31.19</td>
</tr>
</tbody>
</table>

<p>Table 6: Transcription performance for Audio→Text tasks (ASR &amp; S2TT), comparing Qwen3-Omni-Instruct with the baselines. The highest scores are shown in bold.</p>
<sup>a</sup> These 19 languages include Arabic, Cantonese, Chinese, Dutch, English, French, German, Indonesian, Italian, Japanese, Korean, Malay, Portuguese, Russian, Spanish, Thai, Turkish, Urdu, Vietnamese.<br>
<sup>b</sup> Transcription is converted into Simplified Chinese.<br>
<sup>c</sup> The results encompass translations across 15 languages: Arabic, Cantonese, Chinese, English, French, German, Indonesian, Italian, Japanese, Korean, Portuguese, Russian, Spanish, Thai, Vietnamese. For notation, "en2xx" denotes translation from English into each of the other 14 target languages, where "xx" ranges over the remaining language codes.

*   **ASR & S2TT 性能**：Qwen3-Omni-Instruct 在英语和中文的 ASR (Librispeech, Wenetspeech, Fleurs, CommonVoice) 和歌词 ASR (MIR-1K, Opencpop-test) 任务上取得了 SOTA 性能。在多语言 ASR 和 S2TT 任务上，其表现也优于或可与 Voxtral-Small 和 Gemini-2.5-Pro 等专业或通用模型匹敌。这充分展示了 Qwen3-Omni 在语音识别和语音翻译方面的强大能力。

    以下是原文 Table 7 的结果：

    <table>
    <thead>
    <tr>
    <th></th>
    <th>GPT-4o -Audio</th>
    <th>Gemini-2.5 -Flash</th>
    <th>Gemini-2.5 -Pro</th>
    <th>Qwen2.5 -Omni</th>
    <th>Qwen3-Omni -30B-A3B-Instruct</th>
    <th>Qwen3-Omni -30B-A3B-Thinking</th>
    <th>Qwen3-Omni -Flash-Instruct</th>
    <th>Qwen3-Omni -Flash-Thinking</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="9">VoiceBench</td>
    </tr>
    <tr>
    <td>AlpacaEval</td>
    <td>95.6</td>
    <td>96.1</td>
    <td>94.3</td>
    <td>89.9</td>
    <td>94.8</td>
    <td>96.4</td>
    <td>95.4</td>
    <td>96.8</td>
    </tr>
    <tr>
    <td>CommonEval</td>
    <td>89.8</td>
    <td>88.3</td>
    <td>88.4</td>
    <td>76.7</td>
    <td>90.8</td>
    <td>90.5</td>
    <td>91.0</td>
    <td>90.9</td>
    </tr>
    <tr>
    <td>WildVoice</td>
    <td>91.6</td>
    <td>92.1</td>
    <td>93.4</td>
    <td>77.7</td>
    <td>91.6</td>
    <td>90.5</td>
    <td>92.3</td>
    <td>90.9</td>
    </tr>
    <tr>
    <td>SD-QA</td>
    <td>75.5</td>
    <td>84.5</td>
    <td>90.1</td>
    <td>56.4</td>
    <td>76.9</td>
    <td>78.1</td>
    <td>76.8</td>
    <td>78.5</td>
    </tr>
    <tr>
    <td>MMSU</td>
    <td>80.3</td>
    <td>66.1</td>
    <td>71.1</td>
    <td>61.7</td>
    <td>68.1</td>
    <td>83.0</td>
    <td>68.4</td>
    <td>84.3</td>
    </tr>
    <tr>
    <td>OpenBookQA</td>
    <td>89.2</td>
    <td>56.9</td>
    <td>92.3</td>
    <td>80.9</td>
    <td>89.7</td>
    <td>94.3</td>
    <td>91.4</td>
    <td>95.0</td>
    </tr>
    <tr>
    <td>BBH</td>
    <td>84.1</td>
    <td>83.9</td>
    <td>92.6</td>
    <td>66.7</td>
    <td>80.4</td>
    <td>88.9</td>
    <td>80.6</td>
    <td>89.6</td>
    </tr>
    <tr>
    <td>IFEval</td>
    <td>76.0</td>
    <td>83.8</td>
    <td>85.7</td>
    <td>53.5</td>
    <td>77.8</td>
    <td>80.6</td>
    <td>75.2</td>
    <td>80.8</td>
    </tr>
    <tr>
    <td>AdvBench</td>
    <td>98.7</td>
    <td>98.9</td>
    <td>98.1</td>
    <td>99.2</td>
    <td>99.3</td>
    <td>97.2</td>
    <td>99.4</td>
    <td>98.9</td>
    </tr>
    <tr>
    <td>Overall</td>
    <td>86.8</td>
    <td>83.4</td>
    <td>89.6</td>
    <td>73.6</td>
    <td>85.5</td>
    <td>88.8</td>
    <td>85.6</td>
    <td>89.5</td>
    </tr>
    <tr>
    <td colspan="9">Audio Reasoning</td>
    </tr>
    <tr>
    <td>MMAU-v05.15.25</td>
    <td>62.5</td>
    <td>71.8</td>
    <td>77.4</td>
    <td>65.5</td>
    <td>77.5</td>
    <td>75.4</td>
    <td>77.6</td>
    <td>76.5</td>
    </tr>
    <tr>
    <td>MMSU</td>
    <td>56.4</td>
    <td>70.2</td>
    <td>77.7</td>
    <td>62.6</td>
    <td>69.0</td>
    <td>70.2</td>
    <td>69.1</td>
    <td>71.3</td>
    </tr>
    </tbody>
    </table>

<p>Table 7: Voice interaction and audio reasoning performance for Audio→Text tasks, comparing Qwen3-Omni with the baselines. The highest scores are shown in bold.</p>

*   **语音交互和音频推理**：在 VoiceBench 语音交互基准上，Qwen3-Omni-Thinking 取得了89.5的平均分，仅次于 Gemini-2.5-Pro (89.6)，表现出强大的语音交互能力。在 MMAU 和 MMSU 音频推理基准上，Qwen3-Omni-Thinking 甚至超越了 Gemini-2.5-Pro 和 GPT-4o-Audio 等强大的闭源模型，证明其在通用音频理解和推理方面的卓越能力。

    以下是原文 Table 8 的结果：

    <table>
    <thead>
    <tr>
    <th></th>
    <th>Best Specialist Models</th>
    <th>GPT-4o -Audio</th>
    <th>Gemini-2.5 -Pro</th>
    <th>Qwen2.5 -Omni</th>
    <th>Qwen3-Omni -30B-A3B-Instruct</th>
    <th>Qwen3-Omni -Flash-Instruct</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>RUL-MuchoMusic</td>
    <td>47.6 (Audio Flamingo 3) (Goel et al., 2025)</td>
    <td>36.1</td>
    <td>49.4</td>
    <td>47.3</td>
    <td>52.0</td>
    <td>52.1</td>
    </tr>
    <tr>
    <td>GTZAN Acc.</td>
    <td>87.9 (CLaMP 3) (Wu et al., 2025a)</td>
    <td>76.5</td>
    <td>81.0</td>
    <td>81.7</td>
    <td>93.0</td>
    <td>93.1</td>
    </tr>
    <tr>
    <td>MTC Genre Micro F1</td>
    <td>35.8 (MuQ-MuLan) (Zhu et al., 2025)</td>
    <td>25.3</td>
    <td>32.6</td>
    <td>32.5</td>
    <td>39.0</td>
    <td>39.5</td>
    </tr>
    <tr>
    <td>MTCMood/Theme Micro F1</td>
    <td>10.9 (MuQ-MuLan) (Zhu et al., 2025)</td>
    <td>11.3</td>
    <td>14.1</td>
    <td>8.9</td>
    <td>21.0</td>
    <td>21.7</td>
    </tr>
    <tr>
    <td>MTCInstrument Micro F1</td>
    <td>39.8 (MuQ-MuLan) (Zhu et al., 2025)</td>
    <td>34.2</td>
    <td>33.0</td>
    <td>22.6</td>
    <td>40.5</td>
    <td>40.7</td>
    </tr>
    <tr>
    <td>MTCTop50 Micro F1</td>
    <td>33.2 (MuQ-MuLan) (Zhu et al., 2025)</td>
    <td>25.0</td>
    <td>26.1</td>
    <td>21.6</td>
    <td>36.7</td>
    <td>36.9</td>
    </tr>
    <tr>
    <td>MagnaTagATune Micro F1</td>
    <td>41.6 (MuQ) (Zhu et al., 2025)</td>
    <td>29.2</td>
    <td>28.1</td>
    <td>30.1</td>
    <td>44.3</td>
    <td>46.8</td>
    </tr>
    </tbody>
    </table>

<p>Table 8: Music understanding performance for Audio→Text tasks, comparing Qwen3-Omni-Instruct with baselines. The highest scores are shown in bold.</p>

*   **音乐理解**：Qwen3-Omni-Instruct 在 RUL-MuchoMusic 上取得了 SOTA 性能。在 GTZAN, MTG-Jamendo 和 MagnaTagATune 上，其分数显著超越了其他音频语言模型（如 Gemini-2.5-Pro 和 GPT-4o-Audio）以及针对这些数据集探测的自监督音乐专业模型。这表明 Qwen3-Omni 在各类音乐理解任务中具有卓越的能力。

### 6.1.3. 视觉 $\rightarrow$ 文本 (Vision $\rightarrow$ Text)
以下是原文 Table 9 的结果：

<table>
<thead>
<tr>
<th>Datasets</th>
<th>GPT4-o</th>
<th>Gemini-2.0-Flash</th>
<th>Qwen2.5-VL 72B</th>
<th>Qwen3-Omni-30B-A3B -Instruct</th>
<th>Qwen3-Omni-Flash -Instruct</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="6">General Visual Question Answering</td>
</tr>
<tr>
<td>MMStar</td>
<td>64.7</td>
<td>71.4</td>
<td>70.8</td>
<td>68.5</td>
<td>69.3</td>
</tr>
<tr>
<td>HallusionBench</td>
<td>55.0</td>
<td>56.3</td>
<td>55.2</td>
<td>59.7</td>
<td>60.4</td>
</tr>
<tr>
<td>MM-MT-Bench</td>
<td>7.7</td>
<td>6.7</td>
<td>7.6</td>
<td>7.4</td>
<td>7.6</td>
</tr>
<tr>
<td colspan="6">Math &amp; STEM</td>
</tr>
<tr>
<td>MMMUval</td>
<td>69.1</td>
<td>71.3</td>
<td>70.2</td>
<td>69.1</td>
<td>69.8</td>
</tr>
<tr>
<td>MMMU-Pro overall</td>
<td>51.9</td>
<td>56.1</td>
<td>51.1</td>
<td>57.0</td>
<td>58.2</td>
</tr>
<tr>
<td>MathVista mini</td>
<td>63.8</td>
<td>71.4</td>
<td>74.8</td>
<td>75.9</td>
<td>77.4</td>
</tr>
<tr>
<td>MATH-Visionfull</td>
<td>30.4</td>
<td>48.6</td>
<td>38.1</td>
<td>56.3</td>
<td>57.3</td>
</tr>
<tr>
<td colspan="6">Documentation Understanding</td>
</tr>
<tr>
<td>AI2Dw.M.</td>
<td>84.6</td>
<td>86.7</td>
<td>88.7</td>
<td>85.2</td>
<td>86.4</td>
</tr>
<tr>
<td>ChartQA test Avg.</td>
<td>86.7</td>
<td>64.6</td>
<td>89.5</td>
<td>86.8</td>
<td>87.1</td>
</tr>
<tr>
<td colspan="6">Counting</td>
</tr>
<tr>
<td>CountBench</td>
<td>87.9</td>
<td>91.2</td>
<td>93.6</td>
<td>90.0</td>
<td>90.0</td>
</tr>
<tr>
<td colspan="6">Video Understanding</td>
</tr>
<tr>
<td>Video-MMEw/o sub</td>
<td>71.9</td>
<td>72.4</td>
<td>73.3</td>
<td>70.5</td>
<td>71.4</td>
</tr>
<tr>
<td>LVBench</td>
<td>30.8</td>
<td>57.9</td>
<td>47.3</td>
<td>50.2</td>
<td>51.1</td>
</tr>
<tr>
<td>MLVU</td>
<td>64.6</td>
<td>71.0</td>
<td>74.6</td>
<td>75.2</td>
<td>75.7</td>
</tr>
</tbody>
</table>

<p>Table 9: Vision $\rightarrow$ Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.</p>

*   **Instruct 模型**：Qwen3-Omni-Instruct 在 Math & STEM 相关任务（如 MMMU-Pro, MathVista mini, MATH-Visionfull）上取得了优于 GPT4-o 和 Gemini-2.0-Flash 等视觉语言模型的结果，并与 Qwen2.5-VL-72B 表现相当。这表明模型在图像理解和推理任务上具备卓越的能力。

    以下是原文 Table 10 的结果：

    <table>
    <thead>
    <tr>
    <th>Datasets</th>
    <th>Gemini-2.5-Flash -Thinking</th>
    <th>InternVL-3.5-241B-A28B</th>
    <th>Qwen3-Omni-30B-A3B -Thinking</th>
    <th>Qwen3-Omni-Flash -Thinking</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="5">General Visual Question Answering</td>
    </tr>
    <tr>
    <td>MMStar</td>
    <td>75.5</td>
    <td>77.9</td>
    <td>74.9</td>
    <td>75.5</td>
    </tr>
    <tr>
    <td>HallusionBench</td>
    <td>61.1</td>
    <td>57.3</td>
    <td>62.8</td>
    <td>63.4</td>
    </tr>
    <tr>
    <td>MM-MT-Bench</td>
    <td>7.8</td>
    <td>-</td>
    <td>8.0</td>
    <td>8.0</td>
    </tr>
    <tr>
    <td colspan="5">Math &amp; STEM</td>
    </tr>
    <tr>
    <td>MMMUval</td>
    <td>76.9</td>
    <td>77.7</td>
    <td>75.6</td>
    <td>75.0</td>
    </tr>
    <tr>
    <td>MMMU-Pro overall</td>
    <td>65.8</td>
    <td>-</td>
    <td>60.5</td>
    <td>60.8</td>
    </tr>
    <tr>
    <td>MathVista mini</td>
    <td>77.6</td>
    <td>82.7</td>
    <td>80.0</td>
    <td>81.2</td>
    </tr>
    <tr>
    <td>MATH-Visionfull</td>
    <td>62.3</td>
    <td>63.9</td>
    <td>62.9</td>
    <td>63.8</td>
    </tr>
    <tr>
    <td colspan="5">Documentation Understanding</td>
    </tr>
    <tr>
    <td>AI2Dw.M.</td>
    <td>88.6</td>
    <td>87.3</td>
    <td>86.1</td>
    <td>86.8</td>
    </tr>
    <tr>
    <td>ChartQA test Avg.</td>
    <td>-</td>
    <td>88.0</td>
    <td>89.5</td>
    <td>89.3</td>
    </tr>
    <tr>
    <td colspan="5">Counting</td>
    </tr>
    <tr>
    <td>CountBench</td>
    <td>88.6</td>
    <td>-</td>
    <td>88.6</td>
    <td>92.5</td>
    </tr>
    <tr>
    <td colspan="5">Video Understanding</td>
    </tr>
    <tr>
    <td>Video-MMEw/o sub</td>
    <td>79.6</td>
    <td>72.9</td>
    <td>69.7</td>
    <td>69.8</td>
    </tr>
    <tr>
    <td>LVBench</td>
    <td>64.5</td>
    <td>-</td>
    <td>49.0</td>
    <td>49.5</td>
    </tr>
    <tr>
    <td>MLVU</td>
    <td>82.1</td>
    <td>78.2</td>
    <td>72.9</td>
    <td>73.9</td>
    </tr>
    </tbody>
    </table>

<p>Table 10: Vision $\rightarrow$ Text performance of Qwen3-Omni-Thinking and other reasoning baselines. The highest scores are shown in bold.</p>

*   **Thinking 模型**：Qwen3-Omni-30B-A3B-Thinking 在 Math and STEM 基准上比 Instruct 版本提高了4.4分，并且在性能上与更大的基线模型持平。这突出了其在视觉推理方面的有效性和计算效率的平衡。
*   **长视频理解限制**：当前模型的长视频基准性能 suboptimal，这主要受限于位置外推能力 (positional extrapolation) 和上下文长度 (context length)。

### 6.1.4. 视听视频 $\rightarrow$ 文本 (AudioVisual Video $\rightarrow$ Text)
以下是原文 Table 11 的结果：

<table>
<thead>
<tr>
<th>Datasets</th>
<th>Previous Open-source SoTA</th>
<th>Gemini-2.5-Flash</th>
<th>Qwen2.5-Omni</th>
<th>Qwen3-Omni-30B-A3B -Instruct</th>
<th>Qwen3-Omni-Flash -Instruct</th>
</tr>
</thead>
<tbody>
<tr>
<td>WorldSense</td>
<td>47.1(Yang et al., 2025b)</td>
<td>50.9</td>
<td>45.4</td>
<td>54.0</td>
<td>54.1</td>
</tr>
</tbody>
</table>

<p>Table 11: AudioVisual $\rightarrow$ Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.</p>

*   **Instruct 模型**：Qwen3-Omni-Instruct 在 WorldSense 基准上取得了 SOTA 性能，显著超越了其他 Omni 模型。这证明了其在基础多模态集成方面的有效性。

    以下是原文 Table 12 的结果：

    <table>
    <thead>
    <tr>
    <th>Datasets</th>
    <th>Previous Open-source SoTA</th>
    <th>Gemini-2.5-Flash -Thinking</th>
    <th>Qwen3-Omni-30B-A3B -Thinking</th>
    <th>Qwen3-Omni-Flash -Thinking</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>DailyOmni</td>
    <td>69.8(Tang et al., 2025)</td>
    <td>72.7</td>
    <td>75.8</td>
    <td>76.2</td>
    </tr>
    <tr>
    <td>VideoHolmes</td>
    <td>55.6(Tang et al., 2025)</td>
    <td>49.5</td>
    <td>57.3</td>
    <td>57.3</td>
    </tr>
    </tbody>
    </table>

<p>Table 12: AudioVisual $\rightarrow$ Text performance of Qwen3-Omni-30B-A3B-Thinking and other reasoning baselines. The highest scores are shown in bold.</p>

*   **Thinking 模型**：Qwen3-Omni-Thinking 在 DailyOmni 和 VideoHolmes 等需要对视听信息进行复杂推理的基准上表现出增强的性能，这表明其在真实世界场景中的高级感知和推理潜力。

### 6.1.5. X $\rightarrow$ 语音 (X $\rightarrow$ Speech)

以下是原文 Table 13 的结果：

<table>
<thead>
<tr>
<th>Datasets</th>
<th>Model</th>
<th colspan="2">Performance</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="10">Content Consistency</td>
<td></td>
<td>SEED test-zh</td>
<td>test-en</td>
</tr>
<tr>
<td>Seed-TTSICL (Anastassiou et al., 2024)</td>
<td>1.11</td>
<td>2.24</td>
</tr>
<tr>
<td>Seed-TTSRL (Anastassiou et al., 2024)</td>
<td>1.00</td>
<td>1.94</td>
</tr>
<tr>
<td>MaskGCT (Wang et al., 2024c)</td>
<td>2.27</td>
<td>2.62</td>
</tr>
<tr>
<td>E2 TTS (Eskimez et al., 2024)</td>
<td>1.97</td>
<td>2.19</td>
</tr>
<tr>
<td>F5-TTS (Chen et al., 2024c)</td>
<td>1.56</td>
<td>1.83</td>
</tr>
<tr>
<td>Spark TTS (Wang et al., 2025b)</td>
<td>1.20</td>
<td>1.98</td>
</tr>
<tr>
<td>CosyVoice 2 (Du et al., 2024)</td>
<td>1.45</td>
<td>2.57</td>
</tr>
<tr>
<td>CosyVoice 3 (Du et al., 2025)</td>
<td>0.71</td>
<td>1.45</td>
</tr>
<tr>
<td>Qwen2.5-Omni-7B (Xu et al., 2025)</td>
<td>1.42</td>
<td>2.33</td>
</tr>
<tr>
<td>Qwen3-Omni-30B-A3B</td>
<td>1.07</td>
<td>1.39</td>
</tr>
</tbody>
</table>

<p>Table 13: Zero-Shot Speech Generation on Seed-TTS Test Set. The highest scores are shown in bold.</p>

*   **零样本语音生成**：Qwen3-Omni 在 SEED 测试集上展示了高度竞争性的性能。特别是经过强化学习优化后，在 `test-en` 集上取得了最佳性能，表明其预训练和持续预训练开发的语音理解和生成能力非常强大。

    以下是原文 Table 14 的结果：

    <table>
    <thead>
    <tr>
    <th>Language</th>
    <th colspan="3">Content Consistency</th>
    <th colspan="3">Speaker Similarity</th>
    </tr>
    <tr>
    <th></th>
    <th>Qwen3-Omni -30B-A3B</th>
    <th>MiniMax</th>
    <th>ElevenLabs</th>
    <th>Qwen3-Omni -30B-A3B</th>
    <th>MiniMax</th>
    <th>ElevenLabs</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Chinese</td>
    <td>0.716</td>
    <td>2.252</td>
    <td>16.026</td>
    <td>0.772</td>
    <td>0.780</td>
    <td>0.677</td>
    </tr>
    <tr>
    <td>English</td>
    <td>1.069</td>
    <td>2.164</td>
    <td>2.339</td>
    <td>0.773</td>
    <td>0.756</td>
    <td>0.613</td>
    </tr>
    <tr>
    <td>German</td>
    <td>0.777</td>
    <td>1.906</td>
    <td>0.572</td>
    <td>0.738</td>
    <td>0.733</td>
    <td>0.614</td>
    </tr>
    <tr>
    <td>Italian</td>
    <td>1.067</td>
    <td>1.543</td>
    <td>1.743</td>
    <td>0.742</td>
    <td>0.699</td>
    <td>0.579</td>
    </tr>
    <tr>
    <td>Portuguese</td>
    <td>1.872</td>
    <td>1.877</td>
    <td>1.331</td>
    <td>0.770</td>
    <td>0.805</td>
    <td>0.711</td>
    </tr>
    <tr>
    <td>Spanish</td>
    <td>1.765</td>
    <td>1.029</td>
    <td>1.084</td>
    <td>0.744</td>
    <td>0.762</td>
    <td>0.615</td>
    </tr>
    <tr>
    <td>Japanese</td>
    <td>3.631</td>
    <td>3.519</td>
    <td>10.646</td>
    <td>0.763</td>
    <td>0.776</td>
    <td>0.738</td>
    </tr>
    <tr>
    <td>Korean</td>
    <td>1.670</td>
    <td>1.747</td>
    <td>1.865</td>
    <td>0.778</td>
    <td>0.776</td>
    <td>0.700</td>
    </tr>
    <tr>
    <td>French</td>
    <td>2.505</td>
    <td>4.099</td>
    <td>5.216</td>
    <td>0.689</td>
    <td>0.628</td>
    <td>0.535</td>
    </tr>
    <tr>
    <td>Russian</td>
    <td>3.986</td>
    <td>4.281</td>
    <td>3.878</td>
    <td>0.759</td>
    <td>0.761</td>
    <td>0.676</td>
    </tr>
    </tbody>
    </table>

<p>Table 14: Multilingual Speech Generation on MiniMax Multilingual Test Set. The highest scores are shown in bold.</p>

*   **多语言语音生成**：Qwen3-Omni 在多种语言（如中文、英文、法文）上显著超越 MiniMax-Speech 和 ElevenLabs Multilingual v2，并在其他语言上取得竞争性结果。这表明模型在多语言语音克隆中具有高度的稳定性和类人语音质量。

    以下是原文 Table 15 的结果：

    <table>
    <thead>
    <tr>
    <th>Language</th>
    <th>Qwen3-Omni-30B-A3B</th>
    <th>CosyVoice3</th>
    <th>CosyVoice2</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>en-to-zh</td>
    <td>5.37</td>
    <td>5.09</td>
    <td>13.5</td>
    </tr>
    <tr>
    <td>ja-to-zh</td>
    <td>3.32</td>
    <td>3.05</td>
    <td>48.1</td>
    </tr>
    <tr>
    <td>ko-to-zh</td>
    <td>0.99</td>
    <td>1.06</td>
    <td>7.70</td>
    </tr>
    <tr>
    <td>zh-to-en</td>
    <td>2.76</td>
    <td>2.98</td>
    <td>6.47</td>
    </tr>
    <tr>
    <td>ja-to-en</td>
    <td>3.31</td>
    <td>4.20</td>
    <td>17.1</td>
    </tr>
    <tr>
    <td>ko-to-en</td>
    <td>3.34</td>
    <td>4.19</td>
    <td>11.2</td>
    </tr>
    <tr>
    <td>zh-to-ja</td>
    <td>8.29</td>
    <td>7.08</td>
    <td>13.1</td>
    </tr>
    <tr>
    <td>en-to-ja</td>
    <td>7.53</td>
    <td>6.80</td>
    <td>14.9</td>
    </tr>
    <tr>
    <td>ko-to-ja</td>
    <td>4.24</td>
    <td>3.93</td>
    <td>5.86</td>
    </tr>
    <tr>
    <td>zh-to-ko</td>
    <td>5.13</td>
    <td>14.4</td>
    <td>24.8</td>
    </tr>
    <tr>
    <td>en-to-ko</td>
    <td>4.96</td>
    <td>5.87</td>
    <td>21.9</td>
    </tr>
    <tr>
    <td>ja-to-ko</td>
    <td>6.23</td>
    <td>7.92</td>
    <td>21.5</td>
    </tr>
    </tbody>
    </table>

<p>Table 15: Cross-Linguial Speech Generation on CosyVoice3 Cross-Linguial Test Set. The highest scores are shown in bold.</p>

*   **跨语言语音生成**：Qwen3-Omni 在 `any-to-en` (任意语言到英语) 和 `any-to-ko` (任意语言到韩语) 语音克隆任务中超越了 CosyVoice3。即使在没有文本标准化的情况下，在 `any-to-ja` (任意语言到日语) 任务中也能与 CosyVoice3 表现相当，展示了其在不同语言背景下的出色适应性。

### 6.1.6. 非退化跨模态评估 (Non-Degradation Across Modalities)
以下是原文 Table 16 的结果：

<table>
<thead>
<tr>
<th></th>
<th>Datasets</th>
<th>Qwen3-30B-A3B -Base-202507</th>
<th>Qwen3-VL-30B-A3B -Base-202507</th>
<th>Qwen3-Omni-30B-A3B -Base-202507</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5">General Tasks</td>
<td>MMLU</td>
<td>81.24</td>
<td>-</td>
<td>81.69</td>
</tr>
<tr>
<td>MMLU-Redux</td>
<td>80.17</td>
<td>-</td>
<td>80.60</td>
</tr>
<tr>
<td>MMLU-Pro</td>
<td>61.81</td>
<td>-</td>
<td>61.57</td>
</tr>
<tr>
<td>SuperGPQA</td>
<td>38.24</td>
<td>-</td>
<td>40.14</td>
</tr>
<tr>
<td>BBH</td>
<td>83.79</td>
<td>-</td>
<td>83.53</td>
</tr>
<tr>
<td rowspan="2">Math &amp; STEAM Tasks</td>
<td>GSM8K</td>
<td>90.83</td>
<td>-</td>
<td>91.36</td>
</tr>
<tr>
<td>MATH</td>
<td>60.84</td>
<td>-</td>
<td>60.42</td>
</tr>
<tr>
<td rowspan="4">Coding Tasks</td>
<td>EvalPlus</td>
<td>69.70</td>
<td>-</td>
<td>73.96</td>
</tr>
<tr>
<td>MultiPL-E</td>
<td>65.75</td>
<td>-</td>
<td>64.79</td>
</tr>
<tr>
<td>MBPP</td>
<td>72.60</td>
<td>-</td>
<td>72.60</td>
</tr>
<tr>
<td>CRUX-O</td>
<td>66.94</td>
<td>-</td>
<td>69.06</td>
</tr>
<tr>
<td rowspan="2">Multilingual Tasks</td>
<td>MGSM</td>
<td>78.75</td>
<td>-</td>
<td>79.93</td>
</tr>
<tr>
<td>INCLUDE</td>
<td>65.17</td>
<td>-</td>
<td>64.73</td>
</tr>
<tr>
<td>College-level Problems</td>
<td>MMMUval</td>
<td>-</td>
<td>57.22</td>
<td>59.33</td>
</tr>
<tr>
<td rowspan="3">General Visual Question Answering</td>
<td>MMStar</td>
<td>-</td>
<td>67.2</td>
<td>69.6</td>
</tr>
<tr>
<td>RealWorldQAavg</td>
<td>-</td>
<td>73.98</td>
<td>71.89</td>
</tr>
<tr>
<td>OCR-related Tasks</td>
<td>AI2D</td>
<td>-</td>
<td>85.88</td>
<td>86.62</td>
</tr>
<tr>
<td rowspan="6"></td>
<td>TextVQAval</td>
<td>-</td>
<td>81.67</td>
<td>81.65</td>
</tr>
<tr>
<td>DocVQAtest</td>
<td>-</td>
<td>95.19</td>
<td>95.27</td>
</tr>
<tr>
<td>InfoVQAtest</td>
<td>-</td>
<td>81.17</td>
<td>83.31</td>
</tr>
<tr>
<td>ChartQAtest Avg</td>
<td>-</td>
<td>87.12</td>
<td>87.52</td>
</tr>
<tr>
<td>OCRBench</td>
<td>-</td>
<td>85.8</td>
<td>86.0</td>
</tr>
<tr>
<td rowspan="3">Video Understanding Tasks</td>
<td>Video-MMEw/o sub</td>
<td>-</td>
<td>69.22</td>
<td>69.25</td>
</tr>
<tr>
<td>MVBench</td>
<td>-</td>
<td>71.87</td>
<td>69.50</td>
</tr>
<tr>
<td>LVBench</td>
<td>-</td>
<td>48.61</td>
<td>51.07</td>
</tr>
</tbody>
</table>

<p>Table 16: We compare the performance of 30A3 models that are contemporaneous and identical in size in Qwen series. To ensure experimental rigor, all models were trained under the same schedule, using identical datasets for their respective modalities and exactly matched training compute (FLOPs).</p>

*   通过与同尺寸的文本专用 (Qwen3-30B-A3B-Base) 和视觉专用 (Qwen3-VL-30B-A3B-Base) 模型进行严格受控的比较，Qwen3-Omni-30B-A3B-Base 模型展示了：
    1.  **文本性能无损**：在 MMLU, MMLU-Redux, SuperGPQA, GSM8K, EvalPlus, MGSM 等文本基准上，Qwen3-Omni 相比文本专用模型 Qwen3-30B-A3B 表现持平或略有提升（例如 MMLU 81.69 vs 81.24, SuperGPQA 40.14 vs 38.24, GSM8K 91.36 vs 90.83, EvalPlus 73.96 vs 69.70）。
    2.  **视觉性能提升**：在 MMMUval, MMStar, AI2D, DocVQAtest, InfoVQAtest, ChartQAtest Avg, OCRBench 等视觉任务上，Qwen3-Omni 相比视觉专用模型 Qwen3-VL-30B-A3B 有明显提升（例如 MMMUval 59.33 vs 57.22, MMStar 69.6 vs 67.2, AI2D 86.62 vs 85.88, InfoVQAtest 83.31 vs 81.17）。甚至在长视频理解任务 LVBench 上也显示出提升 (51.07 vs 48.61)。
    3.  **多模态协同增强**：结果表明，在文本预训练早期混合单模态和跨模态数据，可以实现所有模态的更好性能。联合多模态训练实现了不同模态之间的相互增强。
    4.  **模态间影响观察**：
        *   早期多模态集成可以使语言模型在与视觉或音频协同训练时，语言能力不下降。
        *   文本模态的加入显著提高了视觉和音频性能。
        *   添加音频数据持续改进了 MMMU 基准和 OCR 相关任务的视觉性能。

## 6.2. 消融实验/参数分析
论文通过比较不同 Qwen3-Omni 变体的性能，间接展示了一些组件的有效性。

### 6.2.1. Thinking 模型的影响 (Table 7, 10, 12, 17, 18)
*   **积极影响**：
    *   在语音交互 (VoiceBench, Table 7) 和音频推理 (MMAU, MMSU, Table 7) 任务上，`Thinking` 模型（Qwen3-Omni-30B-A3B-Thinking / Flash-Thinking）通常比 `Instruct` 模型（Qwen3-Omni-30B-A3B-Instruct / Flash-Instruct）表现更好，尤其在 MMSU 上提升显著（例如 Qwen3-Omni-30B-A3B-Thinking MMSU 83.0 vs Instruct 68.1）。这表明 `Thinking` 模型通过显式推理，能有效增强多模态推理能力。
    *   在视觉 $\rightarrow$ 文本的 Math & STEM 任务 (Table 10) 上，`Thinking` 模型相比 `Instruct` 模型有4.4个点的提升，再次印证了其在复杂推理任务上的优势。
    *   在视听视频 $\rightarrow$ 文本的 DailyOmni 和 VideoHolmes (Table 12) 等需要复杂推理的任务上，`Thinking` 模型也表现出更好的性能。
*   **消极影响**：
    *   在 ASR/S2TT (Table 17) 和音乐理解 (Table 18) 这些以感知为主的任务中，`Thinking` 模型的表现反而不如 `Instruct` 模型。论文解释说，对于这些感知任务，复杂的推理过程未能带来性能提升，甚至可能引入更高的幻觉倾向。这暗示了 `Thinking` 模块并非适用于所有任务，其优势主要体现在需要深层推理的场景。

### 6.2.2. Flash 模型的影响 (Table 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18)
*   **Flash 模型**：论文中提及的 “Flash” 模型旨在提高计算效率和性能效率，并集成新功能（如支持方言）。
*   **性能趋势**：在许多基准测试中，`Qwen3-Omni-Flash-Instruct/Thinking` 模型与对应的非 Flash 版本 (`Qwen3-Omni-30B-A3B-Instruct/Thinking`) 性能非常接近，甚至略有提升。这表明在提高效率的同时，模型能够保持甚至改进性能。例如：
    *   Text $\rightarrow$ Text (Table 4 & 5)：Flash 版本与普通版本性能相近。
    *   Audio $\rightarrow$ Text ASR (Table 6)：Flash 版本与普通版本性能相近。
    *   VoiceBench (Table 7)：Flash-Thinking 略高于 30B-A3B-Thinking。
    *   Music understanding (Table 8)：Flash-Instruct 略高于 30B-A3B-Instruct。
    *   Vision $\rightarrow$ Text (Table 9 & 10)：Flash 版本与普通版本性能相近，在 CountBench 上甚至有显著提升（Flash-Thinking 92.5 vs 30B-A3B-Thinking 88.6）。
*   这说明 `Flash` 版本的优化是有效的，在保持高性能的同时，提高了效率，符合其设计目标。

### 6.2.3. MoE 架构和轻量级组件对延迟和吞吐量的影响 (Table 1, 2)
*   **MoE 架构**：Thinker 和 Talker 采用 MoE 架构，减少了 KV 缓存的 I/O 消耗，在高并发下仍能保持较高的词元生成率 (TPS)，例如 Thinker 在1并发时 75 TPS，6并发时 53 TPS；Talker 在1并发时 140 TPS，6并发时 110 TPS。
*   **轻量级 MTP 和 Code2Wav**：这两个模块的轻量化设计（MTP 80M 参数，Code2Wav 200M 参数）和对批处理推理的支持，显著降低了单次推理成本，从而降低了首包延迟。MTP 模块每词元仅需 14-18ms，Codec Decoder 每码本仅需 3-5ms。
*   **整体延迟**：这些优化使得端到端首包延迟在1并发冷启动设置下达到 234ms (音频) / 547ms (视频)，并且生成实时因子 (RTF) 即使在6并发下也能保持低于1（0.66），确保了实时流式响应。

### 6.2.4. 预训练阶段的对比 (Table 16)
*   对 Qwen3-Omni-30B-A3B-Base、Qwen3-30B-A3B-Base-202507 (文本专用) 和 Qwen3-VL-30B-A3B-Base-202507 (视觉专用) 在严格控制变量下的比较 (Table 16) 是一个重要的消融实验。
*   **结论**：
    1.  **早期多模态集成**：证明在文本预训练早期混合单模态和跨模态数据，语言模型可以在与视觉或音频共同训练时，不损失语言能力。
    2.  **文本模态的促进作用**：文本模态的引入显著改善了视觉和音频性能。
    3.  **音频对视觉的帮助**：经验性地发现，增加音频数据持续改善了 MMMU 基准和 OCR 相关任务的视觉性能。
*   这组实验有力地支持了论文的核心论点：多模态集成训练不仅可以避免性能下降，还能实现模态间的相互增强。

## 6.3. 定性结果 (Qwen3-Omni-30B-A3B-Captioner)
论文在 Appendix 9.2 中提供了 Qwen3-Omni-30B-A3B-Captioner 的定性分析案例，以展示其音频字幕生成能力。

<strong>案例一：表达性语音分析 (Analysis of Expressive Speech)</strong>
*   **音频描述**：一个在录音棚中进行的普通话男性独白，模仿道教人物太乙真人。声音清晰、充满活力、富有戏剧性，通过夸张的语气和语调展现了傲慢与自嘲的喜剧效果。录音干净，无背景噪音，语音居中，混响和均衡处理营造出戏剧性和空间感。
*   **模型表现**：生成的字幕详细描述了说话人的情绪、语气、口音（普通话）、声学环境（录音棚、轻微电子嘶嘶声、低频嗡嗡声）、语音特征（清晰、能量、戏剧性、中频丰富）、表达方式（夸张强调、升调、自嘲）、甚至推断出角色和内容（太乙真人，婴儿肥，逼人的帅气）。还分析了录音后期制作的特点（数字混响、突然剪辑）。
*   **结论**：模型能够捕捉语音的细微情感、风格和声学特征，并将其转化为高度详细和连贯的文本描述。

<strong>案例二：复杂场景音效分析 (Analysis of Complex Scene Sound Effect)</strong>
*   **音频描述**：一个25秒的高度制作的电影级音景，旨在唤起紧张和迫在眉睫的危险。包含深沉的音乐嗡嗡声、金属碰撞声、有节奏的节拍、管弦乐、合成音效、引擎轰鸣、高音金属尖叫、爆炸冲击、碎片声，以及一个人的喘息声。无言语，无环境提示，具有普遍的动作/科幻/惊悚电影风格。
*   **模型表现**：模型逐秒细致地描述了音景中的每一个元素，包括：音乐特征（无人机、管弦乐、合成纹理）、音效（金属撞击、引擎轰鸣、尖叫、爆炸、碎裂声）、声学环境特征（巨大的、硬墙空间、回响）、动态变化（音乐节奏、音量变化）、以及人物动作（喘息、织物摩擦）。它甚至推断出场景类型（工业设施、飞机库、隧道）和叙事含义（灾难、幸存者）。
*   **结论**：模型能够准确解析复杂的音景，识别多种音效，并将其整合为具有叙事感的详细描述，展现了其强大的音频事件检测和上下文理解能力。

<strong>案例三：混合语音、音频和音乐分析 (Analysis of Mixed Speech, Audio, and Music)</strong>
*   **音频描述**：一个包含语音、音效和音乐的混合音频。以金属撞击和低频隆隆声开始，伴随着机械嗡嗡声、电弧声、金属摩擦声。随后出现儿童女声提问“Are we there yet?”，接着是粗犷男声回应“We get there when we get there.”。机械声再次增强，并伴随能量涌动。另一男声呼唤“How you doing, honey?”，女声带着顽皮的恼怒回应“Do I have to answer?”。最后以合成音乐停顿结束。
*   **模型表现**：模型成功地分离并描述了多轨音频信息：
    *   **音效**：金属撞击、隆隆声、机械嗡嗡声、电弧声、金属尖叫、空气呼啸声。
    *   **环境**：巨大的金属环境（飞船机库、工业室）、长混响、高频嘶嘶声。
    *   **语音**：三个说话人的声音特征（远近、音高、语气、语调），对话内容，以及隐含的人际关系（家人、玩笑）。
    *   **音乐**：简短的合成音乐停顿。
*   **结论**：模型展现了在复杂多源音频场景中进行分离、识别和描述的能力，能够理解不同音源的交互和叙事含义，并推断出情境（科幻/奇幻背景下的家庭旅程）。

## 6.4. 局限性与未来工作
*   **长视频理解的局限性**：当前模型在长视频基准测试上的性能不如预期，这主要归因于两个架构限制：<strong>有限的位置外推能力 (limited capacity for positional extrapolation)</strong> 和 <strong>受限的上下文长度 (restricted context length)</strong>。解决这些限制是未来的关键工作目标。
*   **未来工作方向**：论文指出未来将沿着多个方向进一步推进模型发展，包括：
    *   <strong>多说话人 ASR (multi-speaker ASR)</strong>：当前模型可能在单说话人 ASR 上表现出色，但多说话人场景（如会议录音）可能仍是挑战。
    *   <strong>视频光学字符识别 (video OCR)</strong>：从视频流中识别和提取文本。
    *   <strong>视听主动学习 (audiovisual proactive learning)</strong>：模型能够主动从视听信息中学习和提问，而不仅仅是被动回答。
    *   <strong>增强对基于智能体的流式工作和函数调用的支持 (enhanced support for agent-based workflows and function calling)</strong>：进一步提高模型作为智能体进行复杂任务规划和执行的能力。

# 7. 总结与思考

## 7.1. 结论总结
Qwen3-Omni 代表了多模态人工智能领域的一个重要里程碑。它首次证明了构建一个能在文本、图像、音频和视频所有模态上都保持或超越单模态模型性能的单一、集成的多模态模型是可行的，且没有任何性能退化。这得益于其创新的 Thinker-Talker 混合专家 (MoE) 架构、从头训练的 AuT 音频编码器、优化的低延迟流式语音合成机制以及显式多模态推理的 Thinking 模型。Qwen3-Omni 在36个音频和视听基准测试中取得了显著的 SOTA 成果，并实现了234毫秒的超低端到端首包延迟。此外，通过微调，还推出了通用的音频字幕生成模型 Qwen3-Omni-30B-A3B-Captioner，填补了该领域空白。模型广泛支持119种文本语言、19种语音理解语言和10种语音生成语言，能够处理长达40分钟的音频输入。

论文强调，Qwen3-Omni 相比传统的级联管线 (cascaded pipelines) 具有多重优势，包括更强的跨模态推理能力、更低的端到端延迟以及更低的系统复杂性和成本。它为多模态研究领域树立了新的标杆，预示着集成、端到端的多模态训练不再需要牺牲核心语言能力或其他模态的性能。

## 7.2. 局限性与未来工作
论文明确指出了当前模型在**长视频理解**方面的局限性，主要原因在于**有限的位置外推能力**和**受限的上下文长度**。解决这些问题将是未来的研究重点。

此外，作者提出了一系列未来的工作方向，包括：
*   提升**多说话人 ASR** 能力。
*   开发<strong>视频光学字符识别 (video OCR)</strong>。
*   实现**视听主动学习**，使模型能够主动探索信息。
*   增强对<strong>基于智能体 (agent-based) 的工作流和函数调用</strong>的支持，以扩展模型的应用场景和复杂任务处理能力。

## 7.3. 个人启发与批判
### 7.3.1. 个人启发
1.  <strong>“无损集成”</strong>的可能性：Qwen3-Omni 最大的启发在于打破了“多模态模型必然伴随模态间性能权衡”的固有观念。通过精心的架构设计和训练策略（如早期预训练阶段的单模态与跨模态数据混合），实现了各模态的协同增强，而非此消彼长。这为未来构建真正强大的通用人工智能 (AGI) 模型指明了方向：并非是简单地堆叠模态，而是深度融合与协同优化。
2.  <strong>“Thinking-Talker”</strong>架构的潜能：这种感知与生成解耦的架构，不仅在技术上实现了低延迟和高并发，更在概念上契合了人类的认知过程——先思考，再表达。Thought/Reasoning 模型的引入，进一步强调了显式推理在复杂多模态任务中的重要性，这可能成为未来通用多模态模型的一个核心模块。
3.  **音频模态的崛起**：长期以来，视觉和文本是多模态研究的两大热门。Qwen3-Omni 在音频领域的卓越表现（尤其是开源 SOTA 的数量和超越闭源模型的表现），以及对音频字幕这一空白任务的填补，预示着音频模态在多模态大模型中正扮演越来越重要的角色，其潜力远未被完全挖掘。
4.  **低延迟的重要性**：234ms 的端到端首包延迟是一个令人印象深刻的成就。这对于语音助手、实时翻译、虚拟人交互等需要高实时性的应用至关重要。技术的最终价值在于服务于人类体验，而低延迟正是提升用户体验的关键要素。
5.  **工程与研究的结合**：论文中不仅有前沿的理论探索（MoE, TM-RoPE, Thinking Model），也有大量的工程优化（AuT 编码器、MTP 模块、Code2Wav ConvNet、分块预填充）。这表明在构建大型复杂模型时，严谨的学术研究与精湛的工程实践同等重要。

### 7.3.2. 批判
1.  **`Thinking` 模型定位的权衡**：尽管 `Thinking` 模型在推理任务上表现突出，但在 ASR/S2TT 等感知任务上性能反而下降，甚至可能引入幻觉。这表明 `Thinking` 模型的应用并非“万金油”，其启用需要根据任务类型进行判断，或者需要进一步优化以减少在非推理任务中的负面影响。如何智能地决定何时激活或调整 `Thinking` 模型的权重，是一个值得研究的问题。
2.  **长视频理解的挑战**：论文明确指出了长视频理解是当前模型的局限性。位置外推能力和上下文长度限制是许多 Transformer 模型的共性问题。虽然 TM-RoPE 尝试缓解，但对于超长视频，这仍然是瓶颈。未来的工作需要探索更高效、更具扩展性的长序列建模方法，例如更先进的稀疏注意力机制、状态空间模型 (SSMs) 或分层处理策略。
3.  **计算成本与资源可及性**：尽管 MoE 架构提高了推理效率，但模型的总参数量（Thinker 30B-A3B，Talker 3B-A0.3B）仍然非常庞大，预训练数据量更是达到万亿词元级别。这意味着训练和部署 Qwen3-Omni 及其变体需要巨大的计算资源。对于大多数研究机构和个人开发者而言，这仍然是一个难以企及的成本，限制了模型的广泛应用和进一步研究。虽然模型开源，但资源门槛仍然很高。
4.  <strong>“无损”</strong>的定义与泛化能力：论文强调了“无损”性能，但这主要基于特定的基准测试。在更广泛、更开放、更具挑战性的真实世界场景中，这种“无损”特性是否能完全泛化，以及是否存在未被当前基准测试捕捉到的潜在性能权衡，仍需进一步验证。
5.  **对“多模态交互”的深入探索**：目前的交互主要集中在文本和语音的输入输出。未来，模型如何更自然地处理多模态的输入组合（例如，通过手势、视线、身体语言等）、生成更丰富的多模态输出（例如，通过生成图像、视频片段，或控制机器人行为），是值得深入探讨的方向。目前的模型更多是多模态感知和单模态文本/语音生成，真正的“Omni-modal Interaction”还有很长的路要走。