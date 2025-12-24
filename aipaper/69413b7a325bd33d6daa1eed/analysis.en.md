# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models." This title indicates a focus on an improved, efficient, and real-time capable speech synthesis system that leverages advanced language models.

## 1.2. Authors
The paper lists numerous authors from Alibaba Group, China, including Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, Fan Yu, Huadai Liu, Zhengyan Sheng, Yue Gu, Chong Deng, Wen Wang, Shiliang Zhang, Zhijie Yan, and Jingren Zhou. The contact information provided is for `neo.dzh` and `sly.zsl` at `alibaba-inc.com`, suggesting their roles as primary contacts or senior researchers for the project. Their collective affiliation with Alibaba Group points to a strong background in industrial research and development, likely in artificial intelligence, speech technology, and large-scale systems.

## 1.3. Journal/Conference
The paper was published on arXiv, a preprint server, under the identifier `arxiv.org/abs/2412.10117v3`. As a preprint, it is currently undergoing peer review or is intended for future publication in a journal or conference. arXiv is a widely recognized platform for rapid dissemination of research findings in fields like AI and machine learning, allowing researchers to share their work before formal peer review processes are complete.

## 1.4. Publication Year
The UTC publication timestamp is `2024-12-13T12:59:39.000Z`, indicating the paper was published in **2024**.

## 1.5. Abstract
The abstract introduces CosyVoice 2 as an improved streaming speech synthesis model, building upon its predecessor, CosyVoice, which utilized supervised discrete speech tokens and progressive semantic decoding with language models (LMs) and Flow Matching for high prosody naturalness, content consistency, and speaker similarity in speech in-context learning. The motivation for CosyVoice 2 stems from the critical need for minimal response latency and real-time factor in interactive multi-modal large language model (LLM) applications.

CosyVoice 2 incorporates comprehensive and systematic optimizations:
*   **Finite-scalar quantization (FSQ):** Improves the codebook utilization of speech tokens.
*   **Streamlined text-speech LM:** Allows direct use of a pre-trained LLM as the backbone.
*   **Chunk-aware causal flow matching model:** Supports both streaming and non-streaming synthesis within a single model.

    By training on a large-scale multilingual dataset, CosyVoice 2 achieves human-parity naturalness, minimal response latency, and virtually lossless synthesis quality in streaming mode. Demos are available online for listening.

## 1.6. Original Source Link
*   **Original Source Link:** https://arxiv.org/abs/2412.10117v3
*   **PDF Link:** https://arxiv.org/pdf/2412.10117v3.pdf
*   **Publication Status:** This paper is a preprint published on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the **high latency and non-streaming nature of existing advanced text-to-speech (TTS) models**, which severely impacts the user experience in interactive applications, particularly with the rise of multi-modal large language models (LLMs) like GPT-4o. Current state-of-the-art zero-shot TTS models, while achieving high fidelity and naturalness, generally operate in an offline mode. This means they require the entire input text to be known and the full utterance to be synthesized before any audio output is produced. This "wait-and-then-speak" paradigm is unsuitable for real-time conversational AI, where immediate vocal responses are crucial for a natural interaction flow.

The problem is important because interactive voice experiences are becoming a cornerstone of modern human-computer interaction. Delays in speech synthesis can break immersion, lead to unnatural pauses, and hinder effective communication, making such applications feel clunky and unresponsive. Specific challenges or gaps exist in:
*   Developing streaming solutions for diffusion-based TTS models and hybrid systems, as streaming solutions have primarily been explored for language model-based TTS.
*   Maintaining high synthesis quality (prosody naturalness, content consistency, speaker similarity) when transitioning from offline to streaming modes.
*   Effectively leveraging the power of pre-trained LLMs for context understanding in speech synthesis.

    The paper's entry point or innovative idea is to systematically optimize the successful hybrid TTS architecture of its predecessor, CosyVoice, to achieve a unified framework for **both streaming and non-streaming synthesis** with minimal latency and high quality. It does this by making targeted architectural and methodological improvements that address the limitations of previous models in a holistic manner.

## 2.2. Main Contributions / Findings
The paper presents CosyVoice 2, which offers significant advancements in scalable streaming speech synthesis. Its primary contributions and key findings are:

1.  **Unified Streaming and Non-Streaming Synthesis Framework:** CosyVoice 2 proposes a novel unified `text-speech language model` and a `chunk-aware causal flow matching model`. This allows a single model to seamlessly handle both real-time streaming and high-quality offline synthesis scenarios, achieving virtually lossless performance in streaming mode compared to offline. This solves the problem of needing separate models or experiencing quality degradation for streaming applications.

2.  **Streamlined LM Architecture with LLM Backbone:** The model simplifies the `language model (LM)` architecture by removing the `text encoder` and `speaker embedding`. This strategic simplification enables the direct integration of powerful pre-trained textual `large language models (LLMs)` (specifically, `Qwen2.5-0.5B`) as the backbone. This enhances context understanding and leverages advancements in general-purpose LLMs for improved speech synthesis.

3.  **Improved Speech Tokenization with Finite Scalar Quantization (FSQ):** CosyVoice 2 replaces `vector quantization (VQ)` with `finite scalar quantization (FSQ)` in its `speech tokenizer`. FSQ significantly improves `codebook utilization` (achieving 100% utilization) and more effectively captures speech information, leading to better `content consistency` and overall synthesis quality.

4.  **Enhanced Instructed Generation Capacity:** The model integrates an `instructed dataset` during training, allowing it to support a wider range of control instructions. These include `natural language instructions` (e.g., emotion, speaking rate, dialect, role-playing) and `fine-grained instructions` (e.g., vocal bursts like `[laughter]`, emphasis tags $<strong>$). This provides users with a more versatile and intuitive way to generate vivid and expressive speech.

**Key conclusions and findings include:**
*   CosyVoice 2 achieves **human-parity naturalness** in synthesis quality.
*   It demonstrates **minimal response latency** crucial for interactive experiences.
*   The streaming mode offers **virtually lossless synthesis quality** compared to its offline counterpart, which is a major breakthrough for real-time applications.
*   The modular improvements, particularly the `LLM initialization`, `dropping speaker embedding`, and `FSQ`, significantly boost `content consistency` and maintain `speaker similarity`.
*   The `chunk-aware flow matching` design proves effective for streaming, even suggesting potential for `streaming non-autoregressive (NAR)` models.
*   The model exhibits strong `instructed generation` capabilities with high `MOS-I` scores, indicating accurate and natural instruction adherence.
*   `Reinforcement learning (RL)`, particularly `differentiable ASR rewards`, further improves `WER` and `generalization ability` in `multi-speaker fine-tuning (mSFT)` scenarios.

    These findings collectively address the core problem of high latency in advanced TTS, offering a scalable, high-quality, and interactive solution.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand CosyVoice 2, a beginner needs to grasp several core concepts in speech processing, machine learning, and generative models.

*   **Text-to-Speech (TTS):** The technology that converts written text into spoken audio. Traditionally, TTS systems have involved multiple stages, from text analysis and phonetic transcription to acoustic modeling and waveform generation.
*   **Neural Text-to-Speech (Neural TTS):** Modern TTS systems that use deep learning models to achieve more natural and human-like speech. They often learn complex mappings directly from text to audio, or intermediate representations, eliminating the need for hand-crafted linguistic rules.
*   **Zero-shot TTS:** A type of neural TTS model capable of synthesizing speech in the voice of an unseen speaker, based on a short audio reference of that speaker. This "zero-shot" capability means it can generalize to new speakers without explicit training data for them, making it highly versatile. It often involves `in-context learning (ICL)` where a reference speech provides the `timbre`, `prosody`, and `style`.
*   **Speech Codec:** A system that compresses and decompresses digital speech. In the context of neural TTS, a `speech codec` (e.g., `EnCodec`, `SoundStream`, `FunCodec`) can be used to convert raw audio waveforms into a sequence of discrete tokens (similar to how text is tokenized into words) and vice-versa. These discrete tokens, often called `speech tokens` or `acoustic tokens`, capture the essential information of speech in a compact, quantized form.
*   **Language Models (LMs) in TTS:** Deep learning models originally designed for natural language processing, now adapted for speech synthesis. They learn to predict the next token in a sequence based on previous tokens. In TTS, an LM might take text tokens as input and generate a sequence of speech tokens in an `autoregressive` manner (predicting one token at a time, based on its own previous predictions).
*   **Diffusion Models:** A class of generative models that learn to reverse a gradual diffusion process. They start with random noise and gradually transform it into a meaningful data sample (e.g., an image or a speech spectrogram) by learning to denoise at each step. They are known for generating high-quality and diverse samples.
*   **Flow Matching Models (FMs):** A newer class of generative models, similar in spirit to diffusion models, but often more efficient. Instead of learning to reverse a stochastic process, `Flow Matching` learns to transform a simple prior distribution (e.g., Gaussian noise) into a complex data distribution (e.g., speech spectrograms) by training a neural network to estimate the `vector field` of a continuous-time `ordinary differential equation (ODE)`. This `ODE` defines a `probability density path` between the two distributions. Once the vector field is learned, samples can be generated by solving the ODE from noise to data.
*   **Large Language Models (LLMs):** Very large neural networks pre-trained on massive amounts of text data, capable of understanding and generating human-like text. They excel at tasks like translation, summarization, question answering, and are increasingly used as foundational components in multimodal AI systems due to their strong `context understanding` capabilities.
*   **Vector Quantization (VQ):** A technique used in data compression, where continuous input vectors are mapped to a finite set of discrete vectors (a codebook). Each input vector is replaced by the closest vector in the codebook, identified by an index. While effective, VQ can suffer from `codebook underutilization`, where some codes in the codebook are rarely or never used.
*   **Finite Scalar Quantization (FSQ):** An alternative to VQ, used in CosyVoice 2. Instead of quantizing an entire vector, `FSQ` quantizes each dimension of a vector independently into a finite set of scalar values. These quantized scalar values are then combined to form a discrete code. The key advantage claimed by FSQ is better `codebook utilization` because all scalar values are used.
*   **Streaming vs. Non-streaming (Offline) Synthesis:**
    *   **Non-streaming (Offline):** The TTS system processes the entire input text and generates the complete audio utterance before any part of the audio is outputted. This introduces latency proportional to the length of the utterance.
    *   **Streaming:** The TTS system processes input text and generates audio in small chunks, delivering partial audio outputs as soon as they are ready, even while more text is still being received or processed. This significantly reduces `first-package latency` and enables real-time interaction.
*   **Transformer Architecture:** A neural network architecture based on the `self-attention` mechanism, which allows the model to weigh the importance of different parts of the input sequence when processing each element. Transformers are the backbone of many state-of-the-art LMs and are highly effective for sequence-to-sequence tasks.
*   **Mel Spectrogram:** A visual representation of the spectrum of frequencies of a sound as it varies with time. It's a common acoustic feature used in speech processing, representing the energy in different frequency bands (Mel scale) over short time windows. It captures the `acoustic details` of speech but doesn't contain phase information.
*   **Vocoder:** A component that converts acoustic features (like `Mel spectrograms`) back into a raw audio waveform. Since `Mel spectrograms` typically don't contain phase information, a `vocoder` is essential to reconstruct the full audio signal.

## 3.2. Previous Works
CosyVoice 2 builds upon a rich landscape of prior research in neural TTS, zero-shot TTS, and generative models. The paper references several categories and specific models:

*   **Traditional Neural TTS Models:** Early works like `Tacotron` [1], `Deep Voice` [3], and `FastSpeech` [5, 7] established the foundation for high-quality, end-to-end speech synthesis on predefined speakers. These models typically convert text to acoustic features (like Mel spectrograms) and then use a `vocoder` to generate waveforms.
*   **Zero-shot TTS Models:** The capability to synthesize speech for arbitrary speakers by imitating a reference audio is a key focus. These models often benefit from large-scale training data. The paper broadly categorizes them:
    *   **Codec Language Models:** These models first use a `speech codec` to extract discrete speech representations (e.g., `Soundstream` [9], `EnCodec` [10]). Then, an `autoregressive` [12, 13, 14, 15, 16, 17] or `masked language model` [18] predicts these speech tokens, which are finally converted back to waveforms by `codec vocoders` [19, 20]. Notable examples include `VALL-E` [8, 16], `VALL-T` [14], `RALL-E` [15], and `MaskGCT` [18]. These are good at `prosody consistency` and `diversity`.
    *   **Feature Diffusion Models:** Inspired by image generation, `denoising diffusion models` [22, 23] and `flow matching models` [24] are used for `non-autoregressive (NAR)` speech synthesis. Early models often required `duration prediction` [25, 26, 27, 28]. Newer approaches like `E2 TTS` [31], `F5-TTS` [32], and `Seed-TTS` [33] simplify text-speech alignment using padding or predicted durations. Examples include `Voicebox` [25], `NaturalSpeech` [26], `Voiceflow` [27], `Matcha-TTS` [28], `E3 TTS` [29], and `Ditto-TTS` [30]. These models can achieve `superior speech quality` as they are not constrained by codec vocoders.
    *   **Hybrid Systems:** These combine the strengths of `text-to-codec language models` and `codec-to-feature diffusion models` [34, 35, 33]. The LM handles text-speech alignment and duration, while the diffusion model synthesizes acoustic features from generated codecs. `CosyVoice` [34] itself is an example of such a hybrid system, as are `Seed-TTS` [33] and `FireRedTTS` [35]. These models aim for `high diversity`, `prosody consistency`, and `speech quality`.

*   **CosyVoice (Predecessor):** The paper explicitly states that CosyVoice 2 builds on its predecessor, `CosyVoice` [34]. The original CosyVoice was a `multilingual speech synthesis model` based on `supervised discrete speech tokens`. It utilized `progressive semantic decoding` with two generative models: `language models (LMs)` for semantic decoding and `Flow Matching` for acoustic generation. It achieved high `prosody naturalness`, `content consistency`, and `speaker similarity` in `in-context learning (ICL)`. However, it primarily operated in `non-streaming` mode, leading to high latency.

*   **Streaming TTS:** Prior work on streaming synthesis has mainly focused on `language model-based zero-shot TTS` [38, 39, 40, 41], exemplified by `LiveSpeech` [38, 39], `BASE TTS` [40], and approaches like `Speak While You Think` [41]. The challenge remained for diffusion-based and hybrid systems to achieve effective streaming.

## 3.3. Technological Evolution
The field of TTS has evolved from concatenative and statistical parametric methods to sophisticated neural networks. This evolution can be broadly summarized as:
1.  **Early TTS (Pre-Deep Learning):** Rule-based systems and statistical parametric methods. Speech quality was often mechanical.
2.  **Neural TTS (Early Deep Learning):** Models like `Tacotron` and `WaveNet` revolutionized quality, enabling end-to-end learning. These focused on synthesizing speech for specific, well-trained speakers.
3.  **Zero-Shot/Multi-Speaker TTS:** The ability to synthesize for arbitrary, unseen speakers emerged, often relying on `speaker embeddings` or `reference audio`.
4.  **Codec-based TTS:** The introduction of `speech codecs` (e.g., `VALL-E` series) allowed speech to be represented by discrete tokens, enabling the use of powerful `language models` for speech generation. This brought `LLM` capabilities into TTS.
5.  **Diffusion/Flow Matching TTS:** Inspired by image generation, these generative models provided another paradigm for high-fidelity speech synthesis, often excelling in `non-autoregressive` generation.
6.  **Hybrid Systems (like CosyVoice):** Combining the strengths of codec-based LMs and diffusion models for diverse and high-quality generation.
7.  **Streaming & Interactive TTS:** With the rise of `multimodal LLMs` and `voice AI assistants`, the focus shifted to reducing latency and enabling real-time, interactive speech. This required adapting existing high-quality models for streaming.

    CosyVoice 2's work fits into the **seventh stage** of this evolution, addressing the critical need for scalable, high-quality streaming capabilities within hybrid TTS systems, specifically leveraging advancements in `LLMs` and `Flow Matching` for this purpose.

## 3.4. Differentiation Analysis
Compared to the main methods in related work, CosyVoice 2 introduces several core differences and innovations:

*   **Unified Streaming and Non-Streaming Framework for Hybrid Systems:** While streaming solutions existed for `codec language models` [38, 39, 40, 41], CosyVoice 2 is one of the first to provide a comprehensive, `virtually lossless` streaming solution for a `hybrid system` that combines `language models` and `flow matching models`. This is achieved through a `unified text-speech LM` and a novel `chunk-aware causal flow matching model`. This overcomes the latency limitations of previous hybrid and diffusion-based models in real-time scenarios.
*   **Direct LLM Backbone Integration:** Unlike many `codec LMs` that build custom Transformer architectures, CosyVoice 2 streamlines its `text-speech LM` architecture to directly use a **pre-trained textual LLM** (`Qwen2.5-0.5B`) as its backbone. This simplifies the architecture, removes the need for separate `text encoders` and potentially problematic `speaker embeddings` in the LM, and directly leverages the powerful `context understanding` capabilities already present in large text models.
*   **Enhanced Codebook Utilization with FSQ:** The replacement of `vector quantization (VQ)` with `finite scalar quantization (FSQ)` in the `speech tokenizer` is a key innovation. FSQ demonstrably achieves 100% `codebook utilization`, ensuring that more speech information is preserved and captured effectively. This contrasts with VQ, which often suffers from `codebook underutilization`, leading to information loss.
*   **Advanced Instructed Generation:** CosyVoice 2 significantly expands its `instructed TTS capacity` to include more sophisticated `natural language` and `fine-grained controls` over `emotion`, `accent`, `role style`, and `vocal bursts`. This offers a more powerful and user-friendly interface for generating diverse and expressive speech compared to models with more limited control mechanisms.
*   **Decoupled Semantic and Acoustic Modeling:** While CosyVoice (the predecessor) also used this philosophy, CosyVoice 2 refines it by demonstrating that `speaker identity` can be effectively decoupled from the `semantic LM` and primarily handled by the `acoustic flow matching model`. This prevents `information leakage` and improves `cross-lingual capability` and `prosody naturalness` of the LM itself.

    In essence, CosyVoice 2 pushes the boundaries of hybrid TTS by not only achieving human-parity quality but also making it highly practical for interactive, real-time applications through its innovative streaming design, efficient tokenization, and direct LLM integration.

# 4. Methodology

CosyVoice 2 maintains the core design philosophy of its predecessor, CosyVoice, which separates the semantic and acoustic information of speech signals. The speech generation process is viewed as a gradual semantic decoding procedure. This involves a `text-speech language model (LM)` that decodes high-level text tokens into supervised semantic speech tokens, and a `Flow Matching model` that then converts these speech tokens into a `Mel spectrogram`, conditioned on speaker information. Finally, a `vocoder` transforms the `Mel spectrogram` into a waveform. The key innovations in CosyVoice 2 primarily focus on enabling streaming capabilities and improving efficiency and quality.

The overall architecture of CosyVoice 2 is depicted in Figure 1, showcasing the `supervised speech tokenizer`, the `unified text-speech language model`, and the `chunk-aware causal flow matching model`.

The following figure (Figure 1 from the original paper) shows an overview of CosyVoice 2.

![img-0.jpeg](images/1.jpeg)
*该图像是示意图，图1展示了CosyVoice 2的整体架构，包括(a)受监督的语音分词器，(b)支持流式和非流式合成的统一文本-语音语言模型，以及(c)条件于多种输入的块感知因果流匹配模型。图中涉及的公式为概率模型$P(Y|X)$。*

## 4.1. Text Tokenizer
CosyVoice 2 takes raw text directly as input. The text is tokenized using a `Byte Pair Encoding (BPE)`-based text tokenizer. This design choice eliminates the need for a separate `frontend model` that would typically convert graphemes to phonemes (`g2p`). By learning pronunciations end-to-end, the model can handle various contexts more flexibly.

A specific modification for tokenization is applied to Chinese text: if a BPE token encodes more than one Chinese character, it is `masked out`, and each character is encoded separately. This prevents excessively long pronunciations for a single token and mitigates issues arising from data sparsity for `multi-character tokens`. For other languages like English, Japanese, and Korean, this special handling is not applied.

## 4.2. Supervised Semantic Speech Tokenizer
This component is responsible for converting raw speech waveforms into discrete `speech tokens` that represent semantic information. As shown in Figure 1 (a), the core of this tokenizer involves integrating a `Finite Scalar Quantization (FSQ)` module into the `encoder` of the `SenseVoice-Large ASR model`.

The process works as follows:
1.  **Encoder_1:** The input speech waveform $X$ first passes through $Encoder_1$, which consists of six `Transformer blocks` equipped with `rotary positional embedding` [44]. This generates intermediate continuous representations $H$.
2.  **FSQ Module:** The intermediate representations $H$ are then fed into the FSQ module for quantization.
    *   **Projection and Quantization:** The FSQ module first projects the $D$-dimensional continuous representation $H$ into a low-rank space. The values of each dimension in this low-rank space are then quantized using a bounded round operation `ROUND`. This process is formally defined as:
        $$
        \bar {H} = \operatorname {R O U N D} \left(\operatorname {P r o j} _ {\text {d o w n}} (H)\right) \tag {1}
        $$
        where:
        *   $H$: The continuous intermediate representations from $Encoder_1$.
        *   $\operatorname {P r o j} _ {\text {d o w n}}$: A projection operation that maps the input $H$ to a lower-dimensional space.
        *   $\operatorname {R O U N D}$: A bounded rounding operation that quantizes the values in each dimension to be within a range of integers, typically $[-K, K]$.
        *   $\bar{H}$: The quantized low-rank representations.
    *   **Up-Projection:** After quantization, the low-rank representations $\bar{H}$ are projected back to the original dimension $\tilde{H}$ to be compatible with subsequent modules:
        $$
        \hat {H} = \operatorname {P r o j} _ {\mathrm {u p}} (\bar {H}) \tag {1}
        $$
        where:
        *   $\operatorname {P r o j} _ {\mathrm {u p}}$: A projection operation that maps the low-rank $\bar{H}$ back to the original (or a compatible higher) dimension.
        *   $\hat{H}$: The up-projected quantized representations.

3.  **ASR Decoder:** The $\hat{H}$ representations are then passed through the remaining `SenseVoice-Large modules`, including $Encoder_2$ and the `ASR Decoder`, to predict the posterior probabilities of the corresponding text tokens.

    **Training:** During training, `straight-through estimation` is used to approximate the gradients for the FSQ module and $Encoder_1$, allowing for end-to-end optimization despite the non-differentiable rounding operation.

**Speech Token Calculation:** A discrete `speech token` $\mu_i$ is obtained by treating the quantized low-rank representations $\bar{h}_i$ as digits in a `$(2K+1)$-ary system`. Each $\bar{h}_{i,j}$ represents the $j$-th dimension of the $i$-th quantized representation.
$$
\mu_{i}=\sum_{j=0}^{D-1}\bar{h}_{i,j}(2K+1)^{j} \tag {2}
$$
where:
*   $\mu_i$: The $i$-th discrete speech token.
*   $\bar{h}_{i,j}$: The $j$-th dimension of the $i$-th quantized low-rank representation.
*   $D$: The dimensionality of the low-rank space after `Proj_down`.
*   $K$: The bound for scalar quantization, meaning each dimension is quantized into $2K+1$ possible values (from `-K` to $K$).
*   $(2K+1)^{j}$: The base of the $(2K+1)$-ary system for the $j$-th dimension.

    The combination of $Encoder_1$, the `low-rank projector` of the FSQ module, the `bounded round operation`, and the `index calculation` forms the complete `speech tokenizer` for CosyVoice 2. This tokenizer operates at a rate of 25 Hz, meaning it produces 25 speech tokens per second of audio.

## 4.3. Unified Text-Speech Language Model
CosyVoice 2 utilizes a pre-trained textual `Large Language Model (LLM)`, specifically `Qwen2.5-0.5B` [45], as its `text-speech language model`. This LM is responsible for autoregressively generating `speech tokens` based on the input text, which serves as a prompt.

The following figure (Figure 2 from the original paper) shows a diagram of the unified text-speech language model for streaming and non-streaming synthesis in CosyVoice 2.

![img-1.jpeg](images/2.jpeg)

**Key Simplifications from CosyVoice:**
*   **Removal of Speaker Embedding:** Unlike its predecessor, CosyVoice 2 removes the `speaker embedding` from the `text-speech LM`. The authors found that `utterance-level speaker vectors` can contain not only `speaker identity` but also `language` and `paralanguage` information, which can negatively impact `prosody naturalness` and `cross-lingual capability`. This strategic removal helps the LM focus purely on semantic information.
*   **Abandonment of Text Encoder:** The `text encoder` component, present in the original CosyVoice, is also removed. The `Qwen2.5-0.5B` model is deemed powerful enough to directly align text and speech tokens without an additional dedicated text encoder.

**Unified Model for Streaming and Non-Streaming Synthesis:**
A major innovation is the unification of streaming and non-streaming synthesis within a single LM. The difference lies solely in how the input sequence is constructed during training and inference. The LM is trained using a `next-token-prediction` scheme.

*   **Non-Streaming Mode:**
    *   **Sequence Construction:** For offline processing, the sequence is constructed by concatenating tokens in a straightforward manner: a $start of sequence (<s>)$ token, all `text tokens`, a $turn of speech (<t>)$ token, all `speech tokens`, and an $end of sequence (</s>)$ token.
    *   **Loss Calculation:** During training, losses for certain tokens (indicated as "Ignore Token" in Figure 2) are disregarded when minimizing the `cross-entropy objective function`.

*   **Streaming Mode:**
    *   **Sequence Construction:** To enable streaming, text and speech tokens are interleaved in a pre-defined ratio of `N:M`. This means every $N$ text tokens are followed by $M$ speech tokens.
    *   **Filling Token:** If the LM is expected to predict a text token in the mixed sequence, it instead predicts a `filling token`. This `filling token` signals that the next $N$ text tokens should be concatenated at the inference stage.
    *   **End of Text:** Once all text tokens are consumed, the $turn of speech (<t>)$ token and any remaining `speech tokens` are appended sequentially, forming a `hybrid text-speech token sequence`.
    *   **Parameters:** In experiments, $N$ is set to 5, and $M$ is set to 15.

        By training the `text-speech LM` on both `non-streaming` and `streaming` sequence formats simultaneously, the model learns to operate in both modes.

**Inference Scenarios (ICL and SFT):**

*   **ICL (In-Context Learning), Non-Streaming:**
    *   Used when the LM needs to imitate `accent`, `prosody`, `emotion`, and `style` from `reference audio`.
    *   **Sequence:** $<s>$, `prompt_text`, `to_synthesize_text`, $</t>$, `prompt_speech`. The `prompt_speech` tokens are treated as fixed, pre-generated results.
    *   **Generation:** Autoregressive generation starts from this sequence until the $end of sequence (</s>)$ token is detected.

*   **ICL, Streaming:**
    *   Assumes the `to-generate text` is known, and speech tokens are generated in a streaming fashion.
    *   **Sequence:** $<s>$, `mixed_text_speech` (prompt and to-generate text interleaved with prompt speech at `N:M` ratio), $</t>$, `remaining_speech`.
    *   **Handling Text Length:** If the length of text exceeds `prompt speech tokens`, the LM generates `filling tokens`, and the next $N$ text tokens are manually padded.
    *   **Output:** Generation results are returned every $M$ tokens until $</s>$ is detected.

*   **SFT (Speaker Fine-tuning), Non-Streaming:**
    *   Used when the LM is `fine-tuned` for a specific speaker, so `prompt text` and `speech` are not needed.
    *   **Sequence:** $<s>$, `text`, $</t>$.
    *   **Generation:** The LM autoregressively generates speech tokens until $</s>$ is reached.

*   **SFT, Streaming:**
    *   **Sequence:** $<s>$, `first_N_text`.
    *   **Process:** The LM generates $M$ speech tokens, then the next $N$ text tokens are manually padded. This process repeats until all text tokens are exhausted, after which $</t>$ is added. This mode is particularly useful for `speech-to-speech multi-modal LLMs` to achieve extremely low latency.

## 4.4. Chunk-aware Flow Matching
The `Flow Matching model` is responsible for decoding the `semantic speech tokens` into a `Mel spectrogram`, incorporating `speaker embedding` and `reference speech` for `acoustic details` like `timbre`.

The following figure (Figure 3 from the original paper) shows a diagram of the unified chunk-aware flow matching model for streaming and non-streaming synthesis in CosyVoice 2.

![img-2.jpeg](images/3.jpeg)

**Acoustic Features and Token Preparation:**
*   **Mel Spectrogram:** The chosen acoustic feature is a `Mel spectrogram` with a `frame rate of 50 Hz` and a `sampling rate of 24000`.
*   **Upsampling:** Due to the `frame-rate mismatch` between `speech tokens` (25 Hz) and `Mel features` (50 Hz), the speech tokens are `up-sampled` by a factor of two.
*   **Look-ahead Convolution:** Before up-sampling, an additional `look-ahead convolution layer` is applied. This layer provides `future information` to the subsequent causal modules. It is implemented as a `right-padded 1-D convolution` with a `pad size` of $P$ and a `kernel size` of $P+1$.
*   **Alignment:** Several `chunk-aware causal Transformer blocks` follow to align the representation space of speech tokens with the acoustic features.

**Conditional Flow Matching (CFM) for Mel Spectrogram Sampling:**
A `Conditional Flow Matching (CFM)` model is used to sample the `Mel spectrogram` $X_1$, conditioned on `speech tokens` $\mu$, `reference speech` (from which $\tilde{X_1}$ is derived), and a `speaker embedding` $\mathbf{v}$.

*   **Probability Density Path:** The CFM model describes the distribution of the target `Mel spectrogram` $X_1$ (data distribution `q(X)`) by a `probability density path` from a `prior distribution` $p_0(X)$ (typically a simple distribution like standard Gaussian noise).
*   **Optimal-Transport (OT) Flow:** For sampling efficiency, the model learns the `vector field` $\omega_t$ of an `ordinary differential equation (ODE)` that defines the optimal transport path between $X_0$ (sampled from $p_0(X)$) and $X_1$ (sampled from `q(X)`).
    *   The vector field is given by:
        $$
        \omega_{t}(\phi_{t}^{OT}(X_{0},X_{1})|X_{1})=X_{1}-X_{0} \tag {3}
        $$
        where:
        *   $\omega_t$: The vector field at timestep $t$.
        *   $\phi_{t}^{OT}(X_{0},X_{1})$: The point on the `optimal transport path` at timestep $t$ between $X_0$ and $X_1$.
        *   $X_1$: The target `Mel spectrogram` (data).
        *   $X_0$: The initial point, typically sampled from a `prior distribution`.
    *   The path $\phi_{t}^{OT}$ is a linear interpolation:
        $$
        \phi_{t}^{OT}(X_{0},X_{1})=(1-t)X_{0}+tX_{1} \tag {4}
        $$
        where:
        *   $t$: A continuous timestep ranging from 0 to 1.
        *   $X_0$: A sample from the prior distribution $p_0(X)$.
        *   $X_1$: A sample from the data distribution `q(X)`.
    *   The distributions are defined as:
        $$
        X_{0} \sim p_{0}(X)=\mathcal{N}(0,I) \tag {5}
        $$
        $X_{1} \sim q(X) \tag {6}$
        where:
        *   $\mathcal{N}(0,I)$: A standard normal distribution (mean 0, identity covariance matrix).

*   **UNet for Vector Field Estimation:** A `causal convolutional Transformer UNet` is employed to learn to estimate this vector field $\omega_t$. It takes multiple conditions as input:
    $$
    \nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta)=\mathrm{UNet}_{\theta}\left(\phi_{t}^{OT}(X_{0},X_{1}),t;\mathbf{v},\{\mu\}_{1:L},\tilde{X_{1}}\right) \tag {7}
    $$
    where:
    *   $\nu_t(\cdot|\theta)$: The `UNet` model, parameterized by $\theta$, which estimates the vector field.
    *   $\phi_{t}^{OT}(X_{0},X_{1})$: The input state at time $t$ on the probabilistic path.
    *   $t$: The current timestep.
    *   $\mathbf{v}$: The `speaker embedding`.
    *   $\{\mu\}_{1:L}$: The sequence of `semantic speech tokens` of length $L$.
    *   $\tilde{X_1}$: The `masked Mel spectrogram`, derived from the reference speech.

*   **Training Objective:** The `UNet parameters` $\theta$ are optimized by minimizing the `L1 loss` between the predicted vector field and the ground-truth vector field:
    $$
    \theta=\arg\min_{\theta}\mathbb{E}_{p_{0}(X),q(X),t}\Big{|}\omega_{t}(\phi_{t}^{OT}(X_{0},X_{1}))-\nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta;\mu,\tilde{X}_{1},\mathbf{v})\Big{|}_{1} \tag {8}
    $$
    where the expectation is taken over samples $X_0 \sim p_0(X)$, $X_1 \sim q(X)$, and timestep $t$.
    *   **Masked Mel Spectrogram ($\tilde{X_1}$):** During training, $\tilde{X_1}$ is created by randomly masking out 70% to 100% of the final frames of the actual `Mel spectrogram` $X_1$. For inference, $\tilde{X_1}$ is the `Mel spectrogram` extracted from the `reference speech`.
    *   **Timestep Sampling:** At training time, $t$ is sampled from a `uniform distribution` `U[0,1]`.

*   **Inference Timestep Scheduling:** For sampling, a `cosine scheduler` is used, which allocates more steps to the initial generation phase:
    $$
    t:=1-\cos\left(\frac{1}{2}t\pi\right) \tag {9}
    $$

*   **Classifier-Free Guidance (CFG):** To enhance control and sample quality, the model is trained on both conditional and non-conditional situations, enabling `CFG` at inference.
    $$
    \tilde{\nu}_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta;\Psi)=(1+\beta)\cdot\nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta;\Psi)-\beta\cdot\nu_{t}(\phi_{t}^{OT}(X_{0},X_{1})|\theta) \tag {10}
    $$
    where:
    *   $\tilde{\nu}_t$: The guided vector field estimate.
    *   $\Psi$: Denotes the set of conditions, i.e., $\{\mathbf{v},\mu,\tilde{X_{1}}\}$.
    *   $\beta$: The `CFG strength` (set to 0.7 experimentally).
    *   $\nu_t(\cdot|\theta;\Psi)$: The conditional prediction of the vector field.
    *   $\nu_t(\cdot|\theta)$: The unconditional prediction of the vector field.
    *   `Number of Flow Estimations (NFE)` is set to 10.

**Chunk-aware Causal Design for Streaming:**
Traditional `flow matching models` operate in an offline mode, requiring all speech tokens before generating the `Mel spectrogram`. To adapt this for streaming, CosyVoice 2 reinterprets the `multi-step flow estimation` as a `stacked deeper neural network` (repeating the `UNet` ten times). By making this unfolded network `causal`, it can be applied to streaming synthesis. This is achieved using four types of `masks`:

1.  **Non-causal Mask:** Used for `offline mode`, allowing the model to attend to all frames of conditions (past and future). Offers the best performance in latency-insensitive scenarios.
2.  **Full-causal Mask:** Designed for `extremely low latency`, where only `past frames` can be attended.
3.  **Chunk-$M$ Mask:** A compromise between latency and performance. It allows leveraging information from `past frames` and $M$ `future frames`. Suitable for the first chunk of generation with low latency.
4.  **Chunk-`2M` Mask:** Provides performance approximating the `offline mode` by accepting more latency. Useful for subsequent `cascade generation chunks` to improve quality.

    During training, for each mini-batch, one mask is randomly sampled from these four types under a uniform distribution. This `chunk-aware training` allows a single `flow matching model` to be compatible with diverse scenarios, simplifying deployment. Moreover, masks with more context implicitly act as a teacher for those with less context, benefiting from an `implicit self-distillation scheme`.

## 4.5. Latency Analysis for Streaming Mode
The `first-package latency` is a crucial metric for streaming synthesis models, especially in `LLM-based voice chat` applications. This refers to the time it takes to produce the very first chunk of audio.

*   **TTS-specific Latency ($L_{TTS}$):** The latency within the TTS system itself, assuming text input is already available. It comprises the time taken by the `Language Model (LM)` to generate `speech tokens`, the `Flow Matching model (FM)` to generate `Mel spectrogram` frames, and the `vocoder` to synthesize waveforms for each `speech token` chunk.
    $$
    L_{TTS} = M \cdot d_{lm} + M \cdot d_{fm} + M \cdot d_{voc} \tag{11}
    $$
    where:
    *   $L_{TTS}$: The `first-package latency` originating from the TTS model.
    *   $M$: The number of `speech tokens` generated in one chunk (15 in experiments).
    *   $d_{lm}$: The computation time for the `LM` to generate one `speech token`.
    *   $d_{fm}$: The computation time for the `Flow Matching model` to generate `Mel spectrogram` frames corresponding to one `speech token`.
    *   $d_{voc}$: The computation time for the `vocoder` to synthesize waveforms for one `speech token`.

*   **Chatbot Latency ($L_{Chat}$):** In the context of an `LLM-based voice chat` system, the time taken for the `LLM` to generate the initial text response also contributes to the total `first-package latency`.
    $$
    L_{Chat} \leq N \cdot d_{llm} + L_{TTS} \tag{12}
    $$
    where:
    *   $L_{Chat}$: The `first-package latency` in a `LLM-based voice chat` scenario.
    *   $N$: The number of `text tokens` in the first chunk processed by the `LLM` (5 in experiments for the streaming LM's text-speech ratio).
    *   $d_{llm}$: The computation time for the `LLM` to generate one `text token`.

        The inequality arises because the `text tokenizer` in CosyVoice 2 masks out `multi-character tokens`, meaning its `text tokens` generally encode longer raw text segments than `text tokens` used by generic `text LLMs`. Therefore, $N \cdot d_{llm}$ provides an upper bound, and the actual latency might be lower if the initial $N$ text tokens from the `LLM` are very short.

## 4.6. Instructed Generation
To enhance controllability, CosyVoice 2 integrates an `instructed dataset` into its base training. This enables the model to generate speech based on explicit instructions provided by the user.

*   **Instruction Types:**
    *   **Natural Language Instructions:** These are descriptions prepended to the input text, followed by a special $“<endofprompt>”$ token. They cover aspects like:
        *   `Emotion`: e.g., "高兴(Happy)", "悲伤(Sad)", "惊讶(Surprised)".
        *   `Speaking Rate`: e.g., "快速(Fast)", "慢速(Slow)".
        *   `Dialect`: e.g., "粤语" (Cantonese), "四川话" (Sichuanese).
        *   `Role-playing`: e.g., "神秘(Mysterious)", "机器人(Robot)".
    *   **Fine-grained Instructions:** These involve inserting `vocal bursts` (e.g., `[laughter]`, `[breath]`) between text tokens or applying `vocal feature tags` to phrases (e.g., $<strong>XXX</strong>$ for emphasis, $<laughter>XXX</laughter>$ for speaking with laughter).

        The following are the results from Table 1 of the original paper:

        | Natural Language Instruction |
        | :--------------------------- |
        | Emotion: 高兴(Happy), 悲伤(Sad), 惊讶(Surprised), 愤怒(Angry), 恐惧(Fearful), 厌恶(Disgusted), 冷静(Calm), 严肃(Serious) |
        | Speaking Rate: 快速(Fast), 非常快速(Very Fast), 慢速(Slow), 非常慢速(Very Slow) |
        | Dialect: 粤语, 四川话, 上海话, 郑州话, 长沙话, 天津话 |
        | Role-playing: 神秘(Mysterious), 凶猛(Fierce), 好奇(Curious), 优雅(Elegant), 孤独(Lonely), 机器人(Robot), 小猪佩奇(Peppa), etc. |
        | Fine-grained Instruction |
        | Vocal Bursts: [laughter], [breath], etc. |
        | Vocal Features: <laughter></laughter>, **\<\strong\>** |
        | Examples |
        | - 你能用高兴的情感说吗？ < endofprompt > 今天真是太开心了，马上要放假了！ I'm so happy, Spring Festival is coming! |
        | - Please speaking very fast. < endofprompt > Today is a happy day, full of laughter and joy. |
        | - 请问你能模仿粤语的口音吗？ < endofprompt > 多保重，早休息。 |
        | - 尝试一下以机器人的角色和我交流。 < endofprompt > 接收知识光波！ |
        | - [laughter] 有时候，看着小孩子们的天真行为[laughter]，我们总会会心一笑。 |
        | - She pursued her dreams with **enthusiasm** |
        | and **grit** |

Table 10 provides examples of these instruction types. This integrated approach allows for more versatile and vivid speech generation within a single model, combining both `instructed` and `zero-shot` capabilities.

## 4.7. Multi-Speaker Fine-tuning
The paper introduces `multi-speaker fine-tuning (mSFT)` as a method to further improve generation quality and `speaker similarity` for specific speakers. Instead of fine-tuning the pre-trained model on a single speaker, `mSFT` involves simultaneous fine-tuning on multiple speakers.

*   **Benefits:** This approach ensures comprehensive `prosody` and `pronunciation coverage` across multiple speakers and helps mitigate `catastrophic forgetting` of the knowledge learned from the pre-trained models.
*   **Speaker-Prompt Tags:** To prevent `timbre confusion` between different speakers, `speaker-prompt tags` are prepended to the input text. For a specific speaker, the tag would be "Speaker A$<\!endofprompt\>$". If a training sample is not associated with a labeled speaker, a special tag "unknown$<\!endofprompt\>$" is used.
*   **Learning Rate:** The learning rate for `mSFT` is set to `1e-5`.

## 4.8. Reinforcement Learning for SFT
`Reinforcement learning (RL)` is employed to align the `LM` outputs with human preference, specifically to improve `speaker similarity (SS)` and `pronunciation accuracy` (measured by `recognition word error rate (WER)`) during the `fine-tuning stage`.

*   **Reward Function:** `SS` and `WER` are used to define a reward for `preferred samples` $x^w$ and `rejected samples` $x^l$.
*   **Direct Preference Optimization (DPO):** The `TTS system` is optimized using `Direct Preference Optimization (DPO)` [49].
    $$
    L_{DPO}(\pi_{\theta};\pi_{\text{ref}})=-\log\sigma\left(\beta\log\frac{\pi_{\theta}(\mu^{w}|y)}{\pi_{\text{ref}}(\mu^{w}|y)}-\beta\log\frac{\pi_{\theta}(\mu^{l}|y)}{\pi_{\text{ref}}(\mu^{l}|y)}\right) \tag{13}
    $$
    where:
    *   $L_{DPO}$: The `DPO loss` function.
    *   $\pi_{\theta}$: The policy (model under training) generating the speech tokens.
    *   $\pi_{\text{ref}}$: The reference policy, usually a frozen or older version of $\pi_{\theta}$.
    *   $\mu^w$: The sequence of `speech tokens` extracted from the `preferred sample` $x^w$.
    *   $\mu^l$: The sequence of `speech tokens` extracted from the `rejected sample` $x^l$.
    *   $y$: The input text.
    *   $\sigma$: The sigmoid function, which maps values to a range between 0 and 1.
    *   $\beta$: A hyperparameter controlling the strength of the preference.

        **Limitation of DPO:** This method is computationally expensive because it requires repeatedly synthesizing audios through the `TTS system` to obtain `distinguishable preferred and rejected samples`. This means four forward passes are needed for one training step.

*   **Differentiable ASR Rewards:** To simplify the process and reduce computational cost, an alternative method uses a `differentiable ASR reward`.
    1.  **Token Recovery:** The `LM`-predicted `speech token` $\mu_i$ is first recovered into its `quantized low-rank representations` $\bar{H}$:
        $$
        \bar{h}_{i,j}=\left\lfloor\frac{\mu_{i}}{(2K+1)^{j}}\right\rfloor\mod(2K+1) \tag{14}
        $$
        where:
        *   $\bar{h}_{i,j}$: The $j$-th dimension of the $i$-th recovered `quantized low-rank representation`.
        *   $\mu_i$: The $i$-th predicted `speech token` from the `LM`.
        *   $K$: The quantization bound for `FSQ` (from Section 2.2).
        *   $j$: The index of the dimension, from `0` to `D-1`.
    2.  **Up-Projection:** These recovered representations $\bar{H}$ are then up-projected:
        $$
        \hat{H}=\text{Proj}_{up}(\bar{H}) \tag{15}
        $$
    3.  **ASR Prediction:** The `ASR backend` of the `speech tokenizer` (whose parameters $\theta_{ASR}$ are frozen during this stage) then uses $\hat{H}$ to re-predict the input text $Y$. The negative log posterior of this prediction is used as the `ASR reward function`.
        $$
        L_{ASR}=-\log P(Y|\hat{H};\theta_{ASR})
        $$
        where:
        *   $L_{ASR}$: The `ASR loss` serving as a reward.
        *   $P(Y|\hat{H};\theta_{ASR})$: The probability of the input text $Y$ given the recovered speech representations $\hat{H}$, as predicted by the `ASR backend` with frozen parameters $\theta_{ASR}$.
        *   $Y$: The input text.
        *   $\hat{H}$: The recovered and up-projected speech representations.

    *   **Gumbel Softmax:** Since the `sampling operation` for $\mu_i \sim P(\mu_i|\mu_{1:i-1},Y;\theta_{LM})$ is typically non-differentiable, `Gumbel softmax sampling` is used to make it differentiable, allowing for direct optimization of the `LM parameters` $\theta_{LM}$ using the $L_{ASR}$ objective. This approach avoids repeated audio synthesis and is more computationally efficient.

# 5. Experimental Setup

## 5.1. Datasets

### 5.1.1. Training Data for Speech Tokenizer
The `speech tokenizer` was trained on a substantial dataset totaling 200,000 hours of speech, accompanied by normalized transcriptions.

The following are the results from Table 2 of the original paper:

| Language | Duration (hours) |
| :------- | :--------------- |
| Chinese  | 110,884          |
| English  | 99,918           |

**Table 2: Details of training data for speech tokenizer.**

*   **Source:** The data was compiled from three different resources: open-source `ASR datasets`, internal industrial datasets, and `TTS generation datasets`.
*   **Languages:** Primarily Chinese (110,884 hours) and English (99,918 hours).
*   **Zero-shot Capability:** Despite only using Chinese and English for training, subsequent experiments revealed that the `speech tokenizer` exhibited `zero-shot capability` for other languages, including Japanese and Korean. This indicates its ability to generalize to unseen language acoustics.

### 5.1.2. Training Data for CosyVoice 2
CosyVoice 2 shares the same training data as its predecessor. The data collection and processing involved several steps:
1.  **Speech-only Data Collection:** Initial collection of speech audio using internal processing tools.
2.  **Pseudo Text Label Generation:**
    *   `Paraformer` [50] was used to generate pseudo text labels for Chinese speech.
    *   `SenseVoice` [43] was used to generate pseudo text labels for other languages.
3.  **Data Filtering and Punctuation Enhancement:** An internal `force-alignment model` was employed to filter out low-quality data and improve the accuracy of punctuation in the pseudo labels.

    The following are the results from Table 3 of the original paper:

    | Language | Duration (hours) |
    | :------- | :--------------- |
    | Chinese  | 130,000          |
    | English  | 30,000           |
    | Japanese | 4,600            |
    | Korean   | 2,200            |

**Table 3: Details of training data for CosyVoice 2.**

*   **Total Duration:** Approximately 166,800 hours.
*   **Languages:** Chinese (130,000 hours), English (30,000 hours), Japanese (4,600 hours), and Korean (2,200 hours).
*   **Characteristics:** This large-scale multilingual dataset is crucial for training a versatile zero-shot TTS model.

## 5.2. Evaluation Metrics
For every evaluation metric mentioned in the paper, a complete explanation is provided below.

### 5.2.1. Word Error Rate (WER) / Character Error Rate (CER)
*   **Conceptual Definition:** `Word Error Rate (WER)` and `Character Error Rate (CER)` are standard metrics used to assess the `content consistency` or `intelligibility` of synthesized speech. They measure the discrepancy between the transcribed text (ground truth) and the text recognized by an `Automatic Speech Recognition (ASR)` system when processing the synthesized audio. A lower WER/CER indicates better content consistency, meaning the synthesized speech is more accurately recognized as the intended text. WER is typically used for English, while CER is often preferred for character-based languages like Chinese, Japanese, and Korean.
*   **Mathematical Formula:**
    $$
    \text{WER} = \frac{S + D + I}{N} \quad \text{or} \quad \text{CER} = \frac{S + D + I}{N}
    $$
*   **Symbol Explanation:**
    *   $S$: The number of `substitutions` (incorrectly recognized words/characters).
    *   $D$: The number of `deletions` (words/characters in the reference that were missed by the ASR).
    *   $I$: The number of `insertions` (words/characters recognized by the ASR that were not in the reference).
    *   $N$: The total number of words/characters in the reference (ground truth) transcription.

### 5.2.2. Speaker Similarity (SS)
*   **Conceptual Definition:** `Speaker Similarity (SS)` quantifies how well the synthesized speech imitates the voice characteristics (timbre, vocal identity) of a target speaker, often provided via a `reference audio`. It is usually measured by comparing `speaker embeddings` extracted from the reference speech and the synthesized speech. A higher SS value indicates that the synthesized speech sounds more like the target speaker.
*   **Mathematical Formula:** `Speaker similarity` is commonly measured using the `cosine similarity` between two speaker embeddings.
    $$
    \text{Cosine Similarity}(\mathbf{v}_1, \mathbf{v}_2) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|} = \frac{\sum_{i=1}^n v_{1i} v_{2i}}{\sqrt{\sum_{i=1}^n v_{1i}^2}\sqrt{\sum_{i=1}^n v_{2i}^2}}
    $$
*   **Symbol Explanation:**
    *   $\mathbf{v}_1$: The `speaker embedding` vector extracted from the reference speech.
    *   $\mathbf{v}_2$: The `speaker embedding` vector extracted from the synthesized speech.
    *   $\cdot$: The `dot product` operator.
    *   $\|\cdot\|$: The `Euclidean norm` (or magnitude) of a vector.
    *   $v_{1i}, v_{2i}$: The $i$-th component of vectors $\mathbf{v}_1$ and $\mathbf{v}_2$, respectively.
    *   $n$: The dimensionality of the `speaker embedding` vectors.

### 5.2.3. Non-intrusive Mean Opinion Score (NMOS)
*   **Conceptual Definition:** `Non-intrusive Mean Opinion Score (NMOS)` is an objective metric used to evaluate the `perceptual quality` of speech. Unlike traditional `MOS` scores which often require both a reference and degraded signal, `NMOS` attempts to predict human `Mean Opinion Score (MOS)` ratings without needing a clean reference signal. It's particularly useful for evaluating speech in scenarios where a pristine reference is unavailable, such as synthesized speech. NMOS typically ranges from 1 to 5, where 5 represents excellent quality and 1 represents poor quality. A higher NMOS indicates better overall speech quality and naturalness.
*   **Mathematical Formula:** The paper refers to `DNSMOS P.835` [53], which is a neural network-based predictor trained to estimate human MOS scores. There isn't a simple single mathematical formula for its calculation, as it's the output of a complex deep learning model. The model takes a speech signal as input and predicts a MOS score for overall listening quality, noise, and reverberation.
*   **Symbol Explanation:** N/A for a predictive model output like this, beyond describing the output score itself.

### 5.2.4. Mean Opinion Score for Instruction (MOS-I)
*   **Conceptual Definition:** `MOS-I` is a subjective human evaluation metric specifically designed to assess the `accuracy` and `naturalness` of `instructed generation` in TTS models. Human evaluators listen to synthesized speech and rate how well it adheres to specific instructions (e.g., emotion, speaking rate, dialect, fine-grained controls) and whether the instruction application sounds natural. A higher `MOS-I` score (typically on a 1-5 scale) indicates better instruction adherence and naturalness.
*   **Mathematical Formula:** `MOS-I` is an average of subjective ratings, similar to standard MOS.
    $$
    \text{MOS-I} = \frac{1}{K} \sum_{k=1}^K R_k
    $$
*   **Symbol Explanation:**
    *   $\text{MOS-I}$: The `Mean Opinion Score for Instruction`.
    *   $K$: The total number of human evaluators.
    *   $R_k$: The rating given by the $k$-th evaluator for a specific audio sample's instruction adherence and naturalness. Ratings are typically on a scale (e.g., 1 to 5).

## 5.3. Evaluation Settings

### 5.3.1. Test Sets
The models are evaluated on several test sets to cover different languages, domains, and levels of difficulty:

*   **Librispeech test-clean:** This is a subset of the `Librispeech corpus` [51], specifically `test-clean`, used to evaluate `CosyVoice 2` on a limited English domain.
*   **SEED test sets:** Widely used for evaluating recent TTS models [33], covering diverse text domains and reference speeches.
    *   `test-zh`: Approximately 2,000 Chinese samples from `CommonVoice datasets`.
    *   `test-en`: Approximately 1,000 English samples from `CommonVoice datasets`.
    *   `test-hard`: Around 400 challenging test cases designed to assess the robustness of `TTS models` against `text repetition`, `tongue twisters`, and other difficult synthesis scenarios.

### 5.3.2. ASR Models for Content Consistency
*   `Whisper-large V3` [54]: Used as the `ASR model` for English evaluations (e.g., `Librispeech test-clean`, `test-en`, Japanese, Korean).
*   `Paraformer` [50]: Used for Chinese evaluations (`test-zh`, `test-hard`). Punctuation is excluded before `WER`/`CER` calculation.

### 5.3.3. Speaker Similarity (SS) Models
*   `ERes2Net` [52]: Employed to extract `speaker embeddings` for both prompt and generated utterances, with their `raw cosine similarity` serving as the `speaker similarity` score.
*   `WavLM-finetuned SV model`: Also used for `speaker similarity` evaluation on SEED test sets, providing an alternative perspective. The paper notes inconsistencies between `ERes2Net` and `WavLM` results, suggesting a need for more research in this area. `ERes2Net` is adopted for subsequent experiments.

### 5.3.4. Objective Quality
*   `NMOS score` [53]: Used to evaluate the `objective quality` of the synthesized speech.

## 5.4. Benchmark for Japanese and Korean
Specific benchmarks were constructed to evaluate the model's performance on Japanese and Korean.

*   **test-ja (Japanese):**
    *   Consists of 1,000 samples extracted from the `CommonVoice dataset`.
    *   Creation: The entire `CommonVoice JA-test` set was randomly shuffled and paired as `reference utterances` and `target utterances`. From these, 1,000 pairs with text lengths ranging from 8 to 32 characters were selected to form the final test set.
*   **test-ko (Korean):**
    *   `Reference utterances`: 1,000 speech samples from the `CommonVoice dataset` were selected based on `Whisper-Large V3` [54] recognition, requiring a `WER` less than 5% and no deletion or insertion errors.
    *   `Input text`: 1,000 text samples were randomly selected from the remaining data.
*   **Public Release:** The lists of prompt speeches, prompt transcriptions, and input text for these two test sets are released to facilitate reproduction and establish a benchmark for these languages.
*   **ASR Model:** `Whisper-large V3` is used for both Japanese and Korean evaluations.

## 5.5. Baselines
The paper compares CosyVoice 2 against a comprehensive set of state-of-the-art `zero-shot TTS` models, both open-source and commercial.

*   **Open-source Models (General):**
    *   `ChatTTS` [56]: A popular open-source `codec language model`.
    *   `GPT-SoVITs` [57]: Another prominent open-source `codec language model`.
    *   `OpenVoice` [58]: Known for its versatile instant voice cloning capabilities.
    *   `ParlerTTS` [59]: Uses natural language guidance for TTS.
    *   `EmotiVoice` [60]: Focuses on emotional speech synthesis.
    *   `FireRedTTS` [35]: A foundation TTS framework.
    *   `MaskGCT` [18]: A `masked generative codec Transformer`.
*   **Predecessor:** `CosyVoice` [34]: The previous version of the model, a hybrid system.
*   **Other Advanced Models (often closed-source or research-focused):**
    *   `Seed-TTS` [33]: A family of high-quality versatile speech generation models, a hybrid system. Denoted with `†` as a closed-source model.
    *   `E2 TTS` [31]: `Embarrassingly easy fully non-autoregressive zero-shot TTS`. Denoted with `†` as a closed-source model.
    *   `F5-TTS` [32]: A `Flow Matching` based model.

        These baselines represent the current landscape of high-quality, zero-shot, and sometimes streaming-capable TTS models, ensuring a robust comparison for CosyVoice 2's performance across various metrics.

# 6. Results & Analysis

## 6.1. Evaluations on Speech Tokenizer
The paper first evaluates its `supervised speech tokenizer` from multiple perspectives: `codebook utilization`, `ASR error rate`, `token visualization`, and `speaker identification (SID) training`.

The following are the results from Table 4 of the original paper:

| Method | Codebook | | ASR Error Rate (%) | | | |
| :----- | :------- | :- | :----------------- | :- | :- | :- |
| | Size | Util. | C.V. EN | C.V. CN | Fluers EN | Fluers CN |
| VQ | 4,096 | 963 (23%) | 18.26 | 11.56 | 7.65 | 5.03 |
| FSQ | 6,561 | 6,561 (100%) | 10.67 | 7.29 | 6.58 | 4.43 |

**Table 4: The comparison of VQ and FSQ inside Sensevoice-large encoder. C.V. stands for the CommonVoice benchmarks.**

*   **Codebook Utilization & ASR Error Rate (Table 4):**
    *   The table compares `Vector Quantization (VQ)` and `Finite Scalar Quantization (FSQ)` within the `SenseVoice-Large encoder`.
    *   `VQ` (with a codebook size of 4,096) only utilized 963 codes (23%), indicating significant underutilization. This suggests that a large portion of the learned discrete representations were not being used, potentially leading to information loss. Its ASR error rates (e.g., 18.26% on `C.V. EN`) are relatively high.
    *   `FSQ` (with a larger effective codebook size of 6,561) achieves **100% codebook utilization**. This means all possible discrete codes are actively used, indicating more effective information capture. Correspondingly, `FSQ` significantly reduces ASR error rates across all benchmarks (e.g., 10.67% on `C.V. EN`, a relative improvement of $(18.26-10.67)/18.26 ≈ 41.5%$).
    *   **Analysis:** This demonstrates that `FSQ` is highly effective in making full use of the quantization space, preserving more semantic information vital for ASR and, by extension, for TTS content consistency.

*   **Token Visualization (Figure 4):**
    The paper uses `t-SNE visualization` to analyze the characteristics of `FSQ` with respect to `speaker independence`.
    The following figure (Figure 4 from the original paper) shows the t-SNE visualization of speech representations before (a) and after (b) the quantization for three different speakers in Voxceb1 dataset. (c) shows the codebook utilization in terms of the token percentage on the speakers (500 tokens each bin).

    ![img-3.jpeg](images/4.jpeg)
    *该图像是一幅散点图，展示了三个不同编码器输出（6p_vox10102、6p_vox10204、6p_vox10999）通过t-SNE降维后的分布情况，用于比较不同语音编码的嵌入特征分布。*

    ![img-4.jpeg](images/5.jpeg)
    *该图像是一张散点图，展示了三个不同数据集（l6p_vox10102、l6p_vox10204、l6p_vox10999）在t-SNE降维空间中的分布情况，反映了语音量化输出的聚类特征。*

    ![img-5.jpeg](images/6.jpeg)
    *该图像是图表，展示了三个不同数据集（l6p_vox10102、l6p_vox10204、l6p_vox10999）在各量化值区间的百分比分布，反映了有限标量量化后不同离散语音token的使用情况。*

    *   **(a) Before Quantization:** The `t-SNE visualization` of $Encoder_1$'s outputs (continuous representations) shows distinct clusters for different speakers. This indicates that at this stage, the representations contain clear `speaker identity information`.
    *   **(b) After Quantization:** After the `FSQ` module, the `t-SNE visualization` of the `quantized representations` shows that the clusters for different speakers are `nearly indistinguishable`. They are much more intermingled.
    *   **(c) Codebook Utilization:** This figure confirms that the `FSQ tokenizer` fully utilizes its codebook, with tokens distributed across many bins, validating the 100% utilization reported in Table 4.
    *   **Analysis:** This visualization provides strong evidence that the `FSQ` module effectively `decouples speaker identity information` from the `semantic speech tokens`, pushing the representations towards a more speaker-independent space. This is a desirable property for the semantic tokenizer in a zero-shot TTS system, as it allows the `Flow Matching model` to introduce speaker timbre later.

*   **Speaker Identification (SID) Training (Figure 5):**
    The following figure (Figure 5 from the original paper) shows the convergence curves of SID training with tokens before or after quantization.

    ![img-6.jpeg](images/7.jpeg)
    *该图像是图表，展示了SID训练过程中量化前后在训练集和验证集上的准确率随训练步数的变化情况。图中可以看出经过量化的准确率显著下降，展示了编码器训练效果优于量化模型。*

    *   **Methodology:** The `S3prl toolkit` [55] was used to train a `Speaker Identification (SID)` task using representations either before or after the quantization step in the `Sensevoice-large encoder with FSQ`.
    *   **Results:** The `SID layer` trained with `tokens before quantization` shows clear convergence and high accuracy (indicated by the blue lines reaching a high plateau). In contrast, the `SID layer` trained with `quantized tokens` (red lines) `does not converge`, remaining at a low accuracy level.
    *   **Analysis:** This experimental result further corroborates the visualization: the `quantized speech tokens` have largely lost their `speaker-specific information`. This confirms the `decoupling function` of the tokenizer, ensuring that the semantic tokens are primarily focused on content rather than speaker identity, which is crucial for flexible `speaker imitation` in downstream TTS tasks.

## 6.2. Comparison Results with Baselines

### 6.2.1. Librispeech test-clean Evaluation
The paper first evaluates CosyVoice 2 against several open-source models on the `Librispeech test-clean` subset.

The following are the results from Table 5 of the original paper:

| Model | WER (%) | NMOS | SS |
| :---- | :------ | :--- | :--- |
| Human | 2.66 | 3.84 | 0.697 |
| ChatTTS [56] | 6.84 | 3.89 | - |
| GPT-SoVITs [57] | 5.13 | 3.93 | 0.405 |
| OpenVoice [58] | 3.47 | 3.87 | 0.299 |
| ParlerTTS [59] | 3.16 | 3.86 | - |
| EmotiVoice [60] | 3.14 | 3.93 | - |
| CosyVoice [34] | 2.89 | 3.93 | 0.743 |
| CosyVoice 2 | **2.47** | **3.96** | **0.745** |
| CosyVoice 2-S | 2.45 | 3.90 | 0.751 |

**Table 5: Content consistency (WER), speaker similarity (SS) and speech quality (NMOS) results on LibriSpeech test-clean subset of baselines and CosyVoice 2. Whisper-Large V3 is employed as the ASR model and punctuations are excluded before WER calculation.**

*   **Overall Performance:** CosyVoice 2 achieves `state-of-the-art performance` on this English dataset, surpassing all baseline models across `WER`, `NMOS`, and `SS`.
*   **Content Consistency (WER):** CosyVoice 2 has the lowest `WER` at 2.47%, even outperforming `human speech` (2.66%). This indicates excellent `content consistency`. Its predecessor, `CosyVoice`, had a WER of 2.89%.
*   **Speech Quality (NMOS):** CosyVoice 2 achieves the highest `NMOS` at 3.96, marginally better than `GPT-SoVITs`, `EmotiVoice`, and `CosyVoice` (all 3.93). It also slightly surpasses `human speech` (3.84). This suggests superior `perceptual quality`.
*   **Speaker Similarity (SS):** CosyVoice 2 has the highest `SS` at 0.745, improving upon `CosyVoice` (0.743) and significantly outperforming `GPT-SoVITs` (0.405) and `OpenVoice` (0.299). It is also markedly higher than `human speech` (0.697), suggesting that the model can capture and replicate speaker characteristics very effectively.
*   **Streaming Mode (CosyVoice 2-S):** The streaming version `CosyVoice 2-S` shows `virtually lossless` performance compared to the offline `CosyVoice 2`, with comparable `WER` (2.45%), slightly lower `NMOS` (3.90), and even slightly higher `SS` (0.751).
*   **Analysis:** The results strongly support the claim of `human-parity synthesis quality` and highlight the significant improvements of CosyVoice 2 over its predecessor and other leading models, particularly in `content consistency` and `speaker similarity`. The minimal performance drop in streaming mode is also a crucial validation of its core innovation.

### 6.2.2. SEED Test Sets Evaluation
The paper evaluates CosyVoice 2 on the `SEED test sets` (`test-zh`, `test-en`, `test-hard`) which feature diverse input texts and reference speeches.

The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">test-zh</th>
<th colspan="2">test-en</th>
<th colspan="2">test-hard</th>
</tr>
<tr>
<th>CER (%) ↓</th>
<th>SS ↑</th>
<th>WER (%) ↓</th>
<th>SS ↑</th>
<th>WER (%) ↓</th>
<th>SS ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Human</td>
<td>1.26</td>
<td>0.755 (0.775)</td>
<td>2.14</td>
<td>0.734 (0.742)</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Vocoder Resyn.</td>
<td>1.27</td>
<td>0.720</td>
<td>2.17</td>
<td>0.700</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Seed-TTS† [33]</td>
<td><b>1.12</b></td>
<td>0.796</td>
<td>2.25</td>
<td>0.762</td>
<td>7.59</td>
<td><b>0.776</b></td>
</tr>
<tr>
<td>FireRedTTS [35]</td>
<td>1.51</td>
<td>0.635 (0.653)</td>
<td>3.82</td>
<td>0.460 (0.526)</td>
<td>17.45</td>
<td>0.621 (0.639)</td>
</tr>
<tr>
<td>MaskGCT [18]</td>
<td>2.27</td>
<td>0.774 (0.752)</td>
<td>2.62</td>
<td>0.714 (0.730)</td>
<td>10.27</td>
<td>0.748 (0.720)</td>
</tr>
<tr>
<td>E2 TTS (32 NFE)† [31]</td>
<td>1.97</td>
<td>0.730</td>
<td>2.19</td>
<td>0.710</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>F5-TTS (32 NFE) [32]</td>
<td>1.56</td>
<td>0.741 (0.794)</td>
<td><b>1.83</b></td>
<td>0.647 (0.742)</td>
<td>8.67</td>
<td>0.713 (0.762)</td>
</tr>
<tr>
<td>CosyVoice [34]</td>
<td>3.63</td>
<td>0.723 (0.775)</td>
<td>4.29</td>
<td>0.609 (0.699)</td>
<td>11.75</td>
<td>0.709 (0.755)</td>
</tr>
<tr>
<td>CosyVoice 2</td>
<td>1.45</td>
<td>0.748 (<b>0.806</b>)</td>
<td>2.57</td>
<td>0.652 (0.736)</td>
<td><b>6.83</b></td>
<td>0.724 (<b>0.776</b>)</td>
</tr>
<tr>
<td>CosyVoice 2-S</td>
<td>1.45</td>
<td><b>0.753</b> (<b>0.812</b>)</td>
<td>2.38</td>
<td><b>0.654</b> (<b>0.743</b>)</td>
<td>8.08</td>
<td>0.732 (0.785)</td>
</tr>
</tbody>
</table>

**Table 6: Results of CosyVoice 2 and recent TTS models on the SEED test sets. † denotes close-sourced models. For speaker similarity, the result in a bracket are measured by ERes2Net, while the results outside brackets are measured by WavLM-based models.**

*   **test-zh (Chinese):**
    *   CosyVoice 2 achieves a `CER` of 1.45%, which is competitive with human performance (1.26%) and significantly better than its predecessor `CosyVoice` (3.63%). It falls slightly short of the commercial `Seed-TTS` (1.12%) but surpasses all other open-source models.
    *   For `SS`, CosyVoice 2 achieves 0.748 (0.806 with ERes2Net), which is comparable to `human` (0.755 / 0.775) and `Seed-TTS` (0.796).
*   **test-en (English):**
    *   CosyVoice 2's `WER` is 2.57%, placing it fourth among the models, slightly higher than `human` (2.14%), `F5-TTS` (1.83%), and `E2 TTS` (2.19%).
    *   For `SS`, CosyVoice 2 (0.652 / 0.736) is also competitive but not the top performer.
    *   **Analysis:** The authors attribute the slightly lower performance in English compared to Chinese to an `imbalance in training data volume` (130,000 Chinese hours vs. 30,000 English hours for CosyVoice 2's training data). This suggests that more English data could further improve performance.
*   **test-hard (Challenging Cases):**
    *   The offline `CosyVoice 2` achieves the `state-of-the-art WER` of 6.83% among all compared baselines, demonstrating strong `robustness` in challenging scenarios (e.g., `text repetition`, `tongue twisters`). This is a substantial improvement over `CosyVoice` (11.75%).
    *   For `SS`, CosyVoice 2 (0.724 / 0.776) is again highly competitive.
*   **Human-Parity Synthesis:** Overall, CosyVoice 2 exhibits `comparable content consistency` to human speech and `superior speaker similarity`, indicating it has achieved `human-parity synthesis capability`. The authors also note that `recognition errors` can stem from the `ASR model` itself, further supporting this conclusion.
*   **Streaming Mode (CosyVoice 2-S):**
    *   In typical test cases (`test-zh`, `test-en`), the `streaming mode (CosyVoice 2-S)` performs `nearly losslessly` compared to the offline `CosyVoice 2`. Its `CER` for `test-zh` is 1.45% (same as offline) and `WER` for `test-en` is 2.38% (slightly better than offline's 2.57%).
    *   A slight degradation in `content consistency` (WER 8.08%) is observed in `test-hard` cases, highlighting the inherent challenge of `streaming models` with limited context.
    *   **Analysis:** This validates the effectiveness of the `unified streaming/non-streaming framework`, demonstrating that streaming can be achieved with minimal quality compromise for most applications.
*   **Speaker Similarity Discrepancy:** The paper notes that `speaker similarity` results are inconsistent between `WavLM`-based and `ERes2Net` models. This suggests an ongoing research challenge in automatically evaluating `speaker similarity`. The authors chose `ERes2Net` for subsequent experiments to maintain consistency.

## 6.3. Modular Ablation Study

### 6.3.1. Ablation on Text-Speech Language Model
This study investigates the impact of key modifications to the `text-speech language model`.

The following are the results from Table 7 of the original paper:

| Model | test-zh | | test-en | | test-hard | |
| :---- | :------ | :- | :------ | :- | :-------- | :- |
| | CER (%) | SS | WER (%) | SS | WER (%) | SS |
| CosyVoice | 3.63 | 0.775 | 4.29 | 0.699 | 11.75 | 0.755 |
| + LLM init. | 2.96 | 0.808 | 4.57 | 0.730 | 9.94 | 0.789 |
| + Drop Spk Emb. | 2.56 | 0.804 | 3.81 | 0.740 | 9.66 | 0.778 |
| + FSQ (CosyVoice 2) | <b>1.45</b> | 0.806 | 2.57 | 0.736 | <b>6.83</b> | 0.776 |
| + Pitch Loss | <b>1.19</b> | 0.802 | <b>2.40</b> | 0.728 | <b>6.29</b> | 0.769 |

**Table 7: Modular analysis on the modifications of text-speech language model.**

*   **Baseline (CosyVoice):** The predecessor model shows a `CER` of 3.63% (`test-zh`), `WER` of 4.29% (`test-en`), and `WER` of 11.75% (`test-hard`), with decent `SS`.
*   **`+ LLM init.`:** Replacing the randomly initialized `LM` with a pre-trained `LLM backbone` (`Qwen2.5-0.5B`) significantly improves `content consistency`. `CER` on `test-zh` drops from 3.63% to 2.96% (18.46% relative improvement), and `WER` on `test-hard` drops from 11.75% to 9.94% (15.40% relative improvement). `SS` also shows an increase (from 0.775 to 0.808 on `test-zh`).
    *   **Analysis:** This highlights the immense benefit of leveraging powerful pre-trained textual `LLMs` for `context understanding` and `text-to-speech alignment`.
*   **`+ Drop Spk Emb.`:** Removing the `speaker embedding` from the `text-speech LM` further improves `content consistency`. `CER` on `test-zh` drops to 2.56%, `WER` on `test-en` to 3.81%, and `WER` on `test-hard` to 9.66%. `SS` remains stable.
    *   **Analysis:** This validates the hypothesis that `speaker embeddings` in the `LM` can introduce `information leakage` and disturb `in-context learning`. Decoupling `speaker identity` to be primarily handled by the `Flow Matching model` allows the `LM` to focus purely on content.
*   **`+ FSQ (CosyVoice 2)`:** Replacing `VQ` with `FSQ` results in the full CosyVoice 2 model. This brings a dramatic improvement in `content consistency`. `CER` on `test-zh` drops to 1.45% (a 43.3% relative improvement from the previous step), and `WER` on `test-hard` drops to 6.83%. `SS` remains largely unchanged.
    *   **Analysis:** This demonstrates the superior ability of `FSQ` to fully utilize the codebook, capture more `content information` and `context variation`, and achieve better alignment between `text` and `speech tokens`.
*   **`+ Pitch Loss`:** Incorporating `pitch loss` as a constraint during `FSQ-based speech tokenizer` training further improves `CER` to 1.19% (`test-zh`) and `WER` to 6.29% (`test-hard`), slightly improving `test-en` WER to 2.40%.
    *   **Analysis:** This suggests that explicitly modeling `pitch information` during tokenization can provide additional benefits for `downstream TTS tasks`, indicating richer `prosodic information` capture.

### 6.3.2. Ablation on Streaming Modules
This study evaluates the impact of streaming modules on performance. `Chunk size` is set to 15 for streaming modules.

The following are the results from Table 8 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">LM</th>
<th rowspan="2">FM</th>
<th colspan="2">test-zh</th>
<th colspan="2">test-en</th>
<th colspan="2">test-hard</th>
</tr>
<tr>
<th>CER (%)</th>
<th>SS</th>
<th>WER (%)</th>
<th>SS</th>
<th>CER (%)</th>
<th>SS</th>
</tr>
</thead>
<tbody>
<tr>
<td>M1</td>
<td>Offline</td>
<td>Offline</td>
<td>1.45</td>
<td>0.806</td>
<td>2.57</td>
<td>0.736</td>
<td>6.83</td>
<td>0.776</td>
</tr>
<tr>
<td>M2</td>
<td>Offline</td>
<td>Stream.</td>
<td>1.46</td>
<td>0.811</td>
<td>2.60</td>
<td>0.743</td>
<td>7.12</td>
<td>0.788</td>
</tr>
<tr>
<td>M3</td>
<td>Stream.</td>
<td>Offline</td>
<td><b>1.38</b></td>
<td>0.806</td>
<td><b>2.51</b></td>
<td>0.737</td>
<td>7.88</td>
<td>0.773</td>
</tr>
<tr>
<td>M4</td>
<td>Stream.</td>
<td>Stream.</td>
<td>1.45</td>
<td><b>0.812</b></td>
<td>2.38</td>
<td><b>0.743</b></td>
<td>8.08</td>
<td><b>0.785</b></td>
</tr>
</tbody>
</table>

**Table 8: Modular analysis on the impact of streaming modules in CosyVoice 2. Chunk size is set to 15 for streaming modules.**

*   **M1 (Offline LM, Offline FM):** This is the full offline CosyVoice 2 baseline.
*   **M2 (Offline LM, Streaming FM):**
    *   `Content Consistency` (`CER`/`WER`): Shows a minimal impact on `test-zh` (1.46% vs. 1.45%) and `test-en` (2.60% vs. 2.57%). A slight increase is seen in `test-hard` (7.12% vs. 6.83%).
    *   `Speaker Similarity` (`SS`): Shows a slight increase across all test sets (e.g., 0.811 vs. 0.806 on `test-zh`).
    *   **Analysis:** The `streaming flow matching model` has a minor negative impact on `content consistency` but surprisingly `improves speaker similarity`. The authors suggest the `SS` improvement might be due to a higher `prompt-to-generation ratio` for initial chunks in `streaming mode` compared to `offline mode` with many `padding tokens`. This confirms the effectiveness of semantic-acoustic decoupled modeling in CosyVoice 2, making the `FM` less sensitive to context limitations for `content consistency`.
*   **M3 (Streaming LM, Offline FM):**
    *   `Content Consistency`: Interestingly, the `streaming LM` with `offline FM` shows slightly better `CER`/`WER` on `test-zh` (1.38% vs. 1.45%) and `test-en` (2.51% vs. 2.57%). However, it significantly degrades on `test-hard` (7.88% vs. 6.83%).
    *   `Speaker Similarity`: Remains largely stable.
    *   **Analysis:** The `streaming LM` has a more pronounced impact on `content consistency`, especially in challenging cases (`test-hard`), likely due to the inherent loss of `long-range contextual information` when processing text in chunks.
*   **M4 (Streaming LM, Streaming FM):** This represents the full streaming `CosyVoice 2-S` model.
    *   `Content Consistency`: `CER` on `test-zh` is 1.45% (matching offline), `WER` on `test-en` is 2.38% (best for English), but `WER` on `test-hard` is 8.08% (highest).
    *   `Speaker Similarity`: Achieves the highest `SS` across all test sets (e.g., 0.812 on `test-zh`, 0.743 on `test-en`, 0.785 on `test-hard`).
    *   **Analysis:** The combination of `streaming LM` and `streaming FM` generally maintains strong performance in typical cases, but the `streaming LM`'s impact on `content consistency` is evident in `test-hard` scenarios. The `SS` benefits from the `streaming FM` are retained. This confirms the overall `virtually lossless` nature of streaming for typical use cases while identifying `challenging texts` as the primary performance bottleneck for the `streaming LM`.

## 6.4. Results on Japanese and Korean Benchmarks
CosyVoice 2's performance on Japanese and Korean `test sets` was evaluated.

The following are the results from Table 9 of the original paper:

| Model | test-ja | | | test-ko | | |
| :---- | :------ | :- | :-- | :------ | :- | :-- |
| | CER (%) | SS | NMOS | CER (%) | SS | NMOS |
| CosyVoice 2 | 18.79 | 0.630 | 3.42 | 7.98 | 0.707 | 3.73 |
| CosyVoice 2-S | 21.41 | 0.629 | 3.35 | 9.06 | 0.714 | 3.60 |

**Table 9: The content consistency (CER), speaker similarity (SS), and speech quality (NMOS) of CosyVoice 2 and its streaming counterpart on the Japanese test-ja and Korean test-ko test sets.**

*   **Performance Discrepancy:** `CosyVoice 2` performs `significantly better on Korean` than on Japanese across all metrics.
    *   **Korean (test-ko):** `CER` 7.98%, `SS` 0.707, `NMOS` 3.73. These are reasonably good scores.
    *   **Japanese (test-ja):** `CER` 18.79%, `SS` 0.630, `NMOS` 3.42. These scores indicate a notable performance degradation compared to Korean and other languages.
*   **Reasons for Discrepancy:**
    *   **Character Set Overlap:** The primary reason for lower Japanese performance is the `overlap in character sets between Japanese and Chinese`. This leads to `Chinese pronunciations` being generated in Japanese contexts, causing significant `content consistency` issues.
    *   **Korean Advantage:** Korean does not have similar `character overlap` with other languages, allowing for much better `speech synthesis performance`.
    *   **Data Imbalance:** The authors suggest that `data imbalance` (e.g., Japanese and Korean have significantly less training data than Chinese and English, as shown in Table 3) could also contribute. Increasing training data volume is a future direction.
*   **Streaming Mode:** `CosyVoice 2-S` shows a slight drop in performance across `CER` and `NMOS` for both languages compared to the offline model, consistent with findings in `test-hard` cases, but `SS` remains stable or even slightly increases (for Korean).
*   **Analysis:** This highlights a current limitation concerning `multilingual synthesis` for languages with shared script but distinct phonology. Future work will need to address how to enhance `linguistic context` to prevent cross-lingual interference.

## 6.5. Results on Instructed Generation
To evaluate the `instructed generation` capabilities, a Chinese test set of 290 samples (29 instruction types, 10 texts each) was created. Five audio prompts from different speakers were used as conditions for the `Flow Matching model`. Evaluation was done in `offline mode`.

*   **Objective Metrics:** `CER`, `SS`, `NMOS`.
*   **Subjective Metric:** `Mean Opinion Score for Instruction (MOS-I)`.
    *   `MOS-I` ranges from 1 to 5, with scores in 0.5 increments.
    *   Each sample rated by 10 native Chinese speakers.
    *   Criteria: Adherence to specified instructions (emotion, rate, dialect, role-playing), naturalness and accuracy of fine-grained controls (laughter, emphasis).

        The following are the results from Table 10 of the original paper:

        | Model | CER (%) | SS | NMOS | MOS-I |
        | :---------------------- | :------ | :--- | :--- | :---- |
        | CosyVoice-Instruct [34] | 1.72 | 0.797 | 3.94 | 3.09 |
        | CosyVoice 2 | <b>1.52</b> | <b>0.804</b> | 3.94 | <b>4.06</b> |
        | CosyVoice 2 w/o Instruction | 0.97 | 0.817 | 4.02 | 2.28 |

**Table 10: Evaluation results for content consistency (CER), speaker similarity (SS), speech quality (NMOS), and MOS-I (Instruction, assessing the accuracy and naturalness of instruction) on an in-house Chinese test set for CosyVoice-Instruct, CosyVoice 2, and CosyVoice 2 without instruction input. The Paraformer model is used as the ASR system, with punctuation marks excluded from the CER calculation. Dialect data is not included in the CER calculation because the Paraformer model cannot recognize Chinese dialect speech.**

*   **CosyVoice 2 vs. CosyVoice-Instruct:**
    *   CosyVoice 2 shows `superior content consistency` (`CER` 1.52% vs. 1.72%), `higher speaker similarity` (`SS` 0.804 vs. 0.797), and a dramatically `higher MOS-I` (`4.06 vs. 3.09`). `NMOS` is comparable (3.94).
    *   **Analysis:** The significant improvement in `MOS-I` for CosyVoice 2 (from 3.09 to 4.06) demonstrates its much more effective and natural `instruction adherence`. This highlights the success of integrating the instructed dataset and the overall model optimizations in enhancing controllability.
*   **CosyVoice 2 w/o Instruction:**
    *   When explicit instructions are removed, `MOS-I` significantly drops to 2.28, as expected.
    *   However, `CER` (0.97%), `SS` (0.817), and `NMOS` (4.02) actually `improve` in this mode.
    *   **Analysis:** This suggests that `instruction controllability` is a complex task that can sometimes introduce slight compromises in baseline `content consistency` or `speaker similarity`. The fact that `CER` without instruction is even lower than `human CER` (1.26% on `test-zh` in Table 6) implies that the model's fundamental synthesis quality is very high when not burdened with specific instruction requirements. It also indicates that `instruction controllability` is difficult to emerge implicitly from content text alone.

## 6.6. Results on Speaker Fine-tuned Models
The paper explores `multi-speaker fine-tuning (mSFT)` by performing `unsupervised clustering` on `speaker embeddings` to stabilize timbre. They demonstrate effective `SFT` with as few as 400 audio recordings.

The following figure (Figure 6 from the original paper) shows the results of CosyVoice 2 SFT Models under the SEED evaluation settings. CER is used for test-zh and test-hard, while WER is used for test-en.

![img-7.jpeg](images/8.jpeg)
*该图像是图表，展示了CosyVoice 2在SEED评测设置下的SFT模型结果。图中对比了五位说话者（Spk A-E）在三个测试集（test-zh、test-en、test-hard）上的CER和WER指标，展示了不同语言和难度下的性能差异。*

*   **SFT Performance:** Figure 6 illustrates the `CER`/`WER` for `mSFT` models on five different speakers (Spk A-E) across `test-zh`, `test-en`, and `test-hard` sets.
*   **Contextual Understanding Inheritance:** The experiments indicate that most fine-tuned speakers can `inherit the robust contextual understanding and perception` of the `zero-shot TTS model`. This allows them to naturally express various moods and emotions in response to input text.
*   **Objective Metrics Stability:** Only `slight variations` in objective metrics are observed among different speakers after fine-tuning, implying that the `SFT` process successfully adapts the model to new voices without significant degradation.
*   **Analysis:** `mSFT` is a viable and effective strategy for achieving high-quality speech synthesis for specific target speakers, even with limited data (400 recordings). It leverages the base model's strong `zero-shot capabilities` while specializing in a target voice.

## 6.7. LM Fine-tuning with Reinforcement Learning
Reinforcement learning was applied to `Spk E`, a speaker with a `complex voice` and `faster speech speed`, for whom only Chinese recordings were available. This aimed to address cases where `SFT` alone might not be sufficient.

The following are the results from Table 11 of the original paper:

| Model | Inhome Target Speaker | | | SEED tests(%) | | |
| :---- | :-------------------- | :- | :- | :------------ | :- | :- |
| | WER(%) | NMOS | SS | zh | en | hard |
| Ground Truth | 6.00 | 3.87 | 0.697 | 1.26 | 2.14 | - |
| CosyVoice 2 | 5.34 | 3.91 | 0.721 | 1.45 | 2.57 | 6.83 |
| CosyVoice 2-SFT | 7.15 | 3.96 | 0.795 | 1.50 | 4.26 | 7.90 |
| + LASR | 6.79 | 3.96 | 0.795 | <b>1.29</b> | <b>3.53</b> | 7.30 |
| + LDPO | 6.83 | 3.96 | 0.792 | 1.43 | 4.02 | 8.31 |
| + LASR + LDPO | <b>6.64</b> | <b>3.97</b> | <b>0.796</b> | <b>1.25</b> | <b>3.17</b> | <b>6.66</b> |

**Table 11: Content consistency (WER), speaker similarity (SS) and speech quality (NMOS) comparison for reinforcement learning models on Spk E.**

*   **Base Model (CosyVoice 2):** Has a `WER` of 5.34% for the `in-home target speaker` and strong performance on `SEED tests`.
*   **SFT Model (CosyVoice 2-SFT):**
    *   Shows higher `NMOS` (3.96) and `SS` (0.795) compared to the base model (3.91 and 0.721). This is expected as `SFT` improves speaker-specific characteristics.
    *   However, `WER` `worsens` for the `in-home target speaker` (7.15% vs. 5.34%) and on `test-en` (4.26% vs. 2.57%).
    *   **Analysis:** The authors explain that the `base model` often synthesizes at a `slower speed`, which is more favorable to `ASR systems`. `SFT` makes the speech faster and more natural for the speaker, but this `faster speed` might lead to higher `WER` if the ASR is not robust to it.
*   **`+ LASR` (Differentiable ASR Rewards):**
    *   Applying `differentiable ASR rewards` (`LASR`) to the `SFT model` `reduces WER` for the `in-home speaker` (from 7.15% to 6.79%) and significantly improves `WER` on `SEED tests` (`test-zh` from 1.50% to 1.29%, `test-en` from 4.26% to 3.53%).
    *   `NMOS` and `SS` are maintained.
    *   **Analysis:** `LASR` successfully improves `pronunciation accuracy` by directly optimizing for ASR performance without harming `speaker similarity` or `speech quality`. This method shows good `generalization ability` to out-of-domain data, unlike `DPO` in some cases.
*   **`+ LDPO` (Direct Preference Optimization):**
    *   Applying `DPO` (`LDPO`) also `reduces WER` for the `in-home speaker` (from 7.15% to 6.83%) and maintains other metrics.
    *   However, `LDPO` `worsens WER` on `test-hard` (from 7.90% to 8.31%) for `SEED tests`.
    *   **Analysis:** `DPO` can be sensitive to the quality of `preferred/rejected samples`. `Hard samples` (e.g., those with repetitions) might be misconstrued as `rejected samples` during `DPO training`, leading to a decline in robustness on such specific cases.
*   **`+ LASR + LDPO` (Combined RL):**
    *   Combining both `differentiable ASR rewards` and `DPO` yields the best results. `WER` for the `in-home speaker` further decreases to 6.64%, `NMOS` slightly improves to 3.97%, and `SS` is maximized at 0.796.
    *   Crucially, this combination leads to the best `WER` on `SEED tests` across all RL variants (`test-zh` 1.25%, `test-en` 3.17%, `test-hard` 6.66%), even surpassing the base CosyVoice 2 model on `test-zh` and `test-hard`.
    *   **Analysis:** This demonstrates that `differentiable ASR rewards` and `DPO` can be complementary, with `LASR` providing better `generalization ability` and `LDPO` refining preferences, leading to robust improvements across various conditions.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
CosyVoice 2 represents a significant advancement in the field of scalable streaming speech synthesis. Building on its predecessor, CosyVoice, this model integrates comprehensive and systematic optimizations to address the critical need for low-latency, real-time speech generation, especially for interactive multi-modal LLM applications. The key contributions include:
1.  A **unified framework** for both streaming and non-streaming synthesis, achieved through a novel `unified text-speech language model` and a `chunk-aware causal flow matching model`, which enables `virtually lossless synthesis quality` in streaming mode.
2.  A **streamlined LM architecture** that directly leverages powerful **pre-trained textual LLMs** (`Qwen2.5-0.5B`) as its backbone, enhancing `context understanding` by removing `speaker embeddings` and dedicated `text encoders`.
3.  The adoption of **finite scalar quantization (FSQ)** in the `speech tokenizer`, which dramatically improves `codebook utilization` to 100% and better preserves `speech information`.
4.  An **upgraded instructed TTS capacity** supporting both `natural language` and `fine-grained controls` over `emotion`, `accent`, `role style`, and `vocal bursts`, enabling more versatile and vivid speech generation.

    Through rigorous evaluations, CosyVoice 2 consistently achieves `human-parity naturalness`, minimal response latency, and strong robustness in challenging scenarios. The modular design, including `multi-speaker fine-tuning (mSFT)` and `reinforcement learning` (especially `differentiable ASR rewards`), further enhances its adaptability and performance for specific speakers.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations of CosyVoice 2:

1.  **Limited Language Support & Overlap Issues:** The model currently supports a limited number of languages. For languages with `overlapping character sets` (e.g., Japanese and Chinese), `synthesis performance` may degrade due to cross-lingual interference, presenting an open challenge for future research.
2.  **Lack of Acoustic Feature Control by Textual Instructions:** CosyVoice 2 cannot control specific `acoustic characteristics` like `timbre` directly through `textual instructions`. This capability could be valuable for `role-playing applications` and offers an interesting avenue for exploration.
3.  **Poor Singing Performance:** The model does not perform well when tasked with `singing synthesis`. This indicates that its current architecture and training data are primarily optimized for spoken language, and generating melodic speech requires specialized approaches.

    Future work explicitly mentioned by the authors includes:
*   Exploring `data scaling` for English to enhance `content consistency`.
*   Developing ways to enhance `linguistic context` for `multilingual synthesis` to mitigate `character overlap` issues.
*   More detailed experiments and analyses on the benefits of `pitch loss` in the `FSQ-based speech tokenizer`.

## 7.3. Personal Insights & Critique
CosyVoice 2 stands out as a highly practical and technically comprehensive advancement in TTS. The paper rigorously details its innovations, especially concerning the critical transition from offline to streaming synthesis, which is paramount for the current surge in conversational AI.

**Strengths:**
*   **Holistic Streaming Solution:** The unified framework for streaming and non-streaming synthesis, coupled with the `chunk-aware causal flow matching model`, is a particularly strong contribution. This addresses a major practical bottleneck in deploying high-quality generative TTS models in real-time. The `virtually lossless` performance in streaming mode is a significant achievement.
*   **Leveraging LLMs Effectively:** Directly using a pre-trained `LLM` as the `LM` backbone is a smart design choice. It capitalizes on the vast `contextual understanding` capabilities of `LLMs` without reinventing the wheel, simplifying the architecture and improving performance. The careful removal of `speaker embeddings` from the LM to prevent `information leakage` is also insightful.
*   **FSQ for Tokenization:** The `FSQ` approach for the `speech tokenizer` is a robust improvement over `VQ`. Achieving 100% `codebook utilization` directly translates to more efficient information capture and higher fidelity in the discrete representations, which directly impacts downstream quality.
*   **Detailed Ablation Studies:** The thorough `ablation studies` clearly demonstrate the incremental benefits of each modification, providing strong empirical evidence for the design choices.
*   **Instructed Generation:** The comprehensive `instructed generation` capabilities open up new avenues for user control and expressive speech, making the model highly versatile.

**Potential Issues/Unverified Assumptions/Areas for Improvement:**
*   **Language Overlap Challenge:** While acknowledged as a limitation, the character overlap issue for languages like Japanese is a deep-seated problem in multilingual models. Future work might explore language-specific tokenization layers or more advanced `language-ID conditioning` to truly disentangle phonology for shared scripts.
*   **"Human-Parity" Interpretation:** While the `WER`/`CER` and `NMOS` scores often surpass human performance, it's important to remember that `ASR systems` are not perfect proxies for human perception. `ASR` models can sometimes be biased towards certain speech characteristics (e.g., slower speech, as noted for the base model). `MOS` scores for naturalness remain the gold standard for human-like quality. The `MOS-I` also indicates that instructed synthesis can be challenging.
*   **Scalability of Training Data:** While the training data is extensive, the `imbalance` between languages (e.g., Chinese vs. Japanese/Korean) is evident in the results. Future scaling efforts might face diminishing returns or require more sophisticated techniques to balance language representation effectively.
*   **Generalizability of Streaming to Other Architectures:** The `chunk-aware flow matching` design is a clever way to enable streaming for continuous-time generative models. It would be insightful to explore if this "unfolded neural network causality" approach can be broadly applied to other `non-autoregressive generative models` beyond `Flow Matching`, potentially paving the way for wider adoption of streaming in these models.
*   **Computational Cost of LLM Backbone:** While `Qwen2.5-0.5B` is a relatively small `LLM`, using an `LLM` as a backbone might still entail higher computational costs compared to custom-built `Transformer LMs`. The paper does not delve into this trade-off explicitly in terms of inference speed or resource usage beyond the `latency analysis`.

**Transferability/Application to Other Domains:**
*   The `FSQ module` could be a valuable replacement for `VQ` in other `discrete representation learning tasks`, especially in audio, image, and video compression or generative models where `codebook utilization` is a concern.
*   The `unified streaming/non-streaming training paradigm` and `chunk-aware causal design` could inspire similar solutions for other generative tasks where real-time interaction is crucial, such as `streaming video generation`, `real-time animation`, or `interactive music synthesis`.
*   The `differentiable ASR reward` mechanism in `RL fine-tuning` could be adapted to improve other generative models where output quality can be objectively measured by a proxy model (e.g., image generation using an object detector as a reward for correctness).

    Overall, CosyVoice 2 represents a robust engineering feat that successfully tackles the practical challenges of high-quality streaming TTS. Its innovations in tokenization, `LLM` integration, and generative model causality offer valuable insights and blueprints for future research in interactive AI.