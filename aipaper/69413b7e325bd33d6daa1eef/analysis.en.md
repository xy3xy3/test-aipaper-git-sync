# 1. Bibliographic Information

## 1.1. Title
The title of the paper is "Qwen3-Omni Technical Report." This title indicates that the paper details the technical specifications, architecture, and performance of the Qwen3-Omni model, which is a version within the Qwen series. The term "Omni" suggests a comprehensive, all-encompassing nature, particularly across different data modalities.

## 1.2. Authors
The paper lists "Qwen Team" as the primary author group. The authors are further categorized into "Core Contributors" and "Contributors," indicating a large collaborative effort. Prominent contributors include Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. The affiliations are implicitly linked to the QwenLM project, likely associated with a major technology company or research institution given the scale of the project.

## 1.3. Journal/Conference
The paper is published at `https://arxiv.org/abs/2509.17765v1`, which is the arXiv preprint server. While arXiv hosts preprints and is not a peer-reviewed journal or conference in itself, it is a widely recognized platform for disseminating new research in fields like artificial intelligence and machine learning. Its reputation is high for early access to cutting-edge work, though the content has not yet undergone formal peer review for a specific conference or journal. The future publication date of `2025-09-22T13:26:24.000Z` suggests it is an upcoming release or a prospective work.

## 1.4. Publication Year
The publication date (or rather, preprint submission date) is specified as `2025-09-22T13:26:24.000Z`. This indicates it is a very recent or upcoming technical report.

## 1.5. Abstract
The abstract presents Qwen3-Omni as a groundbreaking single multimodal model capable of maintaining state-of-the-art (SOTA) performance across text, image, audio, and video modalities without degradation compared to single-modal models of similar size. It particularly excels in audio tasks, achieving open-source SOTA on 32 benchmarks and overall SOTA on 22, surpassing powerful closed-source models. The model employs a `Thinker-Talker Mixture-of-Experts (MoE)` architecture for unified perception and generation, facilitating fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To achieve low `first-packet latency` in streaming speech synthesis, the `Talker` autoregressively predicts `discrete speech codecs` using a `multi-codebook` scheme. It replaces computationally intensive `block-wise diffusion` with a lightweight `causal ConvNet`, enabling immediate streaming. In cold-start scenarios, it boasts a theoretical end-to-end `first-packet latency` of 234 ms. A dedicated `Thinking model` is introduced for enhanced multimodal reasoning. Furthermore, a specialized version, `Qwen3-Omni-30B-A3B-Captioner`, is fine-tuned for generating detailed, low-hallucination captions for arbitrary audio inputs, addressing a gap in general-purpose audio captioning. Three main model variants (`Qwen3-Omni-30B-A3B`, `Qwen3-Omni-30B-A3B-Thinking`, and `Qwen3-Omni-30B-A3B-Captioner`) are publicly released under the Apache 2.0 license.

## 1.6. Original Source Link
The original source link is `https://arxiv.org/abs/2509.17765v1`.
The PDF link is `https://arxiv.org/pdf/2509.17765v1.pdf`.
This is a preprint published on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the prevalent issue of modality trade-offs in `LLM-centric multimodal models`. Historically, when researchers integrate multiple modalities (like text, image, audio, video) into a single model, gains in one modality often come at the cost of degraded performance in others. This limits the practical utility and holistic intelligence of such models.

This problem is highly important in the current field because human perception and intelligence naturally involve the coordinated use of multiple senses. Developing truly intelligent systems requires `natively multimodal systems` that can process and reason across diverse inputs seamlessly. Previous research has made rapid advances in unimodal large models (like `Large Language Models` or `LLMs`, and `Vision-Language Models`), but integrating these capabilities without sacrificing specialized performance has remained a significant challenge. The existing gaps include:
*   **Performance Degradation:** Multimodal models often don't match the SOTA performance of their unimodal counterparts.
*   **Lack of True Cross-Modal Reasoning:** While some models handle multiple inputs, deep, synergistic reasoning across modalities is still emerging.
*   **Latency in Real-time Interaction:** Integrating complex multimodal processing for real-time applications (like voice assistants) often introduces unacceptable delays.
*   **Absence of General Audio Captioning:** A lack of general-purpose models for generating detailed descriptions from arbitrary audio inputs highlights a specific deficiency in multimodal understanding.

    The paper's entry point and innovative idea revolve around demonstrating that `joint multimodal training` can achieve `parity` across all modalities. This means the model can match the performance of specialized unimodal models while simultaneously enhancing `cross-modal capabilities` (e.g., video understanding). A key innovation is mixing `unimodal` and `cross-modal data` during the early stages of `text pretraining` to achieve this non-degrading performance. The paper also focuses on optimizing for `low-latency real-time audiovisual interaction` and strengthening `multimodal reasoning` with a dedicated `Thinking model`.

## 2.2. Main Contributions / Findings
The primary contributions and key findings of the Qwen3-Omni paper are:

*   **Achieving Non-Degrading Multimodal SOTA Performance:** Qwen3-Omni is presented as the first single multimodal model that maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to same-sized single-modal counterparts. This addresses a major limitation in previous multimodal model development.
*   **Exceptional Audio Capabilities:** The model particularly excels on audio tasks, achieving open-source SOTA on 32 out of 36 audio and audio-visual benchmarks, and overall SOTA on 22, outperforming strong closed-source models like Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe.
*   **Novel Thinker-Talker MoE Architecture:** It introduces an upgraded `Thinker-Talker Mixture-of-Experts (MoE)` architecture that unifies perception and generation across diverse modalities, yielding fluent text and natural real-time speech. This architecture allows for a decoupled yet cohesive system design.
*   **Ultra-Low Latency Streaming Synthesis:** The `Talker` component leverages a `multi-codebook` scheme and a lightweight `causal ConvNet` (`Code2Wav`) to autoregressively predict `discrete speech codecs`, enabling streaming from the first codec frame. This results in a theoretical end-to-end `first-packet latency` of 234 ms in cold-start settings, crucial for real-time interaction.
*   **Enhanced Language Support and Context Processing:** Qwen3-Omni supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. It can also process audio recordings up to 40 minutes for `ASR` (Automatic Speech Recognition) and `spoken-language understanding`.
*   **Introduction of a Dedicated Thinking Model:** To further strengthen multimodal reasoning, a `Thinking model` is introduced that explicitly reasons over inputs from any modality, including complex audio-video and audio-only scenarios.
*   **Development of a General-Purpose Audio Captioner:** By fine-tuning `Qwen3-Omni-30B-A3B`, the paper introduces `Qwen3-Omni-30B-A3B-Captioner`, which produces detailed, low-hallucination captions for arbitrary audio inputs, filling a recognized gap in the research community.
*   **Public Release of Models:** `Qwen3-Omni-30B-A3B`, `Qwen3-Omni-30B-A3B-Thinking`, and `Qwen3-Omni-30B-A3B-Captioner` are publicly released under the Apache 2.0 license, promoting open science and further research.
*   **Evidence of Cross-Modal Synergy:** Experimental results indicate that mixing `unimodal` and `cross-modal data` during early pretraining not only prevents degradation but also enables `mutual enhancement` between modalities, with text improving vision/audio and audio improving vision performance.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand Qwen3-Omni, a novice would benefit from grasping several foundational concepts in deep learning and multimodal AI:

*   **Large Language Models (LLMs):** These are neural networks trained on vast amounts of text data to understand, generate, and predict human language. They typically employ `Transformer` architectures, which use `attention mechanisms` to weigh the importance of different parts of the input sequence.
*   **Multimodal AI:** This field of artificial intelligence deals with models that can process and understand information from multiple modalities, such as text, images, audio, and video. The goal is to build systems that can integrate and reason across these different types of data, similar to how humans perceive the world.
*   **Transformer Architecture:** A neural network architecture introduced in 2017, foundational for many modern `LLMs` and `multimodal models`. It relies entirely on `attention mechanisms` to draw global dependencies between input and output. Unlike `Recurrent Neural Networks (RNNs)`, `Transformers` process sequences in parallel, making them highly efficient for long sequences.
*   **Attention Mechanism:** A core component of `Transformers`. It allows the model to weigh the importance of different parts of the input sequence when processing each element. For a given `query` ($Q$), the attention mechanism computes `attention scores` by comparing $Q$ to all `keys` ($K$) in the input. These scores are then used to take a weighted sum of `values` ($V$), producing the output. The standard scaled dot-product attention is defined as:
    \$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    \$
    Where:
    *   $Q$ is the `query matrix`.
    *   $K$ is the `key matrix`.
    *   $V$ is the `value matrix`.
    *   $d_k$ is the `dimension` of the `keys`, used for scaling to prevent the dot products from growing too large.
    *   $QK^T$ represents the dot product of queries and keys, which measures their similarity.
    *   $\mathrm{softmax}$ normalizes the scores to produce weights.
    *   The result is a weighted sum of the values, where weights are determined by attention scores.
*   **Mixture-of-Experts (MoE) Architecture:** An `MoE` model consists of multiple "expert" neural networks and a "gating network" that learns to choose which experts to activate for each input. This allows models to scale to billions or trillions of parameters while only activating a subset of parameters for any given input, reducing computational cost during inference and improving efficiency, especially for large models or high-concurrency scenarios.
*   **Speech Codecs:** These are algorithms used for encoding (compressing) and decoding (decompressing) digital audio. They represent speech as a sequence of discrete tokens (codes), enabling efficient storage and transmission.
*   **Autoregressive Models:** These models predict the next element in a sequence based on the preceding elements. In speech synthesis, an `autoregressive model` might predict the next `speech codec` token based on the tokens generated so far, allowing for continuous, sequential generation.
*   **Causal Convolutional Network (ConvNet):** A type of `Convolutional Neural Network` where the output at a given timestep only depends on previous timesteps in the input, not future ones. This is crucial for `streaming` applications, as it allows real-time processing without waiting for future input.
*   **First-Packet Latency:** In real-time streaming, this refers to the time it takes for the very first piece of data (the "first packet") to be generated and made available to the user after an input is provided. Lower latency is critical for natural, interactive experiences.
*   **Tokenization (for text and audio):** The process of breaking down raw input (text, audio, etc.) into discrete units called "tokens" that the model can process. For text, these are often words or sub-word units. For audio, they can be `mel-spectrogram` frames or `codec` representations.
*   **Rotary Position Embedding (RoPE):** A method for encoding positional information in `Transformer` models. Instead of adding fixed position embeddings, `RoPE` modifies the self-attention mechanism by rotating query and key vectors based on their absolute or relative positions, allowing `Transformers` to better capture sequence order, especially over long contexts.
*   **Supervised Fine-Tuning (SFT):** A training technique where a pre-trained model (e.g., an `LLM`) is further trained on a smaller, labeled dataset specifically designed for a particular task (e.g., instruction following or dialogue generation).
*   **Direct Preference Optimization (DPO):** A `Reinforcement Learning from Human Feedback (RLHF)` method that directly optimizes a policy to align with human preferences, often represented as pairwise comparisons, without explicitly training a separate reward model.

## 3.2. Previous Works
The paper builds upon significant advancements in both `unimodal` and `multimodal` large models.

*   **Unimodal LLMs:** The introduction cites many influential `LLMs` such as `GPT-3` (Brown et al., 2020), `GPT-4` (OpenAI, 2023), `Gemini` (Gemini Team, 2024), `Claude` (Anthropic, 2023a,b, 2024), `Llama` (Touvron et al., 2023; Dubey et al., 2024), and various `Qwen` models (Bai et al., 2023a; Yang et al., 2024, 2025a; Li et al., 2023; Liu et al., 2023; Zhu et al., 2023; Bai et al., 2025; Chu et al., 2023, 2024). These models established the `Transformer` architecture as dominant and scaled `LLM` capabilities significantly in understanding and reasoning.
*   **Natively Multimodal Systems:** The paper refers to works like `GPT-4o` (OpenAI, 2024), `Gemini 2.5` (Comanici et al., 2025), and `Qwen2.5-Omni` (Xu et al., 2025). These represent the shift towards integrating multiple modalities directly. `Qwen2.5-Omni` is a direct predecessor, providing the `Thinker-Talker architecture` which Qwen3-Omni evolves.
*   **`Qwen2.5-Omni` (Xu et al., 2025):** This model introduced the `Thinker-Talker architecture` which `Qwen3-Omni` builds upon. It also involved `chunked-prefilling` for streaming and `Multimodal Rotary Position Embedding (M-RoPE)`. `Qwen3-Omni` directly compares and highlights improvements over `Qwen2.5-Omni` throughout the paper, such as upgraded `MoE` components, the `AuT` encoder, `multi-codebook` representation, and a `causal ConvNet` for `Code2Wav`. Notably, `Qwen2.5-Omni` segmented audiovisual representations into fixed 2-second chunks, a limitation `Qwen3-Omni` addresses with `TM-RoPE`.
*   **`Whisper` Audio Encoder:** The paper mentions `Qwen3-Omni` replacing the `Whisper audio encoder` (presumably from OpenAI, although not explicitly cited in the context of `Whisper` itself, but as a general strong baseline) with their `AuT` encoder. `Whisper` is a well-known `ASR` model that processes audio into text. `Qwen3-Omni`'s `AuT` aims for stronger, more general-purpose audio representations, trained on a massive 20 million hours of supervised audio.
*   **`SigLIP2-So400m` (Tschannen et al., 2025):** The vision encoder used in `Qwen3-Omni` is initialized from `SigLIP2-So400m`, a strong vision-language model, further indicating reliance on established high-performance unimodal components.

## 3.3. Technological Evolution
The field of `Large Language Models` and `Multimodal AI` has rapidly evolved:
1.  **Text-only LLMs:** Starting with `Transformer` architectures, models grew in size and capability (e.g., GPT series, Llama, Qwen). These primarily focused on text understanding and generation.
2.  **Vision-Language Models (VLMs):** The next step involved integrating vision, often by encoding images/videos into token-like representations that `LLMs` could process (e.g., `Flamingo`, `ViT-G/14`, `Qwen-VL`). These models could answer questions about images or describe visual content.
3.  **Audio-Language Models (ALMs):** Similar to `VLMs`, these models integrated audio, enabling tasks like `ASR`, `speech-to-text translation`, and `audio understanding`.
4.  **Early Multimodal Models (Vision-Audio-Text):** More recent work, like `Gemini` and `GPT-4o`, and previously `Qwen2.5-Omni`, began integrating all three core modalities. However, a common challenge was maintaining `unimodal SOTA` performance while adding new modalities. This often led to `modality trade-offs`.
5.  **Real-time Interaction:** A crucial advancement is moving from offline processing to real-time, interactive systems, necessitating low latency in multimodal generation (e.g., `text-to-speech` and `speech-to-speech` dialogue).

    Qwen3-Omni's work fits into this timeline by pushing the frontier of `multimodal integration` to a point where `modality trade-offs` are claimed to be overcome. It addresses the latency requirements for real-time interaction and introduces specialized components for enhanced `multimodal reasoning` and `audio captioning`, placing it at the cutting edge of truly `omnidirectional multimodal AI`.

## 3.4. Differentiation Analysis
Compared to the main methods in related work, Qwen3-Omni introduces several core differences and innovations:

*   **No Modality Degradation:** The most significant differentiation is the claim that Qwen3-Omni, for the first time, achieves `state-of-the-art performance` across text, image, audio, and video without any degradation relative to single-modal counterparts of the same size. Previous multimodal models often exhibited `modality trade-offs`. This is attributed to a specific training strategy that mixes `unimodal` and `cross-modal data` from the early stage of `text pretraining`.
*   **Upgraded MoE Architecture:** Both the `Thinker` (reasoning/text generation) and `Talker` (speech generation) components are upgraded to `Mixture-of-Experts (MoE)` designs. This enhances `scalability`, `concurrency`, and `inference speed` compared to dense models or prior `MoE` implementations in predecessors like `Qwen2.5-Omni`.
*   **Novel Audio Encoder (AuT):** Qwen3-Omni replaces the `Whisper audio encoder` with its custom-trained `AuT (Audio Transformer)` encoder. `AuT` is trained from scratch on 20 million hours of supervised audio, designed for stronger `general-purpose audio representations`, and employs `block-wise window attention` for `real-time prefetch caching`.
*   **Advanced Streaming Speech Synthesis (Multi-Codebook, Causal ConvNet):**
    *   **Multi-Codebook Representation:** The `Talker` adopts a `multi-codebook` scheme, increasing capacity for faithful modeling of diverse voices and `paralinguistic cues`.
    *   **MTP Module:** Shifts from `single-track` to `multi-track codec modeling`, autoregressively predicting multiple `codebook layers` via `MTP modules`.
    *   **Causal ConvNet for Code2Wav:** Replaces computationally intensive `block-wise diffusion` (used in `Qwen2.5-Omni`) with a lightweight `causal ConvNet` for the `waveform generation` stage (`Code2Wav`). This allows `streaming from the first codec frame` and significantly reduces latency.
    *   **Reduced Code Rates:** Input and output audio `code rates` are reduced to $12.5\mathrm{Hz}$, enabling `single-frame, immediate speech synthesis`. These combined optimizations lead to ultra-low `first-packet latency` (234 ms).
*   **Decoupled Thinker-Talker Functionality:** The `Talker` no longer consumes the `Thinker`'s high-level text representations directly, but conditions on `audio and visual multimodal features`. This allows for independent control of `Thinker`'s response style and `Talker`'s audio style via `distinct system prompts`, and enables intervention by `external modules` on `Thinker`'s textual output before `Talker` processes it.
*   **Explicit Multimodal Reasoning (`Thinking model`):** The introduction of a specific `Thinking model` variant that explicitly reasons over `full-modality inputs` (audio-video, audio-only) strengthens higher-order cognitive functions, which is a step beyond general multimodal understanding.
*   **General-Purpose Audio Captioning (`Captioner`):** The fine-tuned `Qwen3-Omni-30B-A3B-Captioner` addresses a recognized void in the research community for a general-purpose, low-hallucination `audio captioning` model.

    In essence, Qwen3-Omni differentiates itself by not just aggregating modalities but deeply integrating them to overcome historical performance compromises, while simultaneously pushing the boundaries of real-time interactive performance and specialized multimodal reasoning/generation capabilities.

# 4. Methodology

## 4.1. Principles
The core principle behind Qwen3-Omni is to develop a `single multimodal model` that can achieve `state-of-the-art performance` across text, image, audio, and video modalities **without any performance degradation** relative to `same-sized single-modal counterparts`. This is achieved by moving beyond simple concatenation of unimodal encoders and instead focusing on deep, synergistic integration from the earliest stages of training. The theoretical basis is that `joint multimodal training`, particularly by mixing `unimodal` and `cross-modal data` early in `text pretraining`, can foster `mutual enhancement` between modalities rather than trade-offs. This approach aims to mimic human perception where different senses interact and reinforce each other.

Furthermore, a key intuition is that real-time `multimodal interaction` requires not only strong understanding and generation capabilities but also extremely `low latency`. This drives architectural innovations in `speech synthesis` and `concurrency management`. The design also emphasizes decoupling certain components (e.g., `Thinker`'s text output from `Talker`'s direct input) to allow for greater flexibility, control, and modularity in application, while still maintaining an end-to-end training and inference pipeline.

## 4.2. Core Methodology In-depth (Layer by Layer)
Qwen3-Omni employs a `Thinker-Talker architecture`, building on its predecessor Qwen2.5-Omni, but with significant upgrades for scalability, control, and real-time performance.

The overview of Qwen3-Omni is depicted in Figure 2:

![img-1.jpeg](images/2.jpeg)
*Figure 2: The overview of Qwen3-Omni. Qwen3-Omni adopts the Thinker-Talker architecture. Thinker is tasked with text generation while Talker focuses on generating streaming speech tokens by receives high-level representations directly from Thinker. To achieve ultra-low-latency streaming, Talker autoregressively predicts a multi-codebook sequence. At each decoding step, an MTP module outputs the residual codebooks for the current frame, after which the Code2Wav renderer incrementally synthesizes the corresponding waveform, enabling frame-by-frame streaming generation.*

### 4.2.1. Thinker-Talker Architecture Overview
The system is composed of two main components:
*   **Thinker:** This module is responsible for `text generation` and `multimodal reasoning`. It takes various `multimodal inputs` (text, audio, image, video) and processes them to generate textual responses.
*   **Talker:** This module focuses on `generating streaming speech tokens`. It receives high-level representations, potentially from the `Thinker` (after some decoupling), and translates them into natural-sounding speech in real-time.

    Key architectural changes from `Qwen2.5-Omni` include:
1.  Both `Thinker` and `Talker` are upgraded to `Mixture-of-Experts (MoE)` designs, which enhances `scalability` and `inference efficiency`.
2.  The `Talker` no longer directly consumes the `Thinker`'s high-level text representations. Instead, it conditions primarily on `audio and visual multimodal features`. This allows for a `decoupling` where `textual content` (discrete tokens/embeddings) is considered information-equivalent for `textual generation`, while `multimodal conditioning` is used for `audio-video-coordinated speech generation` (e.g., preserving prosody). This decoupling also enables `external modules` (RAG, function calling, safety filters) to intervene on the `Thinker`'s `textual output` before it's passed to the `Talker` for `streaming synthesis`.
3.  Distinct `system prompts` can be used for the `Thinker` (controlling response style) and the `Talker` (controlling audio style).
4.  The `Talker` adopts a `multi-codebook autoregressive scheme` for `speech synthesis`.
5.  The `Code2Wav renderer`, which converts `codec frames` to `waveform`, is implemented as a lightweight `causal ConvNet`, simplifying the final stage of audio synthesis.

    During training and inference, the `Talker` ingests high-dimensional `multimodal features` (presumably from the `Thinker`'s processed inputs or shared representations) and accesses the `full conversational history`. This allows the system to operate as a cohesive `single model` for end-to-end training and unified inference.

### 4.2.2. Audio Transformer (AuT) Encoder
The `AuT (Audio Transformer)` is a newly proposed component that serves as the `audio encoder` for Qwen3-Omni. Its architecture is shown in Figure 3.

![img-2.jpeg](images/3.jpeg)
*Figure 3: The overview of AuT. AuT is an attention-encoder-decoder based auto-regressive model, which is trained from scratch on 20 million hours of supervised audio. Qwen3-Omni employs the AuT encoder as the audio encoder to obtain general purpose audio representations at a token rate of $12.5\mathrm{Hz}$.*

`AuT` is an `attention-encoder-decoder based autoregressive model`.
*   **Training Data:** It's trained from scratch on an extensive dataset of 20 million hours of supervised audio. This data mixture is designed to learn `stronger and more general-purpose audio representations`:
    *   80% Chinese and English `pseudo-labeled ASR` data.
    *   10% `ASR` data from other languages.
    *   10% `audio understanding` data.
*   **Input Processing:** The raw audio `filter bank features` (e.g., `mel-spectrograms`) are `downsampled` 8 times using `Conv2D blocks` before being fed into the `attention layers`. This reduces the `token rate` of the audio representations to $12.5\mathrm{Hz}$. This means each output token from `AuT` corresponds to an 80 ms segment of the original audio signal ($1000 \mathrm{ms} / 12.5 \mathrm{Hz} = 80 \mathrm{ms}$).
*   **Attention Mechanism:** `AuT` utilizes `flash attention` with `dynamic attention window sizes`. This design balances the `efficiency of real-time prefetch caching` with `performance for offline audio tasks`, accommodating attention query patterns ranging from 1 to 8 seconds.
*   **Parameters:** The `AuT encoder` component contains approximately 0.6 billion parameters.

### 4.2.3. Perception Modules
The `Thinker` processes various inputs (Text, Audio, Image, Video) by converting them into unified representations.

*   **Text Inputs:**
    *   Utilizes Qwen's `tokenizer` (Yang et al., 2025a).
    *   Employs `byte-level byte-pair encoding (BPE)`.
    *   Has a vocabulary of 151,643 regular tokens.
*   **Audio Inputs (and Audio extracted from Video):**
    *   Raw waveform is `resampled` to $16\mathrm{kHz}$.
    *   Converted into a 128-channel `mel-spectrogram` using a $25\mathrm{ms}$ window and a $10\mathrm{ms}$ hop.
    *   The `AuT encoder` (described above) is then used to generate audio representations, where each frame corresponds to approximately an 80 ms segment of the original audio signal.
*   **Image and Video Inputs:**
    *   Uses the `vision encoder` from `Qwen3-VL`.
    *   This encoder is initialized from `SigLIP2-So400m` (Tschannen et al., 2025) and has approximately 543 million parameters.
    *   It is trained on a mixture of image and video data to ensure `strong image understanding` and `video comprehension`.
    *   For video, `frames` are sampled at a `dynamic frame rate` to preserve information comprehensively while aligning with the audio sampling rate.

### 4.2.4. Video and Multimodal Position Embedding (TM-RoPE)
Qwen3-Omni introduces `Time-aligned Multimodal Rotary Position Embedding (TM-RoPE)`, an extension of `Multimodal Rotary Position Embedding (M-RoPE)` (Bai et al., 2023b). `TM-RoPE` integrates absolute temporal information to handle multimodal audiovisual streams more effectively.

*   **Factorization of RoPE:** `TM-RoPE` factorizes the conventional `Rotary Position Embedding` into three distinct dimensions: `temporal`, `height`, and `width`.
*   **Angle Redistribution:** Unlike the original `M-RoPE` where initial 16 `rotary angles` (high frequencies) modeled temporal dependencies, `TM-RoPE` redistributes these angles for a more balanced representation: 24 `rotary angles` for temporal, 20 for height, and 20 for width dimensions. This aims to better capture both `local semantics` and `long-range dependencies`, improving extrapolation over extended sequences.
*   **Modality-Specific Application:**
    *   **Text Inputs:** All three components (temporal, height, width) share identical `position identifiers`, making `TM-RoPE` functionally equivalent to a `one-dimensional RoPE` (Su et al., 2024).
    *   **Audio Inputs:** Utilize shared `position IDs` but are augmented with `absolute temporal encodings`. Each `temporal ID` corresponds to a duration of 80 ms.
    *   **Image Data:** A constant `temporal ID` is assigned to all `visual tokens`. Their distinct row and column positions determine the `height and width IDs`.
    *   **Multimodal Audiovisual Streams:**
        *   **Audio Component:** Encoded with a `temporal ID` for every 80 ms.
        *   **Video Component:** Treated as a sequence of frames with `monotonically increasing temporal IDs`. These IDs are dynamically adjusted based on actual timestamps to ensure a consistent `temporal resolution` of 80 ms per ID.
        *   `Height` and `width IDs` for `video frames` are assigned identically to still images.
*   **Contiguous Position Numbering:** To prevent `positional conflicts` when processing multiple modalities, `position numbering` is made `contiguous`. Each subsequent modality starts from one plus the maximum `position ID` of the preceding modality.
*   **Direct Temporal Alignment:** Unlike `Qwen2.5-Omni` which segmented `audiovisual representations` into fixed 2-second chunks, `Qwen3-Omni` directly aligns these representations using their `temporal IDs`, explicitly anchored to `absolute time`. This provides flexibility for `streaming inputs of arbitrary duration`.

### 4.2.5. Speech Generation (Talker Module)
The `Talker` module is central to generating speech responses.
*   **Context Conditioning:** It is conditioned on a rich context inherited from the `Thinker` component, including:
    *   Historical `textual tokens`.
    *   `Multimodal representations`.
    *   The current turn's `streamed text`.
        This `long-context information` is critical for `high-fidelity speech synthesis` to adapt acoustic attributes (prosody, loudness, emotion) to the ongoing discourse.
*   **Operation on RVQ Tokens:** The `Talker` operates directly on `RVQ (Residual Vector Quantization) tokens`. RVQ is a quantization method used in audio codecs to represent a signal efficiently. It quantizes the residual error from a previous quantization step, enabling progressive refinement and a hierarchy of codebooks.
*   **Hierarchical Prediction Scheme:**
    *   The `backbone` of the `Talker` ingests `aggregated codebook features` of the current frame.
    *   It uses a `linear head` to predict the `zeroth codebook` (the base layer of RVQ).
    *   Subsequently, a `multi-token prediction (MTP) module` generates all `residual codebooks` for that frame. This strategy allows the model to learn a complete representation of acoustic details, enhancing vocal expressivity.
*   **Lightweight Waveform Reconstruction (Code2Wav):** The `waveform reconstruction` is simplified to a lightweight `causal ConvNet` (`Code2Wav`). This significantly reduces `inference latency` and `computational cost (FLOPs)` while achieving superior audio fidelity compared to more complex `DiT-based vocoders` (presumably Diffusion Transformer-based vocoders used in prior work).

### 4.2.6. Designs for Streaming and Concurrency
Qwen3-Omni incorporates several algorithmic and architectural optimizations to achieve `low first-packet latency` and `high concurrency`.

*   **Chunked Prefilling and MoE Architecture:**
    *   **Chunked Prefilling:** Retained from `Qwen2.5-Omni`. `Audio` and `vision encoders` output chunks along the temporal dimension.
    *   **Asynchronous Prefilling:** During real-time interaction, `Thinker` and `Talker` perform `asynchronous prefilling`. When `Thinker` completes prefilling a chunk, its high-level representations are immediately used to prefill the `Talker`'s current chunk asynchronously, while `Thinker` proceeds to its next chunk. This reduces `Time-To-First-Token (TTFT)` for both.
    *   **MoE Architecture:** Both `Thinker` and `Talker` adopt `MoE designs`. This is highly effective for improving `service throughput` by reducing `IO consumption` from the `KV cache` during long sequence processing, thereby increasing `tokens per second (TPS)` and enhancing `concurrency`.

*   **Streaming Multi-Codebook Codec Generation:**
    *   To minimize `first-packet latency`, a `left-context only multi-codebook generation mechanism` is proposed.
    *   Once the `Talker` generates the first token, the `MTP module` predicts the remaining tokens for the current frame.
    *   These tokens are then decoded into `waveform` by a `streaming multi-codebook codec decoder` that only attends to the `left context`.
    *   Unlike `Qwen2.5-Omni` (which required waiting for sufficient `block-context` from the `Talker`), `Qwen3-Omni` can output the `waveform immediately` after the `Talker` generates each token, significantly reducing `first-packet latency`.

*   **Lightweight MTP module and ConvNet:**
    *   **MTP Module:** This module is an `ultra-lightweight fixed-step autoregressive dense transformer`. It has `low computational FLOPs` and supports `batched inference`, making it suitable for `high-concurrency scenarios`. Its `fixed-step autoregressive inference mechanism` efficiently leverages a `fixed KV cache memory space` for acceleration, achieving `low inference latency`.
    *   **Codec Decoder (Code2Wav):** The `ConvNet-based codec decoder` also achieves `high throughput` with `low latency`. Its convolutional architecture enjoys `extensive hardware acceleration support` and enables `efficient batched inference`.

        The overall architecture design and end-to-end first-packet latency are summarized in Table 1.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th>Module</th>
<th>Architecture</th>
<th>Params</th>
<th>Streaming</th>
</tr>
</thead>
<tbody>
<tr>
<td>Audio Encoder</td>
<td>AuT</td>
<td>650M</td>
<td>✓</td>
</tr>
<tr>
<td>Vision Encoder</td>
<td>SigLIP2-So400M</td>
<td>540M</td>
<td>-</td>
</tr>
<tr>
<td>Thinker</td>
<td>MoE Transformer</td>
<td>30B-A3B</td>
<td>✓</td>
</tr>
<tr>
<td>Talker</td>
<td>MoE Transformer</td>
<td>3B-A0.3B</td>
<td>✓</td>
</tr>
<tr>
<td>MTP</td>
<td>Dense Transformer</td>
<td>80M</td>
<td>✓</td>
</tr>
<tr>
<td>Code2wav</td>
<td>ConvNet</td>
<td>200M</td>
<td>✓</td>
</tr>
<tr>
<td colspan="4">End-to-End First-Packet Latency: 234/547ms</td>
</tr>
</tbody>
</table>

This table details the `Qwen3-Omni-30B-A3B` model's components, their `parameter counts`, and their `streaming capabilities`. The `Thinker` (MoE Transformer) is the largest component at 30 billion active parameters (3B-A3B likely denotes 30 billion total parameters with 3 billion active per token or similar), while the `Talker` is significantly smaller. The `AuT` and `Vision Encoder` are substantial perception modules. The `MTP` module and `Code2Wav` are relatively lightweight. The theoretical `end-to-end first-packet latency` is stated as 234 ms for audio and 547 ms for video.

The theoretical `first-packet latency` under different `concurrency` levels is presented in Table 2.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th colspan="4">Qwen3-Omni-30B-A3B</th>
</tr>
<tr>
<th></th>
<th>1Concurrency</th>
<th>4Concurrency</th>
<th>6Concurrency</th>
</tr>
</thead>
<tbody>
<tr>
<td>Thinker-Talker Tail Packet Preprocessing Latency</td>
<td>72/160ms</td>
<td>94/180ms</td>
<td>100/200ms</td>
</tr>
<tr>
<td>Thinker Time-to-First-Token (TTPT)</td>
<td>88/160ms</td>
<td>468/866ms</td>
<td>673/1330ms</td>
</tr>
<tr>
<td>Talker Time-to-First-Token (TTPT)</td>
<td>57/210ms</td>
<td>145/450ms</td>
<td>376/734ms</td>
</tr>
<tr>
<td>MTP Module Time Cost Per Token</td>
<td>14ms</td>
<td>16ms</td>
<td>18ms</td>
</tr>
<tr>
<td>Codec Decoder Time Cost Per Code</td>
<td>3ms</td>
<td>5ms</td>
<td>5ms</td>
</tr>
<tr>
<td>Overall Latency (Audio/Video)</td>
<td>234/547ms</td>
<td>728/1517ms</td>
<td>1172/2284ms</td>
</tr>
<tr>
<td>Thinker Token Generation Rate (TPS)</td>
<td>75 tokens/s</td>
<td>63 tokens/s</td>
<td>53 tokens/s</td>
</tr>
<tr>
<td>Talker Token Generation Rate (TPS)</td>
<td>140 tokens/s</td>
<td>125 tokens/s</td>
<td>110 tokens/s</td>
</tr>
<tr>
<td>Generation RTF(Real Time Factor)</td>
<td>0.47</td>
<td>0.56</td>
<td>0.66</td>
</tr>
</tbody>
</table>

This table showcases the `theoretical first-packet latency` (in milliseconds) and `token generation rates` (tokens per second) for `Qwen3-Omni` under varying `concurrency` levels (1, 4, 6 concurrent requests). The `MoE architecture` of `Thinker` and `Talker` helps keep their preprocessing latency and `TTPT` manageable even under `high concurrency`. The lightweight nature of the `MTP Module` and `Codec Decoder` minimizes their impact on latency. The `Generation Real Time Factor (RTF)` consistently remains below 1, indicating that the model can generate audio faster than real-time even under concurrency. The latency numbers are split for audio/video where applicable (e.g., $72/160ms$ means 72ms for audio, 160ms for video).

# 5. Experimental Setup

## 5.1. Datasets
Qwen3-Omni's training and evaluation rely on a diverse array of datasets for its `multimodal capabilities`.

### 5.1.1. Training Datasets
The pre-training phase utilizes a large-scale, diverse dataset encompassing multiple languages, dialects, and modalities:
*   **Image-Text Pairs**
*   **Video-Text Pairs**
*   **Audio-Text Pairs**
*   **Video-Audio Pairs**
*   **Video-Audio-Text Pairs**
*   **Pure Text Corpora**

    The second pre-training phase (General Stage, S2) alone uses approximately 2 trillion tokens, distributed as:
*   Text: 0.57 trillion tokens
*   Audio: 0.77 trillion tokens
*   Image: 0.82 trillion tokens
*   Video: 0.05 trillion tokens
*   Video-Audio: 0.05 trillion tokens

    The `AuT encoder` specifically is trained on 20 million hours of supervised audio data, comprising:
*   80% Chinese and English `pseudo-labeled ASR` data
*   10% `ASR` data from other languages
*   10% `audio understanding` data

    For post-training, the `Thinker` uses data structured in `ChatML` (OpenAI, 2022) format, including:
*   Pure text-based dialogue data
*   Visual modality conversation data
*   Audio modality conversation data
*   Mixed-modality conversation data

    The `Talker` is trained on `hundreds of millions of speech data with multimodal context` and later `high-quality data` for `Continual Pretraining (CPT)`. `Direct Preference Optimization (DPO)` uses `preference pairs from diverse multilingual speech samples`.

The `Qwen3-Omni-30B-A3B-Captioner` is fine-tuned on a `large-scale dataset of detailed audio descriptions`.

### 5.1.2. Evaluation Datasets
The paper uses an extensive suite of benchmarks, categorized by modality input and output.

**Text $\rightarrow$ Text Evaluation:**
*   **General Tasks:** `MMLU-Redux` (Gema et al., 2024), `GPQA` (Rein et al., 2023).
*   **Reasoning:** `AIME25` (AIME, 2025), `ZebraLogic` (Lin et al., 2025), `LiveBench 20241125`.
*   **Coding:** `MultiPL-E` (Cassano et al., 2023).
*   **Alignment Tasks:** `IFEval` (Zhou et al., 2023), `Creative Writing V3` (Paech, 2024), `WritingBench` (Wu et al., 2025b), `Arena-Hard v2`.
*   **Agent:** `BFCL-v3` (Yan et al., 2024).
*   **Multilingual Tasks:** `MultiIF` (He et al., 2024), `PolyMath` (Wang et al., 2025c).

**Audio $\rightarrow$ Text Evaluation:**
*   **ASR & S2TT (Automatic Speech Recognition & Speech-to-Text Translation):**
    *   `Wenetspeech net` and `meeting` subsets for English and Chinese.
    *   `Librispeech clean` and `other` subsets for English.
    *   `CommonVoice (CV15-en, CV15-zh)` for English and Chinese.
    *   `Fleurs-en, Fleurs-zh` for English and Chinese.
    *   `Fleurs-avg (19 lang)` for multilingual ASR.
    *   `MIR-1K (vocal-only)` and `Opencpop-test` for Lyric ASR.
    *   `Fleurs-en2xx`, `Fleurs-xx2en`, `Fleurs-zh2xx`, `Fleurs-xx2zh` for S2TT (across 15 languages: Arabic, Cantonese, Chinese, English, French, German, Indonesian, Italian, Japanese, Korean, Portuguese, Russian, Spanish, Thai, Vietnamese).
*   **Voice Chatting:** `VoiceBench` (Chen et al., 2024b), which includes sub-benchmarks like `AlpacaEval`, `CommonEval`, `WildVoice`, `SD-QA`, `MMSU`, `OpenBookQA`, `BBH`, `IFEval`, `AdvBench`.
*   **Audio Reasoning:** `MMAU` (Sakshi et al., 2024), `MMSU` (Wang et al., 2025a).
*   **Music Understanding:** `RUL-MuchoMusic` (Zang et al., 2025), `GTZAN` (Tzanetakis and Cook, 2002), `MTG-Jamendo` (4 subsets: Genre, Mood/Theme, Instrument, Top50) (Bogdanov et al. (2019)), `MagnaTagATune` (Law et al., 2009).

**Vision $\rightarrow$ Text Evaluation:**
*   **General Visual Question Answering:** `MMStar` (Chen et al., 2024a), `HallusionBench` (Guan et al., 2024), `MM-MT-Bench` (Agrawal et al., 2024).
*   **Mathematical & STEM Reasoning:** `MathVista` (Lu et al., 2024), `MathVision` (Wang et al., 2024a), `MMMU` (Yue et al., 2023), `MMMU-Pro` (Yue et al., 2024).
*   **Document Understanding:** `AI2D` (Kembhavi et al., 2016), `ChartQA` (Masry et al., 2022).
*   **Counting:** `CountBench` (Paiss et al., 2023).
*   **Long Video Understanding:** `Video-MME` (Fu et al., 2024), `LVBench` (Wang et al., 2024b), `MLVU` (Zhou et al., 2025a).

**AudioVisual Video $\rightarrow$ Text Evaluation:**
*   **General Understanding:** `WorldSense` (Hong et al., 2025).
*   **Audiovisual Reasoning:** `DailyOmni` (Zhou et al., 2025b), `VideoHolmes` (Cheng et al., 2025).

**X $\rightarrow$ Speech Evaluation:**
*   **Zero-Shot Speech Generation:** `SEED` (Anastassiou et al., 2024) (`test-zh`, `test-en` subsets).
*   **Multilingual Speech Generation:** `MiniMax multilingual test set` (Zhang et al., 2025).
*   **Cross-Lingual Speech Generation:** `CV3-Eval` (Du et al., 2025).

    These datasets were chosen for their effectiveness in validating specific aspects of the model's performance, ranging from basic perception (ASR, image recognition) to complex reasoning (multimodal Q&A, audio reasoning) and generation quality (speech synthesis, captioning), across diverse languages and content types.

## 5.2. Evaluation Metrics

### 5.2.1. Word Error Rate (WER)
*   **Conceptual Definition:** `Word Error Rate (WER)` is a common metric for measuring the performance of `Automatic Speech Recognition (ASR)` or `Speech-to-Text (STT)` systems. It quantifies the number of errors (substitutions, insertions, and deletions) needed to transform the recognized word sequence into the reference word sequence. A lower `WER` indicates better performance.
*   **Mathematical Formula:**
    \$
    \mathrm{WER} = \frac{S + D + I}{N}
    \$
*   **Symbol Explanation:**
    *   $S$: Number of `substitutions` (a word in the reference is replaced by a different word in the hypothesis).
    *   $D$: Number of `deletions` (a word in the reference is omitted in the hypothesis).
    *   $I$: Number of `insertions` (a word in the hypothesis is not present in the reference).
    *   $N$: Total number of words in the `reference` sequence.

### 5.2.2. BLEU Score
*   **Conceptual Definition:** `BLEU (Bilingual Evaluation Understudy)` score is a metric primarily used for evaluating the quality of text generated by `machine translation` systems. It measures the `n-gram precision` of the generated text against one or more reference translations, penalizing for brevity. Higher `BLEU` scores indicate better quality.
*   **Mathematical Formula:**
    \$
    \mathrm{BLEU} = \mathrm{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
    \$
    where
    \$
    \mathrm{BP} = \begin{cases}
    1 & \text{if } c > r \\
    e^{(1 - r/c)} & \text{if } c \le r
    \end{cases}
    \$
*   **Symbol Explanation:**
    *   $\mathrm{BP}$: `Brevity Penalty`, which penalizes hypotheses that are too short compared to the reference.
    *   $c$: Length of the `candidate` translation (generated text).
    *   $r$: Length of the `reference` translation.
    *   $N$: Maximum `n-gram` order considered (typically 4, meaning unigrams, bigrams, trigrams, and 4-grams are evaluated).
    *   $w_n$: Weight for each `n-gram` precision (often uniform, $w_n = 1/N$).
    *   $p_n$: `n-gram precision`, calculated as the number of matching `n-grams` in the candidate that also appear in the reference (clipped to the maximum count in the reference), divided by the total number of `n-grams` in the candidate.

### 5.2.3. Micro F1 Score
*   **Conceptual Definition:** `Micro F1 Score` is a metric used for evaluating classification models, particularly in `multi-label classification` or `multi-class classification` where class imbalance might exist. It calculates `F1-score` globally by counting the total true positives, false negatives, and false positives across all classes or instances. This metric is suitable when performance across all classes is equally important.
*   **Mathematical Formula:**
    \$
    \text{Micro-Precision} = \frac{\sum_{i=1}^C \text{TP}_i}{\sum_{i=1}^C (\text{TP}_i + \text{FP}_i)}
    \$
    \$
    \text{Micro-Recall} = \frac{\sum_{i=1}^C \text{TP}_i}{\sum_{i=1}^C (\text{TP}_i + \text{FN}_i)}
    \$
    \$
    \text{Micro-F1} = 2 \cdot \frac{\text{Micro-Precision} \cdot \text{Micro-Recall}}{\text{Micro-Precision} + \text{Micro-Recall}}
    \$
*   **Symbol Explanation:**
    *   $\text{TP}_i$: Number of `True Positives` for class $i$.
    *   $\text{FP}_i$: Number of `False Positives` for class $i$.
    *   $\text{FN}_i$: Number of `False Negatives` for class $i$.
    *   $C$: Total number of classes.
    *   `Micro-Precision`: Precision calculated globally.
    *   `Micro-Recall`: Recall calculated globally.
    *   `Micro-F1`: The harmonic mean of `Micro-Precision` and `Micro-Recall`.

### 5.2.4. Speaker Similarity (SIM)
*   **Conceptual Definition:** `Speaker Similarity (SIM)` is a metric used to evaluate how well a synthesized voice matches the `timbre` and `characteristics` of a target speaker's voice. It's often quantified by comparing `speaker embeddings` (vector representations of voice characteristics) from the synthesized speech and the target reference speech. A higher `SIM` score indicates a better match. The paper refers to `SIM` in the context of `zero-shot speech generation` and `multilingual speech generation`. While the exact internal calculation isn't provided, it typically involves cosine similarity between `speaker embeddings`.
*   **Mathematical Formula (Commonly used, not explicitly in paper):**
    \$
    \mathrm{SIM}(E_s, E_t) = \frac{E_s \cdot E_t}{\|E_s\| \|E_t\|}
    \$
*   **Symbol Explanation:**
    *   $E_s$: `Speaker embedding` of the synthesized speech.
    *   $E_t$: `Speaker embedding` of the target (reference) speech.
    *   $\cdot$: Dot product of the two `embedding vectors`.
    *   $\|\cdot\|$: L2 norm (magnitude) of the `embedding vector`.
    *   The formula represents the `cosine similarity`, which measures the cosine of the angle between two non-zero vectors. A value of 1 means identical direction (perfect similarity), and -1 means completely opposite.

## 5.3. Baselines
Qwen3-Omni is compared against a wide range of `leading specialist` and `generalist models`, including both `open-source` and `closed-source systems`.

**For Text $\rightarrow$ Text:**
*   **Instruct Models:** `GPT-4o-0327` (closed-source), `Qwen3-235B-A22B Non Thinking`, `Qwen3-30B-A3B-Instruct-2507` (text-only counterpart).
*   **Thinking/Reasoning Models:** `Gemini-2.5-Flash Thinking` (closed-source), `Qwen3-235B-A22B Thinking`, `Qwen3-30B-A3B-Thinking-2507` (text-only counterpart), `InternVL-3.5-241B-A28B`.

**For Audio $\rightarrow$ Text (ASR & S2TT):**
*   `Seed-ASR` (specialist `ASR` model).
*   `Voxtral-Mini`, `Voxtral-Small` (specialist `ASR` models).
*   `GPT-4o-Transcribe` (closed-source `ASR` from `GPT-4o`).
*   `Gemini-2.5-Pro` (closed-source generalist).
*   `Qwen2.5-Omni` (previous version of Qwen-Omni).

**For Audio $\rightarrow$ Text (Voice Interaction & Audio Reasoning):**
*   `GPT-4o-Audio` (closed-source generalist).
*   `Gemini-2.5-Flash`, `Gemini-2.5-Pro` (closed-source generalists).
*   `Qwen2.5-Omni` (previous version).

**For Audio $\rightarrow$ Text (Music Understanding):**
*   `Best Specialist Models`: `Audio Flamingo 3` (Goel et al., 2025), `CLaMP 3` (Wu et al., 2025a), `MuQ-MuLan` (Zhu et al., 2025), `MuQ` (Zhu et al., 2025). These are highly specialized `music information retrieval` models.
*   `GPT-4o-Audio` (closed-source generalist).
*   `Gemini-2.5-Pro` (closed-source generalist).
*   `Qwen2.5-Omni` (previous version).

**For Vision $\rightarrow$ Text:**
*   `GPT-4o` (closed-source generalist).
*   `Gemini-2.0-Flash`, `Gemini-2.5-Flash Thinking` (closed-source generalists).
*   $Qwen2.5-VL 72B$ (previous `vision-language model` from Qwen series).
*   `InternVL-3.5-241B-A28B`.

**For AudioVisual Video $\rightarrow$ Text:**
*   `Previous Open-source SoTA` (specific models cited in tables, e.g., Yang et al., 2025b for `WorldSense`; Tang et al., 2025 for `DailyOmni` and `VideoHolmes`).
*   `Gemini-2.5-Flash`, `Gemini-2.5-Flash Thinking` (closed-source generalists).
*   `Qwen2.5-Omni` (previous version).

**For X $\rightarrow$ Speech (Speech Generation):**
*   `Seed-TTSICL`, `Seed-TTSRL` (Anastassiou et al., 2024).
*   `MaskGCT` (Wang et al., 2024c).
*   `E2 TTS` (Eskimez et al., 2024).
*   `F5-TTS` (Chen et al., 2024c).
*   `Spark TTS` (Wang et al., 2025b).
*   `CosyVoice 2` (Du et al., 2024), `CosyVoice 3` (Du et al., 2025).
*   `Qwen2.5-Omni-7B` (Xu et al., 2025).
*   `MiniMax` (Zhang et al., 2025), `ElevenLabs Multilingual v2` (for multilingual generation).

    These baselines are representative as they include:
*   **Strong closed-source models:** `GPT-4o`, `Gemini` variants, which are often considered industry benchmarks.
*   **State-of-the-art open-source models:** Other `Qwen` models (including unimodal and previous multimodal versions), specialist models like `Seed-ASR`, `Voxtral`, `Audio Flamingo`, `MuQ` for specific tasks.
*   **Direct predecessors:** `Qwen2.5-Omni` and `Qwen2.5-VL` to demonstrate iterative improvements.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Performance of Text $\rightarrow$ Text
The evaluation of `Text`\rightarrow`Text` performance assesses `Qwen3-Omni`'s capabilities in general tasks, reasoning, coding, alignment, agent, and multilingual tasks.

The following are the results from Table 4 of the original paper:

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
<td>GeneralTasks</td>
<td>MMLU-Redux</td>
<td>91.3</td>
<td>89.2</td>
<td>89.3</td>
<td>86.6</td>
<td>86.8</td>
</tr>
<tr>
<td></td>
<td>GPQA</td>
<td>66.9</td>
<td>62.9</td>
<td>70.4</td>
<td>69.6</td>
<td>69.7</td>
</tr>
<tr>
<td>Reasoning</td>
<td>AIME25</td>
<td>26.7</td>
<td>24.7</td>
<td>61.3</td>
<td>65.0</td>
<td>65.9</td>
</tr>
<tr>
<td></td>
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
<td>Alignment Tasks</td>
<td>IFEval</td>
<td>83.9</td>
<td>83.2</td>
<td>84.7</td>
<td>81.0</td>
<td>81.7</td>
</tr>
<tr>
<td></td>
<td>Creative Writing v3</td>
<td>84.9</td>
<td>80.4</td>
<td>86.0</td>
<td>80.6</td>
<td>81.8</td>
</tr>
<tr>
<td></td>
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
<td>Multilingual Tasks</td>
<td>MultiIF</td>
<td>70.4</td>
<td>70.2</td>
<td>67.9</td>
<td>64.0</td>
<td>64.7</td>
</tr>
<tr>
<td></td>
<td>PolyMATH</td>
<td>25.5</td>
<td>27.0</td>
<td>43.1</td>
<td>37.9</td>
<td>39.3</td>
</tr>
</tbody>
</table>

*Table 4: Text $\rightarrow$ Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.*

As shown in Table 4, the `Qwen3-Omni-30B-A3B-Instruct` model, despite being smaller in parameter count, surpasses the performance of the much larger `Qwen3-235B-A22B Non Thinking` and even the strong closed-source `GPT-4o-0327` on several benchmarks, including `GPQA`, `AIME25`, `ZebraLogic`, `WritingBench`, and `PolyMath`. This highlights its efficiency and strong `instruction-following` capabilities. However, it shows a slight dip in `MMLU-Redux` compared to `Qwen3-30B-A3B-Instruct-2507` (its text-only counterpart). The `Flash` variant (optimized for speed/efficiency) generally performs very similarly to the main `Instruct` model.

The following are the results from Table 5 of the original paper:

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
<td>General Tasks</td>
<td>MMLU-Redux</td>
<td>92.1</td>
<td>92.7</td>
<td>91.4</td>
<td>88.8</td>
<td>89.7</td>
</tr>
<tr>
<td></td>
<td>GPQA</td>
<td>82.8</td>
<td>71.1</td>
<td>73.4</td>
<td>73.1</td>
<td>73.1</td>
</tr>
<tr>
<td>Reasoning</td>
<td>AIME25</td>
<td>72.0</td>
<td>81.5</td>
<td>85.0</td>
<td>73.7</td>
<td>74.0</td>
</tr>
<tr>
<td></td>
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
<td>Alignment Tasks</td>
<td>IFEval</td>
<td>89.8</td>
<td>83.4</td>
<td>88.9</td>
<td>85.1</td>
<td>85.2</td>
</tr>
<tr>
<td></td>
<td>Arena-Hard v2</td>
<td>56.7</td>
<td>61.5</td>
<td>56.0</td>
<td>55.1</td>
<td>57.8</td>
</tr>
<tr>
<td></td>
<td>Creative Writing v3</td>
<td>85.0</td>
<td>84.6</td>
<td>84.4</td>
<td>82.5</td>
<td>83.6</td>
</tr>
<tr>
<td></td>
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
<td>Multilingual Tasks</td>
<td>MultiIF</td>
<td>74.4</td>
<td>71.9</td>
<td>76.4</td>
<td>72.9</td>
<td>73.2</td>
</tr>
<tr>
<td></td>
<td>PolyMATH</td>
<td>49.8</td>
<td>54.7</td>
<td>52.6</td>
<td>47.1</td>
<td>48.7</td>
</tr>
</tbody>
</table>

*Table 5: Text $\rightarrow$ Text performance of Qwen3-Omni-Thinking and other reasoning baselines. The highest scores are shown in bold.*

Table 5 shows the performance of the `Qwen3-Omni-30B-A3B-Thinking` variant. It demonstrates performance comparable to `Gemini-2.5-Flash-Thinking` and `Qwen3-235B-A22B Non-Thinking`, and also to its text-only counterpart, `Qwen3-30B-A3B-Thinking-2507`. While it doesn't consistently lead in all benchmarks, its competitive scores suggest that the multimodal integration (even with a specific `Thinking` architecture) does not degrade core `textual reasoning capabilities`.

### 6.1.2. Performance of Audio $\rightarrow$ Text
This section evaluates `Qwen3-Omni`'s ability to understand various audio inputs and generate textual responses, covering `ASR`, `S2TT`, `voice-chatting`, `audio reasoning`, and `music understanding`.

The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
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
<td colspan="9">EN & ZH ASR (wer)</td>
</tr>
<tr>
<td>Wenetspeech net</td>
<td rowspan="2">4.66</td>
<td rowspan="2">5.69</td>
<td>24.30</td>
<td>31.53</td>
<td>20.33</td>
<td>26.08</td>
<td>15.30</td>
<td>14.43</td>
</tr>
<tr>
<td>meeting</td>
<td>13.47</td>
<td>5.91</td>
<td>7.65</td>
<td>4.69</td>
<td>5.89</td>
<td>4.62</td>
</tr>
<tr>
<td>Librispeech clean</td>
<td rowspan="2">1.58</td>
<td rowspan="2">2.84</td>
<td>1.88</td>
<td>4.12</td>
<td>1.56</td>
<td>3.30</td>
<td>1.39</td>
<td>1.27</td>
</tr>
<tr>
<td>other</td>
<td>3.75</td>
<td>2.89</td>
<td>3.56</td>
<td>1.74</td>
<td>3.45</td>
<td>2.48</td>
<td>2.44</td>
</tr>
<tr>
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
<td colspan="9">Multilingual ASR (wer)</td>
</tr>
<tr>
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
<td colspan="9">Lyric ASR (wer)</td>
</tr>
<tr>
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
<td colspan="9">S2TT (BLEU)</td>
</tr>
<tr>
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

*Table 6: Transcription performance for Audio→Text tasks (ASR & S2TT), comparing Qwen3-Omni-Instruct with the baselines. The highest scores are shown in bold.*

Table 6 demonstrates that `Qwen3-Omni-Instruct` achieves `state-of-the-art (SOTA)` performance across numerous `EN & ZH ASR` benchmarks (Wenetspeech, Librispeech, CV15, Fleurs) and `Lyric ASR` (MIR-1K, Opencpop-test), often outperforming specialist models like `Seed-ASR` and closed-source generalists like `GPT-4o-Transcribe` and `Gemini-2.5-Pro`. For `Multilingual ASR` and `S2TT`, it shows comparable or superior performance to competitors. This confirms its strong capabilities in `speech recognition` and `speech translation`.

The following are the results from Table 7 of the original paper:

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

*Table 7: Voice interaction and audio reasoning performance for Audio→Text tasks, comparing Qwen3-Omni with the baselines. The highest scores are shown in bold.*

For voice interaction (`VoiceBench`), `Qwen3-Omni-Thinking` achieves an impressive overall score of 89.5, very close to `Gemini-2.5-Pro` (89.6), and significantly outperforming `GPT-4o-Audio` (86.8) and `Qwen2.5-Omni` (73.6). Notably, the `Thinking` variant performs better than the `Instruct` variant on `VoiceBench`, especially on `MMSU`, `OpenBookQA`, and `BBH` sub-tasks, indicating the value of its reasoning capabilities in interactive speech scenarios. In `Audio Reasoning` benchmarks (`MMAU` and `MMSU`), `Qwen3-Omni-Instruct` and `Thinking` variants perform strongly, often surpassing `Gemini-2.5-Pro` and `Gemini-2.5-Flash`, demonstrating robust `general audio understanding` and `reasoning`.

The following are the results from Table 8 of the original paper:

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
<td>GTZAN</td>
<td>87.9 (CLaMP 3) (Wu et al., 2025a)</td>
<td>76.5</td>
<td>81.0</td>
<td>81.7</td>
<td>93.0</td>
<td>93.1</td>
</tr>
<tr>
<td>MTC Genre</td>
<td>35.8 (MuQ-MuLan) (Zhu et al., 2025)</td>
<td>25.3</td>
<td>32.6</td>
<td>32.5</td>
<td>39.0</td>
<td>39.5</td>
</tr>
<tr>
<td>MTCMood/Theme</td>
<td>10.9 (MuQ-MuLan) (Zhu et al., 2025)</td>
<td>11.3</td>
<td>14.1</td>
<td>8.9</td>
<td>21.0</td>
<td>21.7</td>
</tr>
<tr>
<td>MTCInstrument</td>
<td>39.8 (MuQ-MuLan) (Zhu et al., 2025)</td>
<td>34.2</td>
<td>33.0</td>
<td>22.6</td>
<td>40.5</td>
<td>40.7</td>
</tr>
<tr>
<td>MTCTop50</td>
<td>33.2 (MuQ-MuLan) (Zhu et al., 2025)</td>
<td>25.0</td>
<td>26.1</td>
<td>21.6</td>
<td>36.7</td>
<td>36.9</td>
</tr>
<tr>
<td>MagnaTagATune</td>
<td>41.6 (MuQ) (Zhu et al., 2025)</td>
<td>29.2</td>
<td>28.1</td>
<td>30.1</td>
<td>44.3</td>
<td>46.8</td>
</tr>
</tbody>
</table>

*Table 8: Music understanding performance for Audio→Text tasks, comparing Qwen3-Omni-Instruct with baselines. The highest scores are shown in bold.*

For `music understanding` (Table 8), `Qwen3-Omni-Instruct` achieves `SOTA` on `RUL-MuchoMusic` and significantly outperforms other `audio language models` (like `Gemini-2.5-Pro` and `GPT-4o-Audio`), and even dedicated `self-supervised music specialist models`, on benchmarks like `GTZAN`, `MTG-Jamendo`, and `MagnaTagATune`. This indicates superior `music information retrieval capabilities`.

### 6.1.3. Performance of Vision $\rightarrow$ Text
This section assesses `Qwen3-Omni`'s ability to process visual inputs (images and videos) and generate textual responses.

The following are the results from Table 9 of the original paper:

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
<td colspan="6">Math & STEM</td>
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

*Table 9: Vision $\rightarrow$ Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.*

Table 9 indicates that `Qwen3-Omni-Instruct` demonstrates comparable performance to `Qwen2.5-VL-72B` (a larger, vision-focused model) and achieves better results on `Math & STEM` tasks like `MMMU-Pro overall`, `MathVista mini`, and `MATH-Visionfull` than `GPT-4o` and `Gemini-2.0-Flash`. This highlights its excellent `image understanding` and `reasoning` capabilities. For `video understanding`, its performance on `LVBench` and `MLVU` is competitive, though `Video-MME` shows a slight dip compared to `Qwen2.5-VL`.

The following are the results from Table 10 of the original paper:

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
<td colspan="5">Math & STEM</td>
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

*Table 10: Vision $\rightarrow$ Text performance of Qwen3-Omni-Thinking and other reasoning baselines. The highest scores are shown in bold.*

Table 10 evaluates `Qwen3-Omni-Thinking` against other `state-of-the-art reasoning models`. It shows significant advancements, outperforming the `Qwen3-Omni-Instruct` baseline by 4.4 points on `Math and STEM benchmarks`. It achieves performance comparable to `substantially larger baselines` (`InternVL-3.5-241B-A28B`), demonstrating good balance of effectiveness and computational efficiency. A limitation noted is `suboptimal performance on long video benchmarks` (`Video-MME`, `LVBench`, `MLVU`), attributed to limited `positional extrapolation capacity` and `restricted context length`, which is an area for future work.

### 6.1.4. Performance of AudioVisual Video $\rightarrow$ Text
This section focuses on the model's ability to integrate and reason over both audio and visual streams from videos.

The following are the results from Table 11 of the original paper:

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

*Table 11: AudioVisual $\rightarrow$ Text performance of Qwen3-Omni-Instruct and other non-reasoning baselines. The highest scores are shown in bold.*

Table 11 shows that `Qwen3-Omni-Instruct` achieves `SOTA` performance on the `WorldSense benchmark` for `general understanding`, significantly surpassing `Gemini-2.5-Flash` and `Qwen2.5-Omni`. This validates its effectiveness in fundamental `multimodal integration`.

The following are the results from Table 12 of the original paper:

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

*Table 12: AudioVisual $\rightarrow$ Text performance of Qwen3-Omni-30B-A3B-Thinking and other reasoning baselines. The highest scores are shown in bold.*

Table 12 further illustrates `Qwen3-Omni-Thinking`'s enhanced performance on `complex reasoning tasks` requiring integration of `audio and visual information`, particularly on `DailyOmni` and `VideoHolmes`, where it sets new `SOTA` scores, surpassing `Gemini-2.5-Flash-Thinking` and previous open-source models. This demonstrates its potential for advanced `audiovisual perception` and `reasoning`.

### 6.1.5. Performance of X $\rightarrow$ Speech
This section evaluates `Qwen3-Omni`'s `speech generation capabilities`.

The following are the results from Table 13 of the original paper:

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
<td colspan="4">Content Consistency</td>
</tr>
<tr>
<td rowspan="10">SEED test-zh</td>
<td>Seed-TTSICL (Anastassiou et al., 2024)</td>
<td>1.11</td>
<td rowspan="10"></td>
</tr>
<tr>
<td>Seed-TTSRL (Anastassiou et al., 2024)</td>
<td>1.00</td>
</tr>
<tr>
<td>MaskGCT (Wang et al., 2024c)</td>
<td>2.27</td>
</tr>
<tr>
<td>E2 TTS (Eskimez et al., 2024)</td>
<td>1.97</td>
</tr>
<tr>
<td>F5-TTS (Chen et al., 2024c)</td>
<td>1.56</td>
</tr>
<tr>
<td>Spark TTS (Wang et al., 2025b)</td>
<td>1.20</td>
</tr>
<tr>
<td>CosyVoice 2 (Du et al., 2024)</td>
<td>1.45</td>
</tr>
<tr>
<td>CosyVoice 3 (Du et al., 2025)</td>
<td>0.71</td>
</tr>
<tr>
<td>Qwen2.5-Omni-7B (Xu et al., 2025)</td>
<td>1.42</td>
</tr>
<tr>
<td>Qwen3-Omni-30B-A3B</td>
<td>1.07</td>
</tr>
<tr>
<td rowspan="10">test-en</td>
<td>Seed-TTSICL (Anastassiou et al., 2024)</td>
<td>2.24</td>
<td rowspan="10"></td>
</tr>
<tr>
<td>Seed-TTSRL (Anastassiou et al., 2024)</td>
<td>1.94</td>
</tr>
<tr>
<td>MaskGCT (Wang et al., 2024c)</td>
<td>2.62</td>
</tr>
<tr>
<td>E2 TTS (Eskimez et al., 2024)</td>
<td>2.19</td>
</tr>
<tr>
<td>F5-TTS (Chen et al., 2024c)</td>
<td>1.83</td>
</tr>
<tr>
<td>Spark TTS (Wang et al., 2025b)</td>
<td>1.98</td>
</tr>
<tr>
<td>CosyVoice 2 (Du et al., 2024)</td>
<td>2.57</td>
</tr>
<tr>
<td>CosyVoice 3 (Du et al., 2025)</td>
<td>1.45</td>
</tr>
<tr>
<td>Qwen2.5-Omni-7B (Xu et al., 2025)</td>
<td>2.33</td>
</tr>
<tr>
<td>Qwen3-Omni-30B-A3B</td>
<td>1.39</td>
</tr>
</tbody>
</table>

*Table 13: Zero-Shot Speech Generation on Seed-TTS Test Set. The highest scores are shown in bold.*

Table 13, showing `zero-shot speech generation` performance (lower `WER` is better), highlights `Qwen3-Omni-30B-A3B` achieving competitive performance. For `test-zh`, `CosyVoice 3` leads, but `Qwen3-Omni` is competitive. For `test-en`, `Qwen3-Omni` achieves the best `WER` (1.39), indicating robust `speech understanding` and `generation`. The paper notes `RL optimization` yields significant improvements in stability.

The following are the results from Table 14 of the original paper:

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

*Table 14: Multilingual Speech Generation on MiniMax Multilingual Test Set. The highest scores are shown in bold.*

In `multilingual speech generation` (Table 14), `Qwen3-Omni` surpasses `MiniMax` and `ElevenLabs Multilingual v2` significantly for languages like Chinese, English, and French in terms of `Content Consistency` (lower is better, likely `WER` given previous context for `Content Consistency`), and delivers competitive or leading results in `Speaker Similarity` (higher is better). This indicates stable `cloned speech` generation with human-like voice characteristics across 10 supported languages.

The following are the results from Table 15 of the original paper:

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

*Table 15: Cross-Linguial Speech Generation on CosyVoice3 Cross-Linguial Test Set. The highest scores are shown in bold.*

For `cross-lingual speech generation` (Table 15), `Qwen3-Omni-30B-A3B` largely outperforms `CosyVoice3` in `any-to-en` (any language to English) and `any-to-ko` (any language to Korean) voice cloning, and achieves comparable performance in `any-to-ja` (any language to Japanese) even without text normalization. This demonstrates its superior adaptability across diverse linguistic contexts for voice cloning.

## 6.2. Evaluating Non-Degradation Across Modalities
A crucial claim of the paper is that `Qwen3-Omni` avoids performance degradation compared to unimodal models. To verify this, a controlled comparative study was conducted.

The following are the results from Table 16 of the original paper:

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
<td>General Tasks</td>
<td>MMLU</td>
<td>81.24</td>
<td>-</td>
<td>81.69</td>
</tr>
<tr>
<td></td>
<td>MMLU-Redux</td>
<td>80.17</td>
<td>-</td>
<td>80.60</td>
</tr>
<tr>
<td></td>
<td>MMLU-Pro</td>
<td>61.81</td>
<td>-</td>
<td>61.57</td>
</tr>
<tr>
<td></td>
<td>SuperGPQA</td>
<td>38.24</td>
<td>-</td>
<td>40.14</td>
</tr>
<tr>
<td></td>
<td>BBH</td>
<td>83.79</td>
<td>-</td>
<td>83.53</td>
</tr>
<tr>
<td>Math &amp; STEAM Tasks</td>
<td>GSM8K</td>
<td>90.83</td>
<td>-</td>
<td>91.36</td>
</tr>
<tr>
<td></td>
<td>MATH</td>
<td>60.84</td>
<td>-</td>
<td>60.42</td>
</tr>
<tr>
<td>Coding Tasks</td>
<td>EvalPlus</td>
<td>69.70</td>
<td>-</td>
<td>73.96</td>
</tr>
<tr>
<td></td>
<td>MultiPL-E</td>
<td>65.75</td>
<td>-</td>
<td>64.79</td>
</tr>
<tr>
<td></td>
<td>MBPP</td>
<td>72.60</td>
<td>-</td>
<td>72.60</td>
</tr>
<tr>
<td></td>
<td>CRUX-O</td>
<td>66.94</td>
<td>-</td>
<td>69.06</td>
</tr>
<tr>
<td>Multilingual Tasks</td>
<td>MGSM</td>
<td>78.75</td>
<td>-</td>
<td>79.93</td>
</tr>
<tr>
<td></td>
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
<td>General Visual Question Answering</td>
<td>MMStar</td>
<td>-</td>
<td>67.2</td>
<td>69.6</td>
</tr>
<tr>
<td></td>
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
<td></td>
<td>TextVQAval</td>
<td>-</td>
<td>81.67</td>
<td>81.65</td>
</tr>
<tr>
<td></td>
<td>DocVQAtest</td>
<td>-</td>
<td>95.19</td>
<td>95.27</td>
</tr>
<tr>
<td></td>
<td>InfoVQAtest</td>
<td>-</td>
<td>81.17</td>
<td>83.31</td>
</tr>
<tr>
<td></td>
<td>ChartQAtest Avg</td>
<td>-</td>
<td>87.12</td>
<td>87.52</td>
</tr>
<tr>
<td></td>
<td>OCRBench</td>
<td>-</td>
<td>85.8</td>
<td>86.0</td>
</tr>
<tr>
<td>Video Understanding Tasks</td>
<td>Video-MMEw/o sub</td>
<td>-</td>
<td>69.22</td>
<td>69.25</td>
</tr>
<tr>
<td></td>
<td>MVBench</td>
<td>-</td>
<td>71.87</td>
<td>69.50</td>
</tr>
<tr>
<td></td>
<td>LVBench</td>
<td>-</td>
<td>48.61</td>
<td>51.07</td>
</tr>
</tbody>
</table>

*Table 16: We compare the performance of 30A3 models that are contemporaneous and identical in size in Qwen series. To ensure experimental rigor, all models were trained under the same schedule, using identical datasets for their respective modalities and exactly matched training compute (FLOPs).*

Table 16 compares three 30B-A3B models from the Qwen series: `Qwen3-30B-A3B-Base` (text-only), `Qwen3-VL-30B-A3B-Base` (vision-only), and `Qwen3-Omni-30B-A3B-Base`. The results show that:
*   **No Degradation in Language Capability:** For `text modality` tasks (General, Math & STEAM, Coding, Multilingual), `Qwen3-Omni` generally matches or slightly surpasses the performance of the `text-only Qwen3-Base`. This is a critical finding, demonstrating that adding other modalities does not degrade the core `LLM`'s capabilities. For instance, on `MMLU`, `Qwen3-Omni` scores 81.69 vs `Qwen3-Base`'s 81.24. On `EvalPlus` for coding, `Qwen3-Omni` scores 73.96 vs `Qwen3-Base`'s 69.70.
*   **Mutual Enhancement for Other Modalities:** For `visual modality` tasks (College-level Problems, General Visual QA, OCR-related, Video Understanding), `Qwen3-Omni` consistently outperforms `Qwen3-VL-Base`. For example, on `MMMUval`, `Qwen3-Omni` scores 59.33 vs `Qwen3-VL`'s 57.22. On `InfoVQAtest`, `Qwen3-Omni` scores 83.31 vs `Qwen3-VL`'s 81.17. This suggests that the `multimodal training` process, particularly the inclusion of `text` and `audio` data, `mutually enhances` visual understanding.
*   **Audio's Impact on Vision:** The authors specifically note that adding `audio data` consistently improves `vision performance` on the `MMMU benchmark` and `OCR-related tasks`, highlighting an unexpected synergistic effect between audio and visual modalities.

    This controlled comparison rigorously supports the paper's central claim: `multimodal integration` can be achieved without `modality-induced degradation` and can even lead to performance gains in individual modalities due to cross-modal synergy.

## 6.3. Ablation Studies / Parameter Analysis
While the paper does not present explicit "ablation study" tables in the traditional sense where individual components are removed, the comparison in Section 6 (`Evaluating Non-Degradation Across Modalities`) serves a similar purpose by contrasting `unimodal` and `multimodal` base models trained under identical conditions. This effectively "ablates" the presence of other modalities to show the effect of `multimodal integration`.

The main findings from this `quasi-ablation study` (Table 16 and accompanying text) are:
1.  **Early Multimodal Integration:** The process of mixing `unimodal` and `cross-modal data` during the early stage of `text pretraining` is crucial. It allows `language models` to be `co-trained` with `vision` or `audio` without `degradation in language capability`. This implies that the method of introducing multimodal data is as important as the data itself.
2.  **Text Modality's Contribution:** The inclusion of the `text modality` substantially improves performance in the `vision and audio modalities`. This indicates that `text` acts as a powerful `semantic anchor` or `reasoning backbone` that benefits other modalities.
3.  **Visual/Audio Impact on Language (Limited):** Empirically, the authors `do not observe measurable gains in language ability from adding visual or audio signals`. This is an interesting asymmetry; while text benefits other modalities, the reverse is not strongly evident for language performance on `text-only benchmarks`.
4.  **Audio's Impact on Vision:** Adding `audio data` consistently improves `vision performance` on the `MMMU benchmark` and `OCR-related tasks`. This suggests a surprising `cross-modal synergy` where audio context can help disambiguate or enrich visual understanding.

    In terms of parameter analysis, Table 1 and Table 2 provide insights into the `MoE architecture` and its impact on performance:
*   The use of `MoE` for both `Thinker` and `Talker` (e.g., `Thinker` at `30B-A3B`, meaning 30 billion total parameters with likely 3 billion active per token) is a design choice to manage `computational cost` and `throughput`.
*   Table 2, showing `first-packet latency` under different `concurrency` levels, demonstrates that the `MoE architecture` ensures that `prefetch latency` and `Time-to-First-Token (TTPT)` remain `largely unaffected under high concurrency`. This highlights the architectural choice's effectiveness for practical deployment.
*   The `lightweight design` of the `MTP Module` (80M parameters) and `Codec Decoder` (`ConvNet` with 200M parameters) is explicitly chosen to minimize `computational overhead` and `inference latency`, reinforcing a clear parameter-based design decision for `real-time streaming`.

## 6.4. Appendix Results (Thinking Model on Perception Tasks)

The following are the results from Table 17 of the original paper:

<table>
<thead>
<tr>
<th></th>
<th>Seed -ASR</th>
<th>Voxtral -Mini</th>
<th>Voxtral -Small</th>
<th>GPT-4o -Transcribe</th>
<th>Gemini-2.5 -Pro</th>
<th>Qwen2.5 -Omni</th>
<th>Qwen3-Omni -30B-A3B-Thinking</th>
<th>Qwen3-Omni -Flash-Thinking</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">EN & ZH ASR (wer)</td>
</tr>
<tr>
<td>Wenetspeech net i meeting</td>
<td rowspan="2">4.66</td>
<td rowspan="2">5.69</td>
<td>24.30</td>
<td>31.53</td>
<td>20.33</td>
<td>26.08</td>
<td>15.30</td>
<td>14.43</td>
</tr>
<tr>
<td></td>
<td>13.47</td>
<td>5.91</td>
<td>7.65</td>
<td>4.69</td>
<td>5.89</td>
<td>6.16</td>
<td>6.85</td>
</tr>
<tr>
<td>Librispeech clean i other</td>
<td rowspan="2">1.58</td>
<td rowspan="2">2.84</td>
<td>1.88</td>
<td>4.12</td>
<td>1.56</td>
<td>3.30</td>
<td>1.39</td>
<td>1.27</td>
</tr>
<tr>
<td></td>
<td>3.75</td>
<td>2.89</td>
<td>3.56</td>
<td>1.74</td>
<td>3.45</td>
<td>2.22</td>
<td>1.82</td>
</tr>
<tr>
<td>CV15-en</td>
<td>-</td>
<td>9.47</td>
<td>7.79</td>
<td>10.01</td>
<td>9.89</td>
<td>7.61</td>
<td>10.44</td>
<td>10.52</td>
</tr>
<tr>
<td>CV15-zh</td>
<td>-</td>
<td>24.67</td>
<td>19.30</td>
<td>9.84</td>
<td>8.00</td>
<td>5.13</td>
<td>6.25</td>
<td>6.61</td>
</tr>
<tr>
<td>Fleurs-en</td>
<td>3.40</td>
<td>3.96</td>
<td>3.77</td>
<td>3.32</td>
<td>2.94</td>
<td>3.77</td>
<td>3.75</td>
<td>3.67</td>
</tr>
<tr>
<td>Fleurs-zh</td>
<td>2.69</td>
<td>12.22</td>
<td>7.98</td>
<td>2.44</td>
<td>2.71</td>
<td>2.54</td>
<td>2.73</td>
<td>2.57</td>
</tr>
<tr>
<td colspan="9">Multilingual ASR (wer)</td>
</tr>
<tr>
<td>Fleurs-avg (19 lang)<sup>a</sup></td>
<td>-</td>
<td>15.67</td>
<td>8.09</td>
<td>4.48</td>
<td>5.55</td>
<td>14.04</td>
<td>8.63</td>
<td>8.88</td>
</tr>
<tr>
<td colspan="9">Lyric ASR (wer)</td>
</tr>
<tr>
<td>MIR-1K (vocal-only)<sup>b</sup></td>
<td>6.45</td>
<td>23.33</td>
<td>18.73</td>
<td>11.87</td>
<td>9.85</td>
<td>8.15</td>
<td>11.15</td>
<td>10.47</td>
</tr>
<tr>
<td>Opencpop-test</td>
<td>2.98</td>
<td>31.01</td>
<td>16.06</td>
<td>7.93</td>
<td>6.49</td>
<td>2.84</td>
<td>6.11</td>
<td>4.52</td>
</tr>
<tr>
<td colspan="9">S2TT (BLEU)</td>
</tr>
<tr>
<td>Fleurs-en2xx<sup>c</sup></td>
<td>-</td>
<td>30.35</td>
<td>37.85</td>
<td>-</td>
<td>39.25</td>
<td>29.22</td>
<td>36.24</td>
<td>36.04</td>
</tr>
<tr>
<td>Fleurs-xx2en</td>
<td>-</td>
<td>27.54</td>
<td>32.81</td>
<td>-</td>
<td>35.41</td>
<td>28.61</td>
<td>30.50</td>
<td>30.22</td>
</tr>
<tr>
<td>Fleurs-zh2xx</td>
<td>-</td>
<td>17.03</td>
<td>22.05</td>
<td>-</td>
<td>26.63</td>
<td>17.97</td>
<td>23.74</td>
<td>23.77</td>
</tr>
<tr>
<td>Fleurs-xx2zh</td>
<td>-</td>
<td>28.75</td>
<td>34.82</td>
<td>-</td>
<td>37.50</td>
<td>27.68</td>
<td>34.51</td>
<td>34.49</td>
</tr>
</tbody>
</table>

*Table 17: Transcription performance for Audio→Text tasks (ASR & S2TT), comparing Qwen3-Omni-Thinking with the baselines. The highest scores are shown in bold.*

The following are the results from Table 18 of the original paper:

<table>
<thead>
<tr>
<th></th>
<th>Best Specialist Models</th>
<th>GPT-4o -Audio</th>
<th>Gemini-2.5 -Pro</th>
<th>Qwen2.5 -Omni</th>
<th>Qwen3-Omni -30B-A3B-Thinking</th>
<th>Qwen3-Omni -Flash-Thinking</th>
</tr>
</thead>
<tbody>
<tr>
<td>RUL-MuchoMusic</td>
<td>47.6 (Audio Flamingo 3) (Goel et al., 2025)</td>
<td>36.1</td>
<td>49.4</td>
<td>47.3</td>
<td>48.3</td>
<td>48.4</td>
</tr>
<tr>
<td>GTZAN</td>
<td>87.9 (CLaMP 3)</td>
<td>76.5</td>
<td>81.0</td>
<td>81.7</td>
<td>89.0</td>
<td>89.0</td>
</tr>
<tr>
<td>MTG Genre</td>
<td>35.8 (MuQ-MuLan)</td>
<td>25.3</td>
<td>32.6</td>
<td>32.5</td>
<td>32.5</td>
<td>33.0</td>
</tr>
<tr>
<td>MTG Mood/Theme</td>
<td>10.9 (MuQ-MuLan)</td>
<td>11.3</td>
<td>14.1</td>
<td>8.9</td>
<td>14.3</td>
<td>15.4</td>
</tr>
<tr>
<td>MTG Instrument</td>
<td>39.8 (MuQ-MuLan)</td>
<td>34.2</td>
<td>33.0</td>
<td>22.6</td>
<td>36.4</td>
<td>36.4</td>
</tr>
<tr>
<td>MTG Top50</td>
<td>33.2 (MuQ-MuLan)</td>
<td>25.0</td>
<td>26.1</td>
<td>21.6</td>
<td>29.1</td>
<td>29.3</td>
</tr>
<tr>
<td>MagnaTagATune</td>
<td>41.6 (MuQ)</td>
<td>29.2</td>
<td>28.1</td>
<td>30.1</td>
<td>32.2</td>
<td>32.6</td>
</tr>
</tbody>
</table>

*Table 18: Music understanding performance for Audio→Text tasks, comparing Qwen3-Omni-Thinking with baselines. The highest scores are shown in bold.*

Tables 17 and 18, presented in the Appendix, show that for primarily `perception-based tasks` like $ASR/S2TT$ and `Music understanding`, the `Qwen3-Omni-Thinking` model is generally outperformed by its `Instruct` counterpart. This suggests that engaging `sophisticated reasoning processes` is not beneficial for these tasks and may even introduce a `higher propensity for hallucinations`, highlighting the importance of specialized instruction-following for direct perceptual tasks.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The Qwen3-Omni technical report introduces a new generation of multimodal AI models (`Qwen3-Omni-30B-A3B`, `Qwen3-Omni-30B-A3B-Thinking`, `Qwen3-Omni-Flash-Instruct`, and `Qwen3-Omni-Flash-Thinking`). The core achievement is demonstrating that `end-to-end multimodal training` can maintain or even surpass `state-of-the-art performance` across text, image, audio, and video modalities without the typical performance degradation observed in prior integrated models. The model exhibits exceptional performance in audio processing and dialogue, achieving `open-source SOTA` on 32 benchmarks and overall `SOTA` on 22, often outperforming strong closed-source competitors.

Key innovations include the `Thinker-Talker MoE architecture`, the custom `AuT audio encoder`, a `multi-codebook streaming speech synthesis` pipeline with a lightweight `causal ConvNet` for `Code2Wav`, leading to an impressive `234 ms first-packet latency`. The model supports a broad range of languages (119 text, 19 speech understanding, 10 speech generation) and extended audio contexts (up to 40 minutes). The dedicated `Thinking variant` further enhances complex `multimodal reasoning`. Additionally, a specialized `Captioner` model addresses the need for general-purpose `audio captioning`. The public release of these models under Apache 2.0 license underscores the team's commitment to open research. The paper concludes that `Qwen3-Omni` sets a milestone by providing the first evidence of `fully integrated, end-to-end multimodal training` without compromising core language capabilities or other modalities.

## 7.2. Limitations & Future Work
The authors explicitly mention a limitation of the current model in `long video benchmarks`, attributing it to two architectural constraints:
1.  A `limited capacity for positional extrapolation`.
2.  A `restricted context length`.
    Addressing these constraints is identified as a key objective for future work.

Beyond the stated limitations, the authors outline several directions for future research and development:
*   **Multi-speaker ASR:** Improving `Automatic Speech Recognition` capabilities for scenarios involving multiple speakers.
*   **Video OCR:** Enhancing `Optical Character Recognition` within video content.
*   **Audiovisual Proactive Learning:** Developing capabilities for the model to learn proactively from `audiovisual inputs`.
*   **Enhanced Agent-based Workflows and Function Calling:** Further integrating the model into `agent-based systems` and improving its ability to execute `function calls` based on multimodal understanding.

## 7.3. Personal Insights & Critique
Qwen3-Omni represents a significant step forward in `multimodal AI`. The most compelling insight is the demonstration that `multimodal integration` doesn't necessarily entail performance trade-offs. The controlled comparison in Table 16, showing `Qwen3-Omni` matching or exceeding its unimodal counterparts (text-only and vision-only) while consuming comparable compute, is a strong empirical validation of this claim. This suggests that the model benefits from `cross-modal synergy`, where knowledge from one modality (e.g., text's reasoning abilities) can enhance the performance in another (e.g., vision or audio). The observation that text generally benefits other modalities, while audio/vision do not show measurable gains on *text-only* benchmarks, is an interesting asymmetry that warrants further investigation into how `multimodal information` is fused and prioritized internally.

The emphasis on `ultra-low latency` for real-time interaction is highly practical and addresses a critical bottleneck for deploying truly interactive AI systems. The combination of `MoE`, `chunked prefilling`, `multi-codebook generation`, and a `lightweight causal ConvNet` is an elegant engineering solution to this problem.

One potential area for deeper exploration or critique is the interpretation of "without degradation relative to single-modal counterparts." While the presented benchmarks are extensive, the definition of "same-sized" is based on parameter count. Future work could investigate if the `multimodal model` truly matches the efficiency of highly specialized, potentially smaller, but exquisitely optimized unimodal models for niche tasks, or if the benefit lies primarily in its generality and integrated reasoning. The qualitative results for `audio captioning` are promising, filling a gap in the literature and showcasing the model's descriptive capabilities. It would be beneficial to see quantitative benchmarks for `audio captioning` in future work to fully assess this new capability.

The model's ability to support an extensive list of languages for both `speech understanding` and `generation` (especially `cross-lingual voice cloning`) has vast implications for global accessibility and communication, enabling more natural human-computer interaction across diverse linguistic backgrounds.

This paper provides a strong foundation for future research in several directions. The concept of a `Thinking model` explicitly reasoning over multimodal inputs is particularly intriguing and could lead to more robust and explainable `multimodal decision-making`. The publicly available models will enable the research community to build upon these advancements, fostering further innovation in truly intelligent, perceptive, and interactive AI systems.