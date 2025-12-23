# 1. Bibliographic Information

## 1.1. Title

**ModRWKV: Transformer Multimodality in Linear Time**

This paper focuses on a novel approach to multimodal large language models (MLLMs) that leverage recurrent neural network (RNN) architectures—specifically, a model named ModRWKV built upon the RWKV7 architecture. It addresses multimodality (combining modalities such as text, vision, and audio) while achieving linear computational complexity analogous to RNNs rather than the quadratic complexity typically seen in Transformer models.

## 1.2. Authors

The authors of this work are:

- Jiale Kang*
- Ziyin Yue*
- Qingyu Yin*
- Jiang Rui
- Weile Li
- Zening Lu
- Zhouran Ji

  They are affiliated with:

1. RWKvOS
2. Zhejiang University
3. The Hong Kong University of Science and Technology

   (*These three authors contributed equally to this work.)

Their domain expertise lies primarily in large language models, RNN architectures, multimodal modeling, and efficient neural network design. The RWKvOS affiliation suggests involvement with the RWKV architecture, a recent linear RNN model, while the university affiliations bring academic rigor and research diversity.

## 1.3. Journal/Conference

This paper is currently published as a **preprint on arXiv** and does not yet appear to be associated with a formal conference or journal publication.

The URL: https://arxiv.org/abs/2505.14505

Its preprint status indicates the work is publicly accessible for early dissemination but has not yet undergone peer review. arXiv is a well-known platform for disseminating research in computer science and machine learning, making the work quickly available to the community.

## 1.4. Publication Year

The paper is dated **May 2025** (specifically, 2025-05-20).

## 1.5. Abstract Summary

The paper tackles the limitation that current multimodal large language models (MLLMs) typically rely on Transformers with quadratic time complexity, leading to expensive inference, especially with longer sequences or multiple modalities. Linear-time models like RNNs, although efficient, have traditionally been applied only to single modality (text).

The authors propose *ModRWKV*, an RNN-based multimodal framework leveraging RWKV7 as the backbone. This model achieves multimodal fusion through adaptable, lightweight modality-specific encoders that dynamically integrate heterogeneous sources. Initializing ModRWKV from pretrained RWKV7 weights accelerates training and improves performance on multimodal understanding tasks.

Experimental results on various benchmarks indicate that modern RNN architectures offer a **viable and more efficient alternative** to Transformers for multimodal learning. The authors also identify the optimal architectural configurations through systematic exploration.

## 1.6. Original Source Link

- Preprint link: https://arxiv.org/abs/2505.14505
- PDF link: https://arxiv.org/pdf/2505.14505v1.pdf
- Status: **Preprint (not peer-reviewed)** as of the latest update.

# 2. Executive Summary

## 2.1. Background & Motivation

- **Core Problem**: Most state-of-the-art multimodal large language models rely on Transformers, which have **quadratic computational complexity** during inference (due to the self-attention mechanism scaling as sequence length squared), resulting in high resource consumption when processing multimodal inputs like images, audio, and text together.
  
- While recurrent neural networks (RNNs) offer **linear time complexity** with constant memory usage and lower inference costs, their application has been largely confined to natural language/text-only models. Extending RNN-based models to handle multimodal inputs effectively remains an open challenge.

- Multimodal fusion using Transformers relies heavily on cross-modal representations and attention mechanisms, but their quadratic complexity limits scalability, especially for large-scale or real-time tasks.

- Prior research on linear models (e.g., RNNs) has not adequately addressed multimodal fusion across heterogeneous modalities, creating a **gap** in the field for efficient, scalable multimodal modeling.

- The **innovative entry point** of this paper is to leverage a modern linear-time RNN architecture (RWKV7), combined with dynamically adaptable encoders for each modality, to create a unified multimodal model that can process vision, audio, time series, and language data efficiently.

## 2.2. Main Contributions / Findings

1. **Proposal of ModRWKV framework**: It pioneers a unified RNN-based multimodal large language model paradigm by integrating modality-specific encoders with a shared RWKV7 backbone. The design is plug-and-play, allowing quick modality switching and scalable integration.

2. **Comprehensive benchmarking**: The authors conduct extensive evaluations on multiple well-known multimodal benchmarks covering vision, audio, and time series data. This establishes a new **benchmark paradigm** to assess RNN architectures' performance in multimodal understanding, contrasting with Transformer models.

3. **Systematic ablation and efficiency studies**: The model design is extremely lightweight, balancing computational efficiency and performance. The paper systematically explores convolutional sequence compression, adapter size scaling, and initialization of pretrained weights, determining configurations that optimize performance under resource constraints.

4. **Key finding**: Modern RNN models such as ModRWKV can provide **competitive or superior accuracy in multimodal tasks** relative to larger Transformer-based baselines, while maintaining much lower inference costs due to linear computational complexity.

   Overall, this paper challenges the Transformer monopoly in MLLMs by demonstrating that linear-time RNNs can serve as an efficient, scalable alternative for multimodal large language modeling.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Transformers

- Transformers are a neural network architecture introduced by Vaswani et al. (2017) known for the `self-attention` mechanism. Unlike RNNs that process sequential data stepwise, Transformers operate on entire input sequences simultaneously using attention, capturing dependencies regardless of positions.

- The **quadratic complexity** (time and memory) comes from calculating the attention scores for each token against every other token in the sequence: for sequence length $n$, $O(n^2)$.

- Example attention formula:

  $$
  \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
  $$

  where $Q$, $K$, and $V$ are query, key, and value matrices derived from input embeddings and $d_k$ is the dimensionality for scaling.

### 3.1.2. Recurrent Neural Networks (RNNs)

- RNNs process sequences one step at a time, maintaining a hidden state $h_t$ that captures information from past tokens.

- Basic linear RNN update:

  $h_t = W h_{t-1} + U x_t$

  where $x_t$ is the input at time $t$, $W$ and $U$ are weight matrices.

- RNNs have **linear computation complexity**—processing sequence length $l$ scales as $O(l)$.

- Traditionally, vanilla RNNs struggle with long-term dependencies, but newer variants and enhancements address these issues.

### 3.1.3. RWKV Model

- RWKV is a modern RNN model that combines RNN efficiency with Transformer-like language modeling power by carefully designing the state update mechanism.

- Core state update equation:

  $$
  s_t = G_t s_{t-1} + a_t k_t v_t^T
  $$

  where:

  - $s_t$: state at time $t$

  - $G_t$: input-dependent generalized transition matrix incorporating decay and gating

  - $a_t$: in-context learning rate vector, projected from input $x_t$

  - $k_t$, $v_t$: key and value vectors derived from input deltas $\Delta x_t$

- This formulation allows parallelizable training and dynamic adaptation, preserving expressiveness and effectively modeling long-range dependencies.

### 3.1.4. Multimodal Large Language Models (MLLMs)

- MLLMs integrate multiple data modalities—commonly text, vision, audio—into a single model capable of joint understanding and generation.

- Multimodal architectures typically incorporate modality-specific encoders (e.g., visual transformers, audio transformers) to convert raw modality data into feature embeddings.

- Fusion approaches include unified tokenization (converting all modalities into token sequences) or cross-modal attention mechanisms enabling modalities to attend to each other.

- Examples include LLaVA, Flamingo, and other Transformer-based MLLMs.

## 3.2. Previous Works

### 3.2.1. Transformer-Based MLLMs

- Many recent works on multimodal large language models rely on the Transformer backbone:

  - LLaVA (Liu et al., 2023): Vision-language instruction tuning adapted on Vicuna Transformers.

  - LLama-Omni (Fang et al., 2025): Extends large language models to speech.

  - VL-Mamba (Qiao et al., 2024): Explores multimodal learning with State Space Models, another linear complexity approach but still Transformer-based.

- These models generally depend on cross-attention or concatenated tokens for multimodal fusion, incurring quadratic complexity.

### 3.2.2. Linear Models and RNN-Based Approaches

- Recent research proposes linear complexity models as alternatives to Transformers, including techniques leveraging the delta rule or modified RNNs:

  - Peng et al. (2025): Developed RWKV7 with expressive dynamic state evolution.

  - Gu and Dao (2024): Mamba, linear-time sequence modeling with selective state spaces.

  - Yang et al. (2024a, 2025): Parallelizing linear transformers with hardware-friendly designs for GPUs.

- These models have shown promise in text-only LLM tasks but have not been robustly applied to multimodal modeling prior to this work.

### 3.2.3. Pretrained Modality Encoders

- Vision: CLIP (Radford et al., 2021) and SigLIP2 (Tschannen et al., 2025) serve as pretrained visual encoders that extract rich image features.

- Audio: WavLM (Chen et al., 2022) and Whisper (Radford et al., 2022) are pretrained models for speech/audio feature extraction.

- Time Series: WaveNet (Van Den Oord et al., 2016) and Timer (Liu et al., 2024b) are encoders designed for temporal sequence processing.

## 3.3. Technological Evolution

- The landscape evolved from text-only Transformers to multimodal Transformers, integrating vision and speech via heavy cross-modal attention at quadratic cost.

- Parallelly, linear-time models such as RNNs and State Space Models emerged to reduce computational overhead.

- However, multimodal fusion with linear models remained unexplored, creating a gap.

- This paper positions ModRWKV as a breakthrough, bringing linear RNN efficiency to the multimodal domain, shifting focus away from Transformers.

## 3.4. Differentiation Analysis

- Unlike Transformer-based MLLMs that rely on quadratic-cost self-attention for modality fusion, ModRWKV uses an **RNN backbone with linear state updates** for efficient inference.

- The fusion is performed by modality-specific adapters and lightweight encoders that project heterogeneous inputs into a shared embedding space compatible with the RWKV7 backbone.

- The modular design enables easy modality switching and adaptability without redesigning the core architecture.

- The model benefits from pretrained RWKV7 weights that speed up training and improve multimodal reasoning—an approach less examined in prior linear models.

- The paper also explores **sequence compression methods** via 1D convolution, balancing token length and efficiency for multimodal inputs.

- Overall, it pioneers RNN use in large-scale multimodal LLMs, demonstrating competitive performance despite smaller model sizes.

# 4. Methodology

## 4.1. Principles

The core principle of ModRWKV lies in combining:

- The **linear time and constant-memory behavior of RNN architectures** (specifically RWKV7), which avoid the quadratic costs of Transformer attention,

- With **multimodal fusion** achieved via **modality-specific encoders and lightweight adapters** that map heterogeneous inputs into a shared space the RWKV7 backbone can process.

  This modular, plug-and-play design:

- Decouples modality encoders from the backbone, allowing flexible addition or replacement of modalities.

- Leverages pretrained RWKV7 weights, capitalizing on strong language modeling capabilities developed in text-only contexts, accelerating multimodal learning.

- Introduces sequence compression with 1D convolution to handle long sequences efficiently.

  Overall, ModRWKV approximates the expressive capacity of Transformers in fusion tasks but maintains linear computational efficiency in inference.

## 4.2. Core Methodology In-depth (Layer by Layer)

The methodology is presented in three main parts: multimodal encoders, adapter design, and sequence compression.

### 4.2.1. Multimodal Encoder Design (Section 3.1)

ModRWKV employs specialized modality encoders that process raw data into sequential embeddings compatible with the RWKV7 backbone.

- **Vision Encoder**

  - Two visual encoders evaluated: `CLIP` (Radford et al., 2021) and `SigLIP2` (Tschannen et al., 2025).

  - Both process raw images and produce sequential feature embeddings of length 577 tokens (visual tokens).

  - These embeddings are further transformed through lightweight adapter layers to align with RWKV input dimensions.

- **Audio Encoder**

  - Uses pretrained models: `WavLM` (Chen et al., 2022) and `Whisper` (Radford et al., 2022).

  - These are chosen in different parameter scales (from ~100M to 400M parameters).

  - Audio is sampled at 16,000 Hz and features are extracted at 50 Hz frequency.

  - For Whisper, fixed 30-second audio segments are padded as input.

- **Time Series Encoder**

  - Two temporal encoders: `WaveNet` (Van Den Oord et al., 2016) and `Timer` (Liu et al., 2024b).

  - WaveNet trained from scratch; Timer initialized with frozen pretrained weights.

  - These transform raw time series into feature embeddings compatible with RWKV7 backbone.

    All encoders transform diverse raw modalities into **sequential embedding vectors** that can be fused within ModRWKV.

### 4.2.2. Adapter Design (Section 3.2)

Adapters are crucial for bridging modality-specific embeddings with the RWKV backbone embedding space, ensuring the embeddings align in dimension and semantics for fusion.

- The adapter is a **single MLP (multi-layer perceptron) module** designed for dimensionality alignment, employing fewer parameters to emphasize RWKV7's role in cross-modal reasoning.

- Adapter architecture:

  $$
  \mathbf{h} = \mathrm{Linear}_2(\mathrm{ReLU}(\mathrm{Linear}_1(\mathbf{x})))
  $$

Where:

- $\mathbf{x}$: input feature vector from modality encoder.

- $\mathrm{Linear}_1$ and $\mathrm{Linear}_2$: fully connected layers.

- $\mathrm{ReLU}$: Rectified Linear Unit activation function introducing non-linearity.

  This minimalistic adapter forces the RWKV7 core to perform most multimodal integration and reasoning, serving as a rigorous testing ground for the RNN-based architecture.

### 4.2.3. Sequence Compression via 1D Convolution (Section 3.3)

Modality encoders often produce long token sequences (e.g., 577 tokens for a single image). To improve efficiency, ModRWKV compresses these sequences using 1D convolution (`Conv1D`).

- Convolutional compression reduces the sequence length while preserving key information.

- Formal computation for a 1D convolution:

  $$
  \mathbf{y}_c = \sum_{i=1}^{C_{in}} \left( \sum_{j=0}^{k-1} W_{c,i,j} \cdot x_{i, s \cdot t + j} \right) + b_c
  $$

Where:

- $\mathbf{y}_c$: output vector channel $c$.

- $C_{in}$: number of input channels.

- $k$: kernel size of convolution.

- $s$: stride length.

- $t = 0, \dots, L' - 1$: output sequence index.

- $L' = \left\lfloor \frac{L + 2p - k}{s} \right\rfloor + 1$: output sequence length, with padding $p$.

- $W_{c,i,j}$: convolution kernel weights.

- $x_{i, \cdot}$: input sequence feature channel.

  This convolutional module compresses embeddings from length $L$ to $L' < L$, reducing computation in the subsequent RWKV layers.

### 4.2.4. RWKV7 Backbone Recap

The RWKV7 model core performs the sequential fusion and reasoning on multimodal embeddings.

The state update at step $t$ is:

$$
s_t = G_t s_{t-1} + a_t k_t v_t^T
$$

Where:

- $s_t$: state matrix at current step.

- $G_t$: input-dependent transition matrix.

- $a_t$: vector-valued learning rate controlling new information contribution, projected from input $x_t$ as $a_t = W_a x_t$.

- $k_t$, $v_t$: key and value vectors generated from input delta $\Delta x_t$.

  Specifically, $G_t$ is defined as:

$$
G_t = (I - a_t k_t k_t^T) \mathrm{diag}(e^{-e^{w_t}})
$$

where $w_t = W_w x_t$ is a vector-valued gating parameter.

These dynamic, context-dependent matrices enable RWKV7 to capture complex dependencies efficiently with linear complexity.

### 4.2.5. Training Procedure

- Two-phase training paradigm:

  1. **Phase I:** Freeze encoders and RWKV backbone. Train only multimodal adapters to map modality embeddings to RWKV's space.

  2. **Phase II:** Unfreeze adapters and RWKV backbone; encoders remain frozen to retain pretrained modal representations. Fine-tune jointly for multimodal understanding.

     This progressive training allows the model to adapt the fusion layers without disrupting pretrained modality knowledge.

# 5. Experimental Setup

## 5.1. Datasets

The authors conduct experiments across multiple modalities using well-established datasets:

- **Vision**

  - *Training*: LLaVA-595K and LLaVA-665K multimodal instruction datasets, containing text-image pairs for instruction tuning.

  - *Benchmarks* (evaluation):

    - VQA-v2 (Goyal et al., 2017): Visual Question Answering dataset.

    - TextVQA (Singh et al., 2019): Emphasizes understanding text within images.

    - GQA (Hudson and Manning, 2019): Compositional visual reasoning dataset.

    - ScienceQA (Lu et al., 2022): Scientific reasoning with multimodal questions.

    - POPE (Li et al., 2023): Dataset targeting object hallucination evaluation.

    - MMMU (Yue et al., 2024): A challenging multi-discipline reasoning benchmark.

    - MMBench (Liu et al., 2024c): Comprehensive multimodal evaluation suite.

- **Audio**

  - *Training*: LibriSpeech (Panayotov et al., 2015), 960 hours of English speech.

  - Aishell-1 (Bu et al., 2017), 170 hours of Mandarin Chinese speech.

- **Time Series**

  - GIFT-Eval (Aksu et al., 2024): A general benchmark for time series forecasting.

  - UTSD (Liu et al., 2024b): Public time series datasets with univariate signals.

    *Example of a data sample:*

- An image from VQA datasets may contain a photo with a question like "How many cars are in the picture?" with an expected natural language answer.

- Audio samples from LibriSpeech consist of spoken English sentences used for automatic speech recognition.

- Time series samples include financial or environmental sequential data over time intervals.

  These datasets cover a wide spectrum of practical multimodal scenarios, enabling thorough validation of the model’s ability to handle cross-modal understanding tasks.

## 5.2. Evaluation Metrics

The paper employs multiple domain-specific metrics depending on modality and task.

### 5.2.1. Accuracy-Based Metrics (Vision and Multimodal QA)

- **Accuracy**

  - *Conceptual Definition*: Measures the proportion of correctly predicted answers or labels among all test samples.

  - *Formula*:

    $$
    \mathrm{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of samples}} \times 100\%
    $$

### 5.2.2. Speech Recognition Metrics

- **Word Error Rate (WER)**

  - *Conceptual Definition*: Quantifies the difference between transcribed text and ground truth by counting the minimum number of word insertions, deletions, and substitutions required to convert the hypothesis into reference.

  - *Formula*:

    $$
    \mathrm{WER} = \frac{S + D + I}{N} \times 100\%
    $$

    where:

    - $S$: number of substitutions

    - $D$: number of deletions

    - $I$: number of insertions

    - $N$: total number of words in the reference transcript

- **Character Error Rate (CER)**

  - Similar to WER but operates at the character level, useful for languages without clear word delimiters (e.g., Chinese).

  - *Formula*:

    $$
    \mathrm{CER} = \frac{S_c + D_c + I_c}{N_c} \times 100\%
    $$

    where the variables correspond to substitutions, deletions, insertions, and reference characters.

### 5.2.3. Time Series Metrics

- **Mean Squared Error (MSE)**

  - *Conceptual Definition*: Measures the average squared difference between predicted and actual numerical values in forecasting tasks.

  - *Formula*:

    $$
    \mathrm{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
    $$

    where:

    - $y_i$: true value at time $i$

    - $\hat{y}_i$: predicted value at time $i$

    - $n$: number of prediction points

- Lower MSE indicates better forecasting accuracy.

## 5.3. Baselines

- **LLM Backbones**: Vicuna (7B, 13B), Phi-2-2.7B, MobileLLaMA-2.7B, Mamba LLM-2.8B

- **Multimodal Models**: LLaVA (1.5 & 1.6 versions), LLaVA-Phi, MobileVLM, VL-Mamba

- **Time Series Models**: TimeFM, Timer, UniTS, TTM, MOIRAI, ROSE

  These baselines represent the current state-of-the-art Transformer-based MLLMs and specialized models in multimodal and time-series forecasting. Comparing to them establishes the competitiveness of ModRWKV in different parameter size regimes and tasks.

# 6. Results & Analysis

## 6.1. Core Results Analysis

### 6.1.1. Vision Multimodal Understanding

The authors evaluated ModRWKV against transformer-based state-of-the-art models on multiple vision-language benchmarks.

The following are the results from **Table 2** of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>LLM</th>
<th>PT (k)</th>
<th>IT (k)</th>
<th>VQA-v2</th>
<th>GQA</th>
<th>SQAI</th>
<th>VQAT</th>
<th>POPE</th>
<th>MMB</th>
<th>MMMU</th>
</tr>
</thead>
<tbody>
<tr>
<td>LLaVA-1.5</td>
<td>Vicuna-7B</td>
<td>558</td>
<td>665</td>
<td>78.5</td>
<td>62.0</td>
<td>66.8</td>
<td>58.2</td>
<td>86.5</td>
<td>64.3</td>
<td>-</td>
</tr>
<tr>
<td>LLaVA-1.5</td>
<td>Vicuna-13B</td>
<td>558</td>
<td>665</td>
<td>80.0</td>
<td>63.3</td>
<td>71.6</td>
<td>61.3</td>
<td>86.2</td>
<td>67.7</td>
<td>-</td>
</tr>
<tr>
<td>LLaVA-1.6</td>
<td>Vicuna-7B</td>
<td>558</td>
<td>665</td>
<td>81.8</td>
<td>64.2</td>
<td>72.8</td>
<td>65.7</td>
<td>86.7</td>
<td>67.7</td>
<td>35.8</td>
</tr>
<tr>
<td>LLaVA-Phi</td>
<td>Phi-2-2.7B</td>
<td>558</td>
<td>665</td>
<td>71.4</td>
<td>-</td>
<td>68.4</td>
<td>48.6</td>
<td>85.0</td>
<td>59.8</td>
<td>-</td>
</tr>
<tr>
<td>MobileVLM-3B</td>
<td>MobileLLaMA-2.7B</td>
<td>558</td>
<td>665</td>
<td>-</td>
<td>59.0</td>
<td>61.2</td>
<td>47.5</td>
<td>84.9</td>
<td>59.6</td>
<td>-</td>
</tr>
<tr>
<td>VL-Mamba</td>
<td>Mamba LLM-2.8B</td>
<td>558</td>
<td>665</td>
<td>76.6</td>
<td>56.2</td>
<td>65.4</td>
<td>48.9</td>
<td>84.4</td>
<td>57.0</td>
<td></td>
</tr>
<tr>
<td><b>MoDRWKV</b></td>
<td><b>RWKV7 LLM-3B</b></td>
<td><b>558</b></td>
<td><b>665</b></td>
<td><b>78.3</b></td>
<td><b>60.8</b></td>
<td><b>70.9</b></td>
<td><b>51.1</b></td>
<td><b>87.1</b></td>
<td><b>66.6</b></td>
<td><b>38.7</b></td>
</tr>
</tbody>
</table>

**Analysis:**

- ModRWKV with a 3B parameter RWKV7 backbone performs competitively, often outperforming similarly or larger sized Transformer models, particularly in POPE (object hallucination reduction) and MMMU (multi-discipline reasoning), indicating good cross-modal reasoning.

- It outperforms VL-Mamba (2.8B) consistently and achieves comparable results to LLaVA-1.5 (7B), indicating better parameter efficiency.

- The results support the claim that an RNN-based linear architecture can match or surpass Transformer models with fewer parameters and lower complexity.

### 6.1.2. Audio Recognition Performance

The following are the results from **Table 3**:

<table>
<thead>
<tr>
<th>Dataset</th>
<th>Data (h)</th>
<th>Encoder</th>
<th>Clean WER (%)</th>
<th>Other WER (%)</th>
<th>Dev CER (%)</th>
<th>Test CER (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">LibriSpeech</td>
<td rowspan="4">960</td>
<td>wavlm large</td>
<td>2.43</td>
<td>6.51</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>wavlm base+</td>
<td>3.08</td>
<td>10.38</td>
<td></td>
<td></td>
</tr>
<tr>
<td>whisper medium</td>
<td>5.33</td>
<td>12.28</td>
<td></td>
<td></td>
</tr>
<tr>
<td>whisper small</td>
<td>6.24</td>
<td>16.92</td>
<td></td>
<td>-</td>
</tr>
<tr>
<td rowspan="4">Aishell-1</td>
<td rowspan="4">178</td>
<td>wavlm large</td>
<td>-</td>
<td></td>
<td>9.68</td>
<td>10.33</td>
</tr>
<tr>
<td>wavlm base+</td>
<td></td>
<td></td>
<td>12.40</td>
<td>13.46</td>
</tr>
<tr>
<td>whisper medium</td>
<td></td>
<td></td>
<td>5.08</td>
<td>5.83</td>
</tr>
<tr>
<td>whisper small</td>
<td></td>
<td></td>
<td>6.29</td>
<td>6.95</td>
</tr>
</tbody>
</table>

**Analysis:**

- Using the Whisper medium encoder, MoDRWKV achieves the best reported character error rates (CER) on Aishell-1 (5.08% dev, 5.83% test), showing strong recognition in Chinese speech.

- On LibriSpeech, WER of 2.43% on clean test set with WavLM large encoder is competitive, signaling precise English speech recognition despite no additional data augmentation.

- Shows effectiveness in handling various languages and audio conditions.

### 6.1.3. Time Series Forecasting

From **Table 4**, zero-shot mean squared error (MSE) results on several datasets using WaveNet encoder with adapter scaling $4 \times$:

<table>
<thead>
<tr>
<th>Model</th>
<th>LB-FL</th>
<th>ECL</th>
<th>ECL</th>
<th>ETTh1</th>
<th>ETTh2</th>
<th>ETTm1</th>
<th>ETTm2</th>
<th>WTH</th>
<th>Traffic</th>
</tr>
</thead>
<tbody>
<tr>
<td>TimeFM</td>
<td>720-96</td>
<td>0.119</td>
<td>0.421</td>
<td>0.326</td>
<td>0.363</td>
<td>0.206</td>
<td>0.123</td>
<td>0.327</td>
</tr>
<tr>
<td>Timer</td>
<td>720-96</td>
<td>0.221</td>
<td>0.414</td>
<td>0.305</td>
<td>0.440</td>
<td>0.203</td>
<td>0.178</td>
<td>0.526</td>
</tr>
<tr>
<td>UniTS</td>
<td>720-96</td>
<td>0.175</td>
<td>0.377</td>
<td>0.323</td>
<td>0.761</td>
<td>0.249</td>
<td>0.194</td>
<td>0.481</td>
</tr>
<tr>
<td>TTM</td>
<td>720-96</td>
<td>0.170</td>
<td>0.368</td>
<td>0.286</td>
<td>0.415</td>
<td>0.186</td>
<td>0.152</td>
<td>0.509</td>
</tr>
<tr>
<td>MOIRAI</td>
<td>720-96</td>
<td>0.212</td>
<td>0.394</td>
<td>0.285</td>
<td>0.516</td>
<td>0.222</td>
<td>0.208</td>
<td>1.359</td>
</tr>
<tr>
<td>ROSE</td>
<td>720-96</td>
<td>0.209</td>
<td>0.382</td>
<td>0.298</td>
<td>0.512</td>
<td>0.224</td>
<td>0.200</td>
<td>0.572</td>
</tr>
<tr>
<td>MoDRWKV (25% gift-eval)</td>
<td>720-96</td>
<td>0.342</td>
<td>0.746</td>
<td>0.633</td>
<td>0.754</td>
<td>0.559</td>
<td>0.797</td>
<td>0.512</td>
</tr>
<tr>
<td>MoDRWKV (100% gift-eval)</td>
<td>720-96</td>
<td>0.342</td>
<td>0.648</td>
<td>0.453</td>
<td>0.227</td>
<td>0.426</td>
<td>0.203</td>
<td>0.342</td>
</tr>
</tbody>
</table>

**Analysis:**

- MoDRWKV shows solid performance, particularly with full $100\%$ gift-eval dataset, surpassing several baselines on multiple datasets.

- WaveNet encoder outperforms Timer across many tasks, likely due to better capturing long-range dependencies with dilated convolution.

- Increasing training data size improves accuracy and stability, indicating data quantity remains critical.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Vision Encoder Comparison (Table 6)

The authors compare CLIP and SigLIP2 vision encoders across different ModRWKV model sizes:

<table>
<thead>
<tr>
<th>Vision Encoder</th>
<th>Model Size</th>
<th>VQA-v2</th>
<th>TextVQA (VQAT)</th>
<th>GQA</th>
<th>ScienceQA-IMG (SQAI)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">CLIP</td>
<td>0.4B</td>
<td>62.04</td>
<td>31.72</td>
<td>49.32</td>
<td>51.10</td>
</tr>
<tr>
<td>1.5B</td>
<td>72.31</td>
<td>40.27</td>
<td>54.56</td>
<td>62.77</td>
</tr>
<tr>
<td>3B</td>
<td>73.13</td>
<td>45.56</td>
<td>57.00</td>
<td>70.66</td>
</tr>
<tr>
<td rowspan="3">SigLIP2</td>
<td>0.4B</td>
<td>72.04</td>
<td>38.75</td>
<td>55.52</td>
<td>43.32</td>
</tr>
<tr>
<td>1.5B</td>
<td>76.95</td>
<td>44.96</td>
<td>58.88</td>
<td>63.10</td>
</tr>
<tr>
<td>3B</td>
<td>78.30</td>
<td>51.09</td>
<td>60.75</td>
<td>70.93</td>
</tr>
</tbody>
</table>

- SigLIP2 consistently outperforms CLIP across all model sizes and tasks, despite having only ~90M parameters vs. ~300M in CLIP.

- Indicates that encoder pretraining quality and design impact multimodal representation more than size.

### 6.2.2. Effect of Sequence Compression (Table 7)

Evaluations of varying convolution kernel size and stride in Conv1D for sequence length compression on a 1.5B model (SigLIP2 vision encoder):

<table>
<thead>
<tr>
<th>Size</th>
<th>(Kernel, Stride)</th>
<th>Token Length</th>
<th>VQA-v2</th>
<th>TextVQA (VQAT)</th>
<th>GQA</th>
<th>ScienceQA-IMG (SQA1)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">1.5B</td>
<td>(0,0) - no compression</td>
<td>577</td>
<td>76.95</td>
<td>44.96</td>
<td>58.88</td>
<td>63.10</td>
</tr>
<tr>
<td>(3,2)</td>
<td>288</td>
<td>75.21</td>
<td>45.75</td>
<td>58.28</td>
<td>66.02</td>
</tr>
<tr>
<td>(4,3)</td>
<td>192</td>
<td>74.17</td>
<td>44.27</td>
<td>57.53</td>
<td>65.72</td>
</tr>
<tr>
<td>(5,4)</td>
<td>144</td>
<td>73.21</td>
<td>42.65</td>
<td>57.07</td>
<td>65.29</td>
</tr>
</tbody>
</table>

- Moderate compression (to 288 tokens) slightly reduces VQA-v2 accuracy by ~1.74% but improves TextVQA and ScienceQA.

- Higher compression leads to gradual accuracy loss but improves efficiency.

- Trade-off with sequence length is reasonable; compression enables faster inference.

### 6.2.3. Effect of Pretrained Weights (Table 8)

Comparison of two pretrained RWKV7 checkpoints (`base` and $g1$) on 0.4B and 1.5B models:

<table>
<thead>
<tr>
<th>Size</th>
<th>Model</th>
<th>VQAv2</th>
<th>VQAT</th>
<th>GQA</th>
<th>ScienceQA-IMG</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">0.4B</td>
<td>base</td>
<td>72.04</td>
<td>38.75</td>
<td>55.52</td>
<td>43.32</td>
</tr>
<tr>
<td>g1</td>
<td>73.21</td>
<td>41.13</td>
<td>57.34</td>
<td>55.58</td>
</tr>
<tr>
<td rowspan="2">1.5B</td>
<td>base</td>
<td>76.95</td>
<td>44.96</td>
<td>58.88</td>
<td>63.10</td>
</tr>
<tr>
<td>g1</td>
<td>77.87</td>
<td>50.91</td>
<td>60.18</td>
<td>64.63</td>
</tr>
</tbody>
</table>

- The $g1$ weights, obtained via posttraining on "think"-style data, improve all metrics, especially ScienceQA-IMG by ~28% relative increase.

- Shows that pretraining strategies impact multimodal reasoning power significantly.

### 6.2.4. Adapter Scaling Study (Table 5)

Adapter hidden dimension scaling affects time series forecasting performance. A scaling of $4 \times$ input dimension yields the best mean squared error across multiple datasets.

## 6.3. Qualitative Evaluation

The paper includes interactive examples showing that ModRWKV can:

- Integrate visual information and general knowledge for question answering.

- Perform logical reasoning about images (e.g., counting objects).

- Process speech queries and respond with coherent answers.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary

- This paper introduces **ModRWKV**, the first RNN-based multimodal large language model framework built on the RWKV7 linear RNN architecture.

- ModRWKV successfully processes vision, audio, and time series data through adaptable encoders and lightweight MLP adapters that interface with the RWKV backbone.

- The methodology reduces inference cost from quadratic (typical of Transformers) to linear time, improving efficiency without sacrificing accuracy.

- Extensive experiments demonstrate that ModRWKV achieves competitive (and sometimes superior) performance against Transformer-based MLLMs of larger sizes across multiple benchmarks.

- The use of pretrained RWKV weights significantly improves multimodal learning.

- Systematic architectural exploration revealed optimal configuration points for compression, adapter scaling, and encoder choice.

- ModRWKV challenges the prevailing belief that multimodal fusion requires Transformer-based architectures, positioning RNNs as viable alternatives.

## 7.2. Limitations & Future Work

- The current work mainly explores **bi-modal fusion** (e.g., vision-text, audio-text) or univariate scenarios; it does not yet extend to more complex **tri-modal or higher multimodal fusion** involving simultaneous speech, vision, and language.

- The authors note that future work will target richer multimodal settings, exploring more intricate fusion techniques and model enhancements.

- The pretrained encoders for some modalities remain frozen, limiting end-to-end multimodal optimization.

- Model scaling above 3B parameters and leveraging larger pretrained RWKV weights remain open areas.

## 7.3. Personal Insights & Critique

- **Innovation:** ModRWKV is a pioneering step in challenging Transformer dominance in multimodal LLMs by demonstrating that RNNs, previously thought insufficient for complex fusion, can compete effectively.

- **Efficiency Gains:** Linear inference time promises scalability to longer sequences and more modalities, crucial for deployment in resource-constrained or real-time scenarios.

- **Modularity:** The explicit plug-and-play modality encoder design adds flexibility, enabling future research in encoder upgrade, modality addition, or domain adaptation with minimal backbone changes.

- **Pretraining Emphasis:** The strong influence of RWKV7 pretrained weights on multimodal understanding suggests that future research should focus on multimodal pretraining for RNNs.

- **Potential for Extension:** The methodology could be adapted to other fields requiring efficient multimodal fusion, like robotics, medical diagnosis combining images and sensor data, or cross-modal retrieval.

- **Areas for Improvement:**

  - The modality adapter uses a simple MLP, possibly limiting cross-modal interaction capacity; future work could explore more expressive fusion layers.

  - Frozen encoders restrict adaptation; end-to-end training or encoder fine-tuning might yield further gains.

  - Expanding to tri-modal or video modalities could test the limits of this approach.

- **Educational Value:** The paper carefully balances rigorous formula presentation with intuition, valuable for learners exploring alternatives to Transformers in the multimodal context.

  ---

**Overall, ModRWKV advances understanding of how efficient linear RNN architectures can power multimodal language models, opening avenues for scalable, resource-friendly MLLMs beyond Transformers.**
