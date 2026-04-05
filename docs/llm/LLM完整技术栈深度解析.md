# LLM 完整技术栈深度解析

> 如果说深度学习是一座大厦，那么 Transformer 就是地基，预训练是打地基的过程，微调是装修，推理部署是把房子交付给住户，而评估体系则是验收标准。本文将带你从地基到验收，系统走完 LLM 的完整技术链路。

---

## 目录

- [一、Transformer 架构演进](#一transformer-架构演进)
- [二、预训练](#二预训练)
- [三、微调技术](#三微调技术)
- [四、推理与部署](#四推理与部署)
- [五、评估体系](#五评估体系)
- [六、技术选型决策指南](#六技术选型决策指南)
- [附录：面试高频考点速查](#附录面试高频考点速查)
- [附录 B：LLM 技术发展时间线](#附录-bllm-技术发展时间线)
- [附录 C：常用工具与资源速查](#附录-c常用工具与资源速查)

---

## 一、Transformer 架构演进

### 1.1 三大架构流派：谁赢了，为什么？

Transformer 自 2017 年诞生以来，衍生出三大架构流派：

```
┌─────────────────────────────────────────────────────┐
│              Transformer (2017)                      │
│         Encoder      +      Decoder                  │
└─────────┬──────────────────────┬────────────────────┘
          │                      │
    ┌─────▼─────┐         ┌─────▼─────┐
    │ Encoder-  │         │ Decoder-  │    ┌──────────────┐
    │   Only    │         │   Only    │    │  Encoder-    │
    │  (BERT)   │         │  (GPT)    │    │  Decoder     │
    │           │         │           │    │  (T5/BART)   │
    └───────────┘         └───────────┘    └──────────────┘
    双向注意力             单向因果注意力      交叉注意力
    NLU 为主               NLG 为主          翻译/摘要
```

**三者核心区别：**

| 特征 | Encoder-Only (BERT) | Decoder-Only (GPT) | Encoder-Decoder (T5) |
|------|-------------------|-------------------|---------------------|
| 注意力方向 | 双向（Bidirectional） | 单向因果（Causal） | Encoder 双向 + Decoder 因果 |
| 预训练目标 | Masked LM | Next Token Prediction | Span Corruption / Seq2Seq |
| 核心能力 | 理解（分类、NER） | 生成（续写、对话） | 条件生成（翻译、摘要） |
| 代表模型 | BERT, RoBERTa, DeBERTa | GPT-2/3/4, LLaMA, Qwen | T5, BART, Flan-T5 |

> 🎯 **面试重点**：为什么 Decoder-Only 最终胜出？

**Decoder-Only 胜出的四个关键原因：**

1. **统一的生成范式**：所有任务都可以统一为 "给定前文，预测下一个 token" 的形式。分类、问答、翻译……都是生成问题。这种范式统一性极大简化了多任务处理。

2. **Scaling 效率更高**：在相同计算预算下，Decoder-Only 模型的 scaling 曲线更优。因为每个 token 位置都在做有效的预测训练，而 BERT 的 Masked LM 只有 15% 的 token 参与损失计算。

3. **In-Context Learning 的涌现**：GPT-3 发现了 In-Context Learning 能力——模型可以通过 prompt 中的几个示例学会新任务，而不需要梯度更新。这项能力在 Encoder-Only 模型上表现很弱。

4. **KV Cache 友好**：因果注意力天然支持增量推理——已生成 token 的 KV 可以缓存复用，推理时只需计算新 token 的注意力。BERT 的双向注意力每次都要全量计算。

**但 Encoder-Only 并非无用：**
- BERT 系列在**判别式任务**（分类、NER、语义匹配）上仍有优势
- 在 embedding 生成（如 RAG 的检索阶段）中，双向编码器仍是首选
- 实际工程中常见组合：**BERT 做检索 + GPT 做生成**

### 1.2 现代 LLM 关键架构改进

从 GPT-2 到 LLaMA-3 / Qwen-2.5，Decoder-Only 架构经历了一系列精心设计的改进。这些改进看似微小，但累积效果非常显著。

#### 1.2.1 GQA — Grouped-Query Attention

> 🎯 **面试重点**

标准 Multi-Head Attention（MHA）中，每个注意力头都有独立的 Q、K、V 投影矩阵。假设 32 个 head，就有 32 套 KV。

**问题**：推理时 KV Cache 的显存占用与 head 数成正比，成为长序列推理的瓶颈。

**解决方案演进**：

```
MHA (Multi-Head Attention)      MQA (Multi-Query Attention)     GQA (Grouped-Query Attention)
┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────────┐
│ Q₁ Q₂ Q₃ ... Q₃₂   │        │ Q₁ Q₂ Q₃ ... Q₃₂   │        │ Q₁ Q₂ Q₃ ... Q₃₂   │
│ K₁ K₂ K₃ ... K₃₂   │        │      K₁ (shared)     │        │ K₁   K₁    K₂   K₂  │
│ V₁ V₂ V₃ ... V₃₂   │        │      V₁ (shared)     │        │ V₁   V₁    V₂   V₂  │
└──────────────────────┘        └──────────────────────┘        └──────────────────────┘
KV Cache: 32 套                  KV Cache: 1 套                  KV Cache: 8 套 (4Q per group)
质量: ★★★★★                     质量: ★★★☆☆                     质量: ★★★★★
效率: ★★☆☆☆                     效率: ★★★★★                     效率: ★★★★☆
```

- **MQA**（Multi-Query Attention）：所有 Q head 共享 1 套 KV。KV Cache 减少到 1/32，但质量下降明显。
- **GQA**（Grouped-Query Attention）：将 Q 分组（如 32 → 8 组，每组 4 个 Q 共享 1 套 KV）。KV Cache 减少到 1/4，质量损失极小。

**LLaMA-2 70B 首次采用 GQA，之后成为标配。** LLaMA-3、Qwen-2、Mistral 系列全部使用 GQA。

**KV Cache 显存计算公式**：

$$\text{KV Cache} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times \text{batch} \times \text{dtype\_bytes}$$

以 LLaMA-3-8B 为例（8 KV heads, head_dim=128, FP16）：
- 4K 上下文, batch=1: $2 \times 8 \times 128 \times 4096 \times 1 \times 2 = 16 \text{ MB}$
- 128K 上下文, batch=1: $2 \times 8 \times 128 \times 131072 \times 1 \times 2 = 512 \text{ MB}$

#### 1.2.2 SwiGLU — 激活函数升级

传统 Transformer 使用 ReLU 激活的 FFN（Feed-Forward Network）：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

LLaMA 引入 **SwiGLU**：

$$\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xV) W_2$$

其中 $\text{Swish}(x) = x \cdot \sigma(\beta x)$，$\odot$ 表示逐元素乘法。

**关键差异**：
- 引入门控机制（$xV$ 是 gate），让网络学会"选择性激活"
- 虽然增加了 1/3 的参数（多了 $V$ 矩阵），但在相同计算预算下效果更好
- PaLM 论文的 ablation 显示 SwiGLU 比 ReLU 在困惑度上好约 1-2 个点

#### 1.2.3 RMSNorm — 更简洁的归一化

LayerNorm 的计算：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

RMSNorm 去掉了均值偏移（re-centering），只做缩放：

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

**优势**：
- 计算量减少约 10-15%（去掉均值计算和偏置项）
- 实验证明效果与 LayerNorm 相当甚至更好
- 所有现代 LLM（LLaMA、Qwen、Mistral）都采用 **Pre-RMSNorm**（在注意力/FFN 之前做归一化，而非之后）

**Pre-Norm vs Post-Norm**：Pre-Norm 的残差连接更稳定，训练深层网络（40+ 层）时更容易收敛。这也是现代 LLM 的标配。

#### 1.2.4 RoPE — 旋转位置编码

> 🎯 **面试重点**

原始 Transformer 使用绝对位置编码（加法），信息容易被后续层"冲淡"。

**RoPE 核心思想**：用旋转矩阵对 Q、K 向量进行编码，使得内积自然包含相对位置信息。

$$\text{RoPE}(x_m, m) = x_m \cdot e^{im\theta}$$

直觉理解：把位置 $m$ 处的向量在复平面上旋转 $m\theta$ 角度。两个位置的向量做内积时，结果只取决于相对距离 $m - n$。

```
Position 0:  向量旋转 0°
Position 1:  向量旋转 θ°
Position 2:  向量旋转 2θ°
...
Position m:  向量旋转 mθ°

Attention(pos_m, pos_n) ∝ cos((m-n)θ)  ← 只依赖相对位置
```

**RoPE 的优势**：
1. 相对位置信息直接嵌入注意力计算，不占额外参数
2. 支持长度外推（extrapolation）——配合 NTK-aware 插值，可在微调后扩展窗口
3. 计算高效——只是旋转操作，对 GPU 友好

### 1.3 MoE 架构：稀疏激活的扩展之路

> 🎯 **面试重点**（2024-2025 热门话题，DeepSeek 系列全面采用）

**核心思想**：用多个"专家"（Expert，每个是一个 FFN）替代单一 FFN，每次推理只激活其中少数几个。

```
                    输入 x
                      │
                ┌─────▼─────┐
                │   Router   │  ← 门控网络，决定激活哪些专家
                │  (Softmax) │
                └──┬──┬──┬──┘
                   │  │  │
           ┌───────┘  │  └───────┐
           ▼          ▼          ▼
      ┌─────────┐ ┌─────────┐ ┌─────────┐
      │Expert 1 │ │Expert 2 │ │Expert N │  ← 只激活 Top-K 个
      │  (FFN)  │ │  (FFN)  │ │  (FFN)  │
      └────┬────┘ └────┬────┘ └────┬────┘
           │          │          │
           └──────┬───┘          │  (inactive experts skipped)
                  ▼
             加权求和 → 输出
```

**关键参数**（以 DeepSeek-V3 为例）：
- 总专家数 N = 256，每次激活 Top-K = 8
- 额外 1 个共享专家（shared expert），始终参与计算
- 总参数 671B，但每次推理只激活约 37B

**MoE 的核心挑战——负载均衡（Load Balancing）**：

如果不加干预，Router 容易陷入"赢家通吃"——少数专家被过度使用，其余专家退化为零。

解决方案：

| 方法 | 原理 | 代表作 |
|------|------|--------|
| Auxiliary Loss | 在训练损失中加入负载均衡正则项 | Switch Transformer |
| Expert Capacity | 限制每个专家处理的 token 上限，超出丢弃或溢出 | GShard |
| Auxiliary-Loss-Free | 引入 bias 项自适应调整，不需要额外损失函数 | DeepSeek-V3 |

**DeepSeek-V3 的 Auxiliary-Loss-Free 方案**特别值得关注：
- 传统 auxiliary loss 需要调 loss 权重超参，权重太小不起作用，太大影响模型效果
- DeepSeek-V3 给每个专家加一个可学习的 bias 项，在路由时加到 logits 上
- 训练中自适应调整 bias，使得负载自然趋于均衡，且不影响模型质量

**MoE 的工程挑战**：
- **All-to-All 通信**：不同 GPU 上的 token 需要路由到不同 GPU 上的专家，产生大量通信开销
- **显存放大**：虽然激活参数少，但总参数需要全部加载，显存并未按比例减少
- **量化困难**：不同专家的权重分布可能差异大，统一量化策略效果不佳

### 1.4 长上下文技术：从 2K 到 1M

> 🎯 **面试重点**

标准 Transformer 的注意力计算复杂度为 $O(L^2)$，窗口扩展面临计算和位置编码两大挑战。

#### 技术路线总览

```
2K (GPT-2)  →  4K (GPT-3)  →  32K (GPT-4)  →  128K (Claude-3)  →  1M+ (Gemini 1.5)

关键技术：
├── 位置编码扩展
│   ├── ALiBi (线性偏置)
│   ├── NTK-aware RoPE 插值
│   └── YaRN (NTK + 注意力缩放)
├── 注意力机制优化
│   ├── Flash Attention (IO 感知计算)
│   ├── Ring Attention (序列并行)
│   └── Sliding Window Attention (Mistral)
└── 训练策略
    ├── 长文本续训 (Long-context continual pretraining)
    └── 渐进式长度扩展
```

**ALiBi**（Attention with Linear Biases）：
- 不使用位置编码，而是在注意力分数上加线性偏置：$\text{score}(i, j) = q_i \cdot k_j - m \cdot |i - j|$
- $m$ 是每个 head 的固定斜率，距离越远惩罚越大
- 优点：训练窗口短也能外推到长序列；缺点：长距离信息衰减过快

**YaRN**（Yet another RoPE extensioN）：
- 结合 NTK-aware 插值 + 注意力温度缩放
- 在 RoPE 的频率维度上做分段处理：低频（长距离信息）做插值，高频（局部信息）保持不变
- LLaMA-3.1 使用类似技术将 8K 窗口扩展到 128K

**Ring Attention**：
- 核心思想：将长序列切分到多个 GPU，每个 GPU 计算局部注意力，Q 块固定、KV 块以环形方式在 GPU 间传递
- 计算和通信重叠，理论上可以把上下文长度扩展到 GPU 数量 × 单 GPU 窗口长度
- Gemini 1.5 Pro 的 1M 上下文就依赖类似的序列并行技术

**Flash Attention**（虽然不是纯"长上下文"技术，但却是根基）：
- 标准注意力需要把 $L \times L$ 的注意力矩阵写入 HBM（GPU 显存），IO 成为瓶颈
- Flash Attention 通过 tiling（分块计算）+ online softmax，将注意力计算完全在 SRAM 中完成
- 不存储完整注意力矩阵，IO 复杂度从 $O(L^2)$ 降到 $O(L^2 d / M)$（$M$ 为 SRAM 大小）
- Flash Attention 2/3 进一步优化了并行度和工作分配

---

## 二、预训练

### 2.1 预训练数据工程

> 🎯 **面试重点**：数据质量决定模型质量上限，算法只是逼近这个上限。

#### 完整数据流水线

```
数据采集           数据清洗           数据去重           质量过滤          数据配比
┌──────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│Common│    │URL/HTML 过滤 │    │MinHash   │    │Perplexity│    │语言配比   │
│Crawl │───▶│语言检测      │───▶│SimHash   │───▶│分类器评分  │───▶│领域配比   │
│书籍   │    │毒性/PII过滤  │    │Exact Dedup│   │人工抽检   │    │格式配比   │
│代码   │    │格式标准化    │    │Fuzzy Dedup│    │GPT评分   │    │合成数据   │
│学术   │    │长度过滤      │    │           │    │         │    │          │
└──────┘    └─────────────┘    └──────────┘    └──────────┘    └──────────┘
~100T tokens   ~30T tokens      ~15T tokens     ~5T tokens      ~2T tokens
```

**各环节关键细节**：

**1. 数据采集**：
- Common Crawl 是最大的开放语料来源（每月 PB 级别），但质量参差不齐
- 高质量数据来源：Wikipedia、Books（BookCorpus、Gutenberg）、arXiv、GitHub 代码、StackExchange
- 中文数据相对稀缺——CulturaX、WuDaoCorpora、MNBVC 是主要中文语料

**2. 数据去重**（关键环节）：

| 方法 | 粒度 | 原理 | 优缺点 |
|------|------|------|--------|
| Exact Dedup | 文档/段落 | SHA-256 哈希完全匹配 | 快速但只能处理完全相同的文本 |
| MinHash LSH | 文档 | N-gram 的 MinHash 签名 + LSH 分桶近似匹配 | 工业界最常用，可调阈值（如 Jaccard > 0.8） |
| SimHash | 文档 | 将文档映射为定长二进制串，比较汉明距离 | 计算快，但精度低于 MinHash |
| Suffix Array | 子串 | 在 suffix array 上查找重复子串 | 可以发现文档内部的重复段落 |
| SemDedup | 语义 | 用 embedding 模型做语义级去重 | 质量最高但计算成本极大 |

**3. 质量过滤的实用策略**：
- **困惑度过滤**：用一个小语言模型（如 KenLM）计算困惑度，过滤掉困惑度极高（乱码）或极低（重复内容）的文档
- **分类器过滤**：训练一个二分类器（如 fastText），区分"高质量"（Wikipedia-like）和"低质量"文本
- **Heuristic 规则**：行均字符数、特殊字符比例、大写比例、stopword 比例等
- **GPT 评分**：用大模型对数据样本打分（LLaMA-3 的训练数据就大量使用了此方法）

**4. 数据配比（Data Mixture）**：

| 数据类型 | 典型配比 | 说明 |
|---------|---------|------|
| 通用网页 | 50-60% | 经过严格清洗的 Common Crawl |
| 代码 | 15-20% | 提升推理能力的关键——代码训练对逻辑推理有正迁移 |
| 书籍/学术 | 10-15% | 长文本理解、知识密度 |
| 百科/知识库 | 5% | 高质量知识 |
| 对话/问答 | 5% | 对话能力基础 |
| 数学 | 3-5% | 数学推理能力 |
| 多语言 | 5-10% | 取决于目标语言覆盖 |

> **工程经验**：代码数据的配比对最终模型的推理能力影响巨大。很多研究发现，在预训练中加入 15-20% 的代码数据，即使是非代码任务的表现也会提升。这可能因为代码的逻辑结构训练了模型的推理能力。

### 2.2 预训练目标

三种主要预训练目标对比：

| 目标 | 公式 | 特点 | 代表模型 |
|------|------|------|---------|
| Causal LM | $P(x_t \| x_{<t})$ | 从左到右逐 token 预测 | GPT 系列 |
| Masked LM | $P(x_{\text{mask}} \| x_{\text{context}})$ | 随机遮蔽 15% token 进行预测 | BERT |
| Prefix LM | Prefix 部分双向 + 生成部分因果 | 混合模式 | UniLM, GLM |

**为什么 Causal LM 成为主流？**

1. **训练效率**：每个 token 都参与损失计算，而 Masked LM 只有 ~15% 的 token 计算 loss
2. **生成自然**：训练和推理目标一致（都是 next token prediction）
3. **Scaling 表现**：在大规模上 Causal LM 的 scaling 曲线更优

### 2.3 Scaling Laws：越大越好？

> 🎯 **面试重点**

**Kaplan Scaling Law**（OpenAI, 2020）：

$$L(N, D, C) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_{\infty}$$

发现：模型参数量 $N$、训练数据量 $D$、计算量 $C$ 三者与 Loss 呈幂律关系。在固定计算预算下，应该优先增大模型。

**Chinchilla 定律**（DeepMind, 2022）——**修正了上述结论**：

- **核心发现**：最优方案是模型参数和训练 token 数等比例增长
- **经验公式**：最优训练 token 数 ≈ 20 × 模型参数量
- 70B 模型应训练 1.4T tokens，而非 GPT-3 的 300B tokens

| 模型 | 参数量 | 训练 tokens | tokens/params | Chinchilla 最优？ |
|------|--------|-------------|---------------|------------------|
| GPT-3 | 175B | 300B | 1.7× | ❌ 严重 under-trained |
| Chinchilla | 70B | 1.4T | 20× | ✅ |
| LLaMA-1 65B | 65B | 1.4T | 21.5× | ✅ |
| LLaMA-2 70B | 70B | 2T | 28.6× | Over-trained（有意为之） |
| LLaMA-3 8B | 8B | 15T | 1875× | 极度 over-trained |

**现代趋势**：刻意 over-train 小模型。因为训练是一次性成本，但推理是持续成本。用更多数据训练更小模型，推理时更便宜。LLaMA-3-8B 用了 15T tokens 训练，大幅超过 Chinchilla 定律的建议，但推理成本远低于 70B 模型且效果接近。

### 2.4 分布式训练

> 🎯 **面试重点**

当模型大到单卡装不下，或者要高效利用集群时，需要分布式训练。核心有三种并行策略（通常组合使用）：

```
┌────────────────────────────────────────────────┐
│              3D 并行                            │
│                                                │
│   ┌──────────────────────────────────────┐     │
│   │  数据并行 (Data Parallelism, DP)      │     │
│   │  每张卡有完整模型副本                    │     │
│   │  不同卡处理不同 mini-batch              │     │
│   │  梯度 AllReduce 同步                   │     │
│   │                                      │     │
│   │  DP → DDP → FSDP/ZeRO               │     │
│   └──────────────────────────────────────┘     │
│   ┌──────────────────────────────────────┐     │
│   │  张量并行 (Tensor Parallelism, TP)    │     │
│   │  将单层的矩阵运算切分到多卡             │     │
│   │  如 Column Parallel + Row Parallel    │     │
│   │  需要 AllReduce 同步中间结果           │     │
│   │  ⚠️ 通信密集，要求高速互联 (NVLink)    │     │
│   └──────────────────────────────────────┘     │
│   ┌──────────────────────────────────────┐     │
│   │  流水线并行 (Pipeline Parallelism, PP)│     │
│   │  按层切分模型到不同卡                   │     │
│   │  micro-batch 流水调度                  │     │
│   │  存在 bubble（空闲时间）              │     │
│   │  1F1B 调度减少 bubble                  │     │
│   └──────────────────────────────────────┘     │
└────────────────────────────────────────────────┘
```

**DDP vs FSDP vs ZeRO**（数据并行的演进）：

| 方案 | 模型副本 | 优化器状态 | 梯度 | 通信量 | 显存效率 |
|------|---------|-----------|------|--------|---------|
| DDP | 完整副本/每卡 | 完整/每卡 | AllReduce | $2\Phi$ | 低 |
| ZeRO-1 | 完整副本/每卡 | **分片** | AllReduce | $2\Phi$ | 中 |
| ZeRO-2 | 完整副本/每卡 | **分片** | **分片** | $2\Phi$ | 中高 |
| ZeRO-3 / FSDP | **分片** | **分片** | **分片** | $3\Phi$ | 高 |

$\Phi$ = 模型参数量。ZeRO-3 通信量虽然增加 50%（需要额外 All-Gather 参数），但显存接近线性切分。

**工程实践中的典型配置**（以 LLaMA-70B 训练为例）：

```
集群: 256 × A100-80G
- TP = 8 (8 卡一组做张量并行，走 NVLink)
- PP = 4 (4 个 TP 组做流水线并行，走 IB)
- DP = 8 (8 个完整 pipeline 做数据并行，走 IB)
总并行度 = 8 × 4 × 8 = 256
```

### 2.5 训练稳定性

大模型训练动辄数周到数月，训练稳定性至关重要。

**Loss Spike 问题**：
- 训练过程中 loss 突然飙升，有时能自行恢复，有时导致发散
- 常见原因：数据中的异常样本、学习率过大、梯度爆炸
- 应对策略：
  - **梯度裁剪**（Gradient Clipping）：$\|\nabla\| > \text{threshold}$ 时按比例缩放
  - **跳过异常 batch**：监控 loss，超出阈值时丢弃该 batch
  - **从 checkpoint 回滚**：遇到严重 spike 时回退若干步

**学习率调度**：

```
         WSD 调度 (Warmup-Stable-Decay)
         
学习率 │      ┌──────────────────────────┐
       │    ╱ │           Stable          │╲
       │   ╱  │                            │ ╲
       │  ╱   │                            │  ╲
       │ ╱    │                            │   ╲
       │╱     │                            │    ╲
       └──────┴────────────────────────────┴─────▶ 训练步数
       Warmup        恒定学习率              Decay
       (~2000步)     (主要阶段)             (余弦衰减)
```

现代 LLM 训练普遍采用 **WSD**（Warmup-Stable-Decay）调度：
1. **Warmup**：从 0 线性增到峰值（如 3e-4），约 2000 步
2. **Stable**：保持峰值不变，这是训练的主体阶段
3. **Decay**：余弦衰减到峰值的 1/10

**其他稳定性技巧**：
- **BF16 混合精度**：比 FP16 更稳定（更大的指数范围），LLaMA-3 全程使用 BF16
- **Gradient Accumulation**：在 micro-batch 较小时累积多步梯度，等效增大 batch size
- **Z-Loss**：对 logits 的 log-sum-exp 加正则，防止 softmax 数值不稳定

---

## 三、微调技术

微调是将通用大模型适配到特定任务的关键步骤。从简单到复杂，微调技术可以分为以下层次：

```
                    成本/复杂度 ↑
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │Prompt   │   │  LoRA /   │   │   全参    │
    │Tuning   │   │  QLoRA    │   │   微调    │
    │         │   │           │   │  + RLHF   │
    └─────────┘   └───────────┘   └───────────┘
    冻结模型        冻结大部分参数      全部参数更新
    最低成本        性价比最高 ⭐      效果上限最高
```

### 3.1 SFT — 指令微调

> 🎯 **面试重点**

SFT（Supervised Fine-Tuning）是让模型从"文本补全"变为"遵循指令"的关键步骤。

**训练数据格式**：

```json
{
  "instruction": "请将以下英文翻译成中文",
  "input": "The quick brown fox jumps over the lazy dog.",
  "output": "那只敏捷的棕色狐狸跳过了那只懒狗。"
}
```

**核心要点**：

1. **Loss 只计算 output 部分**：instruction 和 input 部分不计算 loss（mask 掉），只训练模型生成正确的回复
2. **数据质量 >>> 数据数量**：LIMA 论文（"Less Is More for Alignment"）证明 1000 条高质量 SFT 数据就能显著提升指令遵循能力
3. **多样性很重要**：覆盖不同任务类型（问答、翻译、摘要、代码、数学、角色扮演等）

**SFT 数据构建最佳实践**：

| 数据来源 | 优缺点 | 建议 |
|---------|--------|------|
| 人工标注 | 质量最高，成本最贵 | 核心场景必须有 |
| Self-Instruct | 用大模型生成指令+回复 | 规模化的首选方案 |
| ShareGPT / 对话日志 | 真实用户交互，多样性好 | 需要脱敏和质量过滤 |
| Evol-Instruct | 逐步增加指令复杂度 | 提升模型处理复杂指令的能力 |

**实战参数参考**（7B 模型 SFT）：
- 数据量：5K-50K 条高质量数据
- 学习率：1e-5 到 2e-5
- Epoch：2-3（过多会过拟合）
- Batch size：128（通过 gradient accumulation）

### 3.2 RLHF — 基于人类反馈的强化学习

> 🎯 **面试重点**

RLHF 是 ChatGPT 的核心技术之一。目标：让模型的输出与人类偏好对齐（Alignment）。

**三步流程**：

```
Step 1: SFT                   Step 2: RM Training           Step 3: PPO Training
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│ 指令微调          │          │ 训练奖励模型       │          │ 用 PPO 优化策略    │
│                  │          │                  │          │                  │
│ (instruction,    │          │ prompt → 生成多个  │          │ π_θ(y|x) 策略模型 │
│  response) 对    │─────▶    │  response → 人类  │─────▶    │ R(x,y) 奖励信号   │
│                  │          │  排序 → 训练 RM   │          │ PPO clipping     │
│ 产出: SFT Model  │          │                  │          │                  │
│                  │          │ 产出: Reward Model │          │ 产出: RLHF Model  │
└──────────────────┘          └──────────────────┘          └──────────────────┘
```

**Step 2: Reward Model（RM）训练细节**：

偏好数据格式：$(x, y_w, y_l)$ —— 同一 prompt $x$，人类认为 $y_w$ 优于 $y_l$。

RM 的训练目标（Bradley-Terry 模型）：

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(R(x, y_w) - R(x, y_l)) \right]$$

直觉：让好回复的分数高于差回复的分数。

**Step 3: PPO 训练细节**：

PPO 的优化目标：

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(x, y) \sim \pi_\theta} \left[ R(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

- $R(x, y)$：RM 给出的奖励
- $\text{KL}(\pi_\theta \| \pi_{\text{ref}})$：当前策略和 SFT 模型的 KL 散度
- $\beta$：控制 KL 惩罚强度，防止模型偏离太远（reward hacking）

**RLHF 的痛点**：
1. **系统复杂度高**：需要同时维护 4 个模型（SFT, RM, Policy, Reference），显存压力大
2. **训练不稳定**：PPO 超参敏感，reward hacking（模型学会骗奖励模型而非真正提升质量）
3. **标注成本高**：需要大量人类偏好标注数据
4. **RM 质量瓶颈**：RM 不完美，错误会传导到策略模型

### 3.3 DPO — 直接偏好优化

> 🎯 **面试重点**

DPO 的核心洞察：**可以跳过 RM 训练和 PPO，直接用偏好数据优化策略模型**。

数学推导表明，RLHF 的最优策略可以用以下封闭形式表示：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{R(x,y)}{\beta}\right)$$

反解出隐式奖励：

$$R(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

代入 Bradley-Terry 模型，$Z(x)$ 消去，得到 DPO 的损失函数：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**DPO vs RLHF 对比**：

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 需要 RM | ✅ 需要单独训练 | ❌ 不需要 |
| 模型数量 | 4 个（SFT + RM + Policy + Ref） | 2 个（Policy + Ref） |
| 训练稳定性 | 差，PPO 超参敏感 | 好，类似标准 SFT |
| 实现复杂度 | 高 | 低 |
| 效果上限 | 更高（在线学习，持续探索） | 略低（离线学习，受限于数据） |
| 适用场景 | 大规模对齐、追求极致效果 | 中小规模、快速迭代 |

**DPO 的局限**：
- 离线方法——只能从固定的偏好数据中学习，不能像 PPO 那样在线探索
- 对偏好数据质量敏感——如果偏好标注有噪声，DPO 容易过拟合到噪声
- 可能存在"长度偏好"——倾向于生成更长的回复

### 3.3.1 GRPO — 组相对策略优化

> 🎯 **面试重点**（DeepSeek-R1 核心技术）

**GRPO**（Group Relative Policy Optimization）是 DeepSeek 提出的一种更简洁的对齐方法，是 DeepSeek-R1 训练的核心技术。

**核心思想**：不需要训练单独的奖励模型（RM），而是通过**组内采样和相对排序**来计算优势函数（advantage）。

**GRPO 的工作流程**：

```
对于每个 prompt x:
1. 从当前策略 π_θ 采样 G 个回复: {y₁, y₂, ..., y_G}
2. 用规则/验证器对每个回复打分: {r₁, r₂, ..., r_G}
3. 在组内计算归一化优势:
   Â_i = (r_i - mean(r)) / std(r)
4. 用优势函数更新策略（类似 PPO 的 clipping）
```

**GRPO 的损失函数**：

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{x} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \hat{A}_i, \text{clip}(\cdot) \hat{A}_i \right) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

**GRPO vs RLHF vs DPO 三方对比**：

| 维度 | RLHF (PPO) | DPO | GRPO |
|------|-----------|-----|------|
| 需要 RM | ✅ 需要 | ❌ 不需要 | ❌ 不需要 |
| 奖励信号 | RM 打分 | 隐式（偏好对） | 规则/验证器打分 |
| 采样方式 | 在线采样 | 离线数据 | 组内多次采样 |
| 模型数量 | 4 个 | 2 个 | 2 个（策略 + 参考） |
| 适用场景 | 通用对齐 | 通用对齐 | **可验证任务**（数学、代码） |
| 代表作 | ChatGPT | Zephyr | **DeepSeek-R1** |

**GRPO 的关键优势**：

1. **无需 RM**：省去了训练奖励模型的复杂步骤和成本
2. **适合可验证任务**：数学题有标准答案、代码可以执行验证，天然适合用规则打分
3. **组内相对排序**：通过组内归一化，自动适应不同难度的 prompt，避免绝对分数的校准问题
4. **训练更稳定**：相比 PPO，超参数更少，训练过程更稳定

**GRPO 的局限**：
- 依赖可验证的奖励信号——对于开放性对话、创意写作等无标准答案的任务，需要额外设计评分规则
- 组内采样需要多次前向传播，计算成本高于 DPO
- 在非推理类任务上的效果尚未被充分验证

**DeepSeek-R1 的训练流程**：

```
DeepSeek-V3-Base
      │
      ▼
  冷启动 SFT (少量高质量 CoT 数据)
      │
      ▼
  GRPO 强化学习 (数学 + 代码 + 推理任务)
      │  ← 奖励信号: 答案正确性 + 格式规范性
      ▼
  拒绝采样 + SFT (用 RL 模型生成高质量数据)
      │
      ▼
  第二轮 GRPO (全场景对齐)
      │
      ▼
  DeepSeek-R1
```

> **关键洞察**：DeepSeek-R1 证明了一个重要结论——**通过纯强化学习（不依赖蒸馏），模型可以自发涌现出 Chain-of-Thought 推理能力**。这对 AI 研究有深远影响。

### 3.4 LoRA / QLoRA — 低秩适配

> 🎯 **面试重点**

**LoRA 核心思想**：模型微调时，权重更新矩阵 $\Delta W$ 通常是低秩的。用两个小矩阵 $A$ 和 $B$ 近似表示：

$$W' = W + \Delta W = W + BA$$

其中 $W \in \mathbb{R}^{d \times k}$，$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

```
原始权重 W (d×k)              LoRA 适配
┌───────────────┐            ┌───────────────┐    ┌───┐
│               │            │               │    │   │
│   d × k       │            │   frozen W    │ +  │ B │ × ┌─────────┐
│  (全部冻结)   │     →      │               │    │d×r│   │  A(r×k) │
│               │            │               │    │   │   └─────────┘
└───────────────┘            └───────────────┘    └───┘
参数量: d×k                   可训练参数: d×r + r×k ≈ 2dr (当 r<<k)
如 4096×4096 = 16.7M          如 r=16: 4096×16×2 = 131K (0.78%)
```

**关键超参数**：

| 参数 | 含义 | 推荐值 | 说明 |
|------|------|--------|------|
| r (rank) | 低秩矩阵的秩 | 8-64 | 越大越接近全参微调，但参数量增加 |
| alpha | 缩放因子 | 通常 = 2r | $\Delta W = \frac{\alpha}{r} BA$ |
| target_modules | 应用 LoRA 的层 | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj | 应用到所有线性层效果最好 |
| dropout | LoRA dropout | 0.05 | 防过拟合 |

**QLoRA** — 在 4bit 量化基础上做 LoRA：

```
QLoRA 核心：
1. 基础模型用 NF4 (NormalFloat 4-bit) 量化 → 显存减少 ~75%
2. 在量化模型上插入 LoRA 适配器（FP16/BF16）
3. 反向传播时通过量化层计算梯度，只更新 LoRA 参数
4. 双重量化（Double Quantization）: 对量化常数再量化

结果: 单张 24GB 显卡 (3090/4090) 可以微调 33B 模型
      单张 48GB 显卡 (A6000) 可以微调 65B 模型
```

**LoRA 实战经验**：

1. **r 的选择**：简单任务 r=8 足够；复杂任务（如领域适配）r=32-64 更好；当 r 增大到一定程度后收益递减
2. **target_modules**：只用 q_proj, v_proj 是最低配；加上 k_proj, o_proj 效果明显提升；再加上 FFN 层（gate_proj, up_proj, down_proj）可以逼近全参微调
3. **学习率**：LoRA 的学习率通常比全参微调大 5-10 倍（如 1e-4 到 3e-4）
4. **合并权重**：部署时可以把 LoRA 权重合并回原模型（$W' = W + BA$），不增加推理开销

### 3.4.1 其他参数高效微调方法

除了 LoRA/QLoRA，还有几种重要的参数高效微调（PEFT）方法值得了解：

#### Adapter Tuning

**核心思想**：在 Transformer 每一层的注意力和 FFN 之后插入小型的 Adapter 模块，只训练 Adapter 参数。

```
原始 Transformer 层:              加入 Adapter:
┌──────────────┐                 ┌──────────────┐
│  Attention   │                 │  Attention   │
└──────┬───────┘                 └──────┬───────┘
       │                                │
       │                         ┌──────▼───────┐
       │                         │   Adapter    │ ← 新增
       │                         │  Down (d→r)  │
       │                         │  ReLU        │
       │                         │  Up   (r→d)  │
       │                         └──────┬───────┘
       │                                │ + 残差连接
┌──────▼───────┐                 ┌──────▼───────┐
│     FFN      │                 │     FFN      │
└──────┬───────┘                 └──────┬───────┘
       │                                │
       │                         ┌──────▼───────┐
       │                         │   Adapter    │ ← 新增
       │                         └──────┬───────┘
       ▼                                ▼
```

**Adapter 结构**：Down-projection ($d \to r$) → 非线性激活 → Up-projection ($r \to d$) + 残差连接

**与 LoRA 的关键区别**：

| 维度 | Adapter | LoRA |
|------|---------|------|
| 插入方式 | 串行（在层之后） | 并行（与原始权重并行） |
| 推理开销 | 有额外延迟（多了前向传播层） | **零额外延迟**（可合并回原权重） |
| 参数效率 | 中等 | 更高 |
| 多任务切换 | 需要切换 Adapter 模块 | 可以热切换 LoRA 权重 |

> **工程建议**：由于 LoRA 在推理时可以合并回原权重（零额外开销），且参数效率更高，**LoRA 已基本取代 Adapter 成为主流方案**。但 Adapter 在某些多任务场景（如同时服务多个客户的定制模型）中仍有价值。

#### Prefix Tuning

**核心思想**：在每一层的注意力计算中，在 Key 和 Value 前面拼接一组可学习的"虚拟 token"（prefix），只训练这些 prefix 参数。

```
标准注意力:
Q × [K₁, K₂, ..., Kₙ]ᵀ → Attention Weights → [V₁, V₂, ..., Vₙ]

Prefix Tuning:
Q × [P_K₁, P_K₂, ..., P_Kₘ, K₁, K₂, ..., Kₙ]ᵀ → Weights → [P_V₁, ..., P_Vₘ, V₁, ..., Vₙ]
     └──────── prefix ────────┘                              └──── prefix ────┘
     可学习参数 (每层独立)                                     可学习参数 (每层独立)
```

**与 Prompt Tuning 的区别**：

| 维度 | Prompt Tuning | Prefix Tuning |
|------|-------------|---------------|
| 作用位置 | 仅在输入 embedding 层 | **每一层**的 K、V 前面 |
| 参数量 | 极少（~100K） | 较多（~1-10M） |
| 效果 | 较弱，依赖模型规模 | 更强，小模型也有效 |
| 直觉 | 学习一个"软提示" | 学习每层的"注意力引导" |

**Prefix Tuning 的局限**：
- 占用序列长度——prefix token 会占用有效上下文窗口
- 参数效率不如 LoRA——相同参数量下效果通常不如 LoRA
- 推理时无法消除——不像 LoRA 可以合并回原权重

#### P-Tuning v2

P-Tuning v2 本质上是 Prefix Tuning 的改进版本，由清华大学提出：
- 在每一层都加入可学习的 prefix（与 Prefix Tuning 相同）
- 针对 NLU 任务做了优化，在小模型上也能接近全参微调效果
- 在中文 NLP 社区中使用较广

#### PEFT 方法总览

```
                    参数效率微调方法谱系
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ 加法式   │   │  选择式    │   │  重参数化  │
    │ Additive │   │ Selective │   │ Reparameter│
    └────┬────┘   └─────┬─────┘   └─────┬─────┘
         │               │               │
    ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
    │Adapter  │     │BitFit   │     │LoRA     │
    │Prefix   │     │(只训练   │     │QLoRA    │
    │Tuning   │     │ bias)   │     │DoRA     │
    │P-Tuning │     │         │     │         │
    │Prompt   │     │         │     │         │
    │Tuning   │     │         │     │         │
    └─────────┘     └─────────┘     └─────────┘
```

### 3.5 微调方法对比与选择

| 方法 | 可训练参数 | 显存需求 (7B) | 效果 | 推理开销 | 适用场景 |
|------|-----------|---------------|------|---------|---------|
| Prompt Tuning | ~100K | ~16GB | ⭐⭐ | 无 | 任务明确、数据少、大模型 |
| Prefix Tuning | ~1-10M | ~16GB | ⭐⭐⭐ | 占用序列长度 | NLU 任务、小模型 |
| Adapter | ~5-20M | ~17GB | ⭐⭐⭐ | 有额外延迟 | 多任务服务 |
| LoRA (r=16) | ~20M | ~18GB | ⭐⭐⭐⭐ | **无**（可合并） | **通用推荐方案** ⭐ |
| QLoRA (r=16) | ~20M | ~6GB | ⭐⭐⭐⭐ | 量化推理 | 显卡显存受限 |
| 全参 SFT | ~7B | ~120GB (ZeRO-3) | ⭐⭐⭐⭐⭐ | 无 | 追求极致效果 |
| 全参 SFT + RLHF | ~7B × 4 | ~400GB+ | ⭐⭐⭐⭐⭐+ | 无 | 对齐最高要求 |

> **工程建议**：2024-2025 年的实践中，**LoRA/QLoRA 是绝对的主流选择**。Adapter 和 Prefix Tuning 在学术上有价值，但工程中已被 LoRA 全面取代。只有在追求极致效果时才考虑全参微调。

**选择决策树**：

```
你有多少 GPU 显存？
├── < 24GB (单卡消费级)
│   └── QLoRA (4bit 量化 + LoRA)
├── 24-80GB (单卡专业级)
│   ├── 数据 < 10K → LoRA
│   └── 数据 > 10K + 效果优先 → 全参 SFT
├── 多卡 (2-8 × A100)
│   ├── 追求性价比 → LoRA (多卡加速)
│   └── 追求极致 → 全参 SFT + DPO/RLHF
└── 集群 (16+ × A100)
    └── 全参 SFT + RLHF (PPO)
```

### 3.6 微调常见坑

> 🎯 **面试重点**

| 问题 | 现象 | 原因 | 解决方案 |
|------|------|------|---------|
| **灾难性遗忘** | 微调后通用能力下降 | 在特定领域过拟合 | 混入通用数据（10-20%）；降低学习率；减少 epoch |
| **过拟合** | 训练 loss 低但测试效果差 | 数据量不足或质量差 | 减少 epoch；增大 dropout；增加数据多样性 |
| **格式坍塌** | 回复格式混乱/忽略指令 | Chat template 不一致 | 严格遵循模型原始 chat template |
| **长度偏好** | 倾向于生成过长回复 | RLHF/DPO 数据中长回复分数偏高 | 在奖励函数中加入长度惩罚 |
| **幻觉加重** | 微调后幻觉增多 | 训练数据中包含错误信息 | 数据质量审核；加入"我不知道"样本 |
| **Chinese-English 混杂** | 中文回复中夹英文 | 预训练数据以英文为主 | 增加纯中文 SFT 数据比例 |
| **LoRA 秩不够** | 微调效果不理想 | 任务复杂度超出低秩假设 | 增大 r；扩大 target_modules |

**特别注意 Chat Template**：

不同模型有不同的对话模板，格式不匹配会导致严重的效果下降：

```
# LLaMA-3 格式
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Hello!<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

# Qwen-2 格式
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

---

## 四、推理与部署

### 4.1 量化技术

> 🎯 **面试重点**

量化是将模型权重从高精度（FP16/BF16）转换为低精度（INT8/INT4）的技术，核心目标是减少显存占用和加速推理。

**基础概念**：

```
FP16:  ████████████████  (16 bit, 2 bytes)  → 基线
INT8:  ████████          ( 8 bit, 1 byte)   → 显存减半
INT4:  ████              ( 4 bit, 0.5 byte) → 显存 1/4
INT2:  ██                ( 2 bit, 0.25 byte)→ 实验阶段
```

**主流量化方案对比**：

| 方案 | 位宽 | 类型 | 原理 | 质量损失 | 速度 | 典型工具 |
|------|------|------|------|---------|------|---------|
| **GPTQ** | 4/3/2 bit | PTQ (训练后量化) | 逐层最优量化，用 Hessian 信息最小化误差 | 较小 | GPU 推理快 | AutoGPTQ |
| **AWQ** | 4 bit | PTQ | 保护"关键通道"——按激活值重要性加权量化 | 很小 | GPU 推理快 | vLLM 原生支持 |
| **GGUF** | 2-8 bit | PTQ | llama.cpp 格式，支持 CPU 推理 | 取决于位宽 | CPU/GPU 混合 | llama.cpp, Ollama |
| **BitsAndBytes** | 4/8 bit | PTQ | NF4/INT8 量化，QLoRA 的基础 | 小 | 速度一般 | HuggingFace |
| **SmoothQuant** | INT8 | PTQ | 将激活的量化难度"迁移"到权重 | 很小 | INT8 矩阵乘加速 | TensorRT-LLM |

**选择指南**：

```
部署目标是什么？
├── 生产级 GPU 推理 (追求吞吐)
│   └── AWQ 4-bit (vLLM 原生支持, 质量损失最小)
├── 开发/测试 GPU 推理
│   └── GPTQ 4-bit (生态成熟, 工具链丰富)
├── 本地 CPU/消费级 GPU
│   └── GGUF Q4_K_M 或 Q5_K_M (Ollama 直接用)
└── 超低延迟/嵌入式
    └── INT8 SmoothQuant (TensorRT-LLM)
```

**7B 模型不同量化方案的显存占用**：

| 方案 | 权重大小 | 推理显存 (4K ctx) | 质量 (MMLU) |
|------|---------|------------------|-------------|
| FP16 | 14 GB | ~16 GB | 基线 100% |
| INT8 | 7 GB | ~9 GB | ~99.5% |
| GPTQ-4bit | 3.5 GB | ~5.5 GB | ~98% |
| AWQ-4bit | 3.5 GB | ~5.5 GB | ~99% |
| GGUF Q4_K_M | 4.1 GB | ~6 GB | ~98% |
| GGUF Q2_K | 2.7 GB | ~4.5 GB | ~90% |

### 4.2 KV Cache 与 PagedAttention

> 🎯 **面试重点**（vLLM 核心技术）

**KV Cache 基础**：

自回归推理时，每生成一个新 token，都需要和所有历史 token 做注意力计算。如果每次都重新计算所有 KV，效率极低。

**解决方案**：缓存已计算的 K、V 向量，每次只计算新 token 的 Q，与缓存的 KV 做注意力。

```
Step 1: "The"          → 计算 K₁,V₁ → 存入 Cache
Step 2: "The cat"      → 计算 K₂,V₂ → Cache = [K₁V₁, K₂V₂]
Step 3: "The cat sat"  → 计算 K₃,V₃ → Cache = [K₁V₁, K₂V₂, K₃V₃]
...
Step n: 只计算 Qₙ × [K₁...Kₙ] → Attention → [V₁...Vₙ]
```

**问题**：KV Cache 的显存是**动态增长**的——每个请求的序列长度不同、不可预测。传统方案预分配最大长度的连续内存，造成严重浪费（内部碎片 60-80%）。

**PagedAttention（vLLM 核心创新）**：

借鉴操作系统的**虚拟内存分页**思想：

```
传统方案 (连续内存)                     PagedAttention (分页)
┌─────────────────────────┐           ┌────┬────┬────┬────┐
│ Request A (KV Cache)    │           │ A  │ B  │ A  │ C  │  ← Page Table
│ [████████░░░░░░░░░░░░░] │           ├────┼────┼────┼────┤
│  (实际用 40%, 预分配100%)│           │ B  │ C  │ B  │ A  │
├─────────────────────────┤           ├────┼────┼────┼────┤
│ Request B (KV Cache)    │           │ C  │ A  │ free│free│
│ [██████████████░░░░░░░] │           └────┴────┴────┴────┘
│  (实际用 70%, 预分配100%)│           ↑ 物理内存块不需要连续
└─────────────────────────┘           ↑ 按需分配，接近 0 浪费
     内存浪费: ~50%                    内存浪费: < 4%
```

**PagedAttention 的核心优势**：
1. **接近零的内存浪费**：按需分配 page（如 16 tokens/page），用完即释放
2. **支持动态 batch**：不同请求可以随时加入/退出
3. **支持 prefix sharing**：系统 prompt 相同的请求可以共享 KV Cache pages
4. **吞吐量提升 2-4×**：因为同样的显存可以服务更多并发请求

### 4.3 推理框架对比

| 框架 | 核心优势 | 适用场景 | 关键特性 |
|------|---------|---------|---------|
| **vLLM** | PagedAttention, 高吞吐 | 生产级在线推理首选 | Continuous Batching, Prefix Caching, AWQ/GPTQ 支持, OpenAI 兼容 API |
| **TGI** | HuggingFace 生态集成 | HuggingFace 用户, 快速部署 | Flash Attention, 量化支持, Docker 友好 |
| **TensorRT-LLM** | NVIDIA 极致优化 | 追求最低延迟/最高吞吐 | INT8/FP8 内核优化, Inflight Batching, 编译期优化 |
| **SGLang** | RadixAttention, 结构化生成 | Agent/多轮对话/JSON 输出 | 自动 prefix cache, constrained decoding, 并行采样 |
| **llama.cpp** | CPU 推理, 跨平台 | 本地/边缘/嵌入式 | GGUF 格式, Metal/CUDA/Vulkan, 极低依赖 |
| **Ollama** | 一键部署体验 | 开发调试, 个人使用 | 基于 llama.cpp, 模型管理, API 简洁 |

**关键指标对比**（LLaMA-3-8B, A100-80G, 相同负载条件）：

| 指标 | vLLM | TGI | TensorRT-LLM | SGLang |
|------|------|-----|-------------|--------|
| 首 token 延迟 (TTFT) | ~50ms | ~60ms | ~30ms | ~45ms |
| 吞吐 (tokens/s) | ~2500 | ~1800 | ~3000 | ~2700 |
| 最大并发 | 高 | 中 | 高 | 高 |
| 易用性 | ★★★★★ | ★★★★ | ★★★ | ★★★★ |
| 量化支持 | AWQ/GPTQ/FP8 | GPTQ/AWQ | INT8/FP8/INT4 | AWQ/GPTQ |

> **工程建议**：如果只选一个框架 → **vLLM**。它在吞吐、易用性、社区活跃度之间取得了最好的平衡。如果需要极致延迟且有 NVIDIA GPU → TensorRT-LLM。如果做 Agent/结构化输出 → SGLang。

### 4.4 Speculative Decoding（投机解码）

> 🎯 **面试重点**

自回归解码的瓶颈：每生成一个 token 都需要完整的模型前向传播。大模型（如 70B）的单步延迟高，成为延迟瓶颈。

**核心思想**：用小模型（draft model）快速"猜"多个 token，再用大模型（target model）一次性验证。

```
Draft Model (1B, 快):  生成 [t₁, t₂, t₃, t₄, t₅]  ← 5次快速前向传播

Target Model (70B, 慢): 
  输入 [prompt, t₁, t₂, t₃, t₄, t₅]
  一次前向传播，得到每个位置的概率分布
  验证: t₁ ✅  t₂ ✅  t₃ ✅  t₄ ❌  → 接受 3 个 token

结果: 1 次大模型调用 = 3 个 token (而非 1 个)
      加速比 ≈ 2-3×
```

**数学保证**：通过拒绝采样（rejection sampling），投机解码可以保证输出的概率分布与纯大模型解码**完全一致**（lossless）。

**实际效果**：
- 加速比取决于 draft model 的"猜对率"（acceptance rate）
- 代码生成等确定性强的任务：acceptance rate ~80%，加速 2.5-3×
- 开放性创意任务：acceptance rate ~50%，加速 1.5-2×
- Draft model 选择：同系列小模型（如 LLaMA-3-8B → LLaMA-3-70B）或 n-gram 模型

### 4.5 Continuous Batching

传统的 **static batching** 问题：一个 batch 中的请求长度不同，短请求完成后必须等长请求，GPU 利用率低。

**Continuous Batching**（也叫 iteration-level batching）：

```
Static Batching:
Step  1  2  3  4  5  6  7  8
Req A ■  ■  ■  ■  □  □  □  □  ← 完成后空等
Req B ■  ■  ■  ■  ■  ■  ■  ■
Req C ■  ■  □  □  □  □  □  □  ← 完成后空等
                              GPU 利用率 ~50%

Continuous Batching:
Step  1  2  3  4  5  6  7  8
Req A ■  ■  ■  ■
Req B ■  ■  ■  ■  ■  ■  ■  ■
Req C ■  ■
Req D          ■  ■  ■        ← A 完成后加入
Req E             ■  ■  ■  ■  ← C 完成后加入
                              GPU 利用率 ~95%
```

核心原则：**每生成一步**就检查是否有请求完成（可移出）或新请求到来（可加入），而非等整个 batch 完成。

vLLM、TGI、TensorRT-LLM 都支持 Continuous Batching。

### 4.5.1 Prefix Caching（前缀缓存）

在实际生产中，大量请求共享相同的 System Prompt（如 "你是一个有帮助的助手..."），或者多轮对话中前几轮的内容完全相同。Prefix Caching 利用这一特性避免重复计算。

**核心思想**：缓存共享前缀的 KV Cache，新请求只需计算差异部分。

```
请求 A: [System Prompt] + [User: 你好]
请求 B: [System Prompt] + [User: 帮我写代码]
请求 C: [System Prompt] + [User: 翻译这段话]

无 Prefix Caching:
  A: 计算 System Prompt KV + 计算 "你好" KV
  B: 计算 System Prompt KV + 计算 "帮我写代码" KV    ← 重复计算!
  C: 计算 System Prompt KV + 计算 "翻译这段话" KV    ← 重复计算!

有 Prefix Caching:
  A: 计算 System Prompt KV → 存入缓存 + 计算 "你好" KV
  B: 命中缓存 ✅ + 计算 "帮我写代码" KV              ← 节省!
  C: 命中缓存 ✅ + 计算 "翻译这段话" KV              ← 节省!
```

**vLLM 的 Automatic Prefix Caching (APC)**：
- 基于 PagedAttention 的分页机制，自动检测请求间的共享前缀
- 使用 hash 匹配 KV Cache pages，无需手动配置
- 对多轮对话特别有效——每轮新增的内容只需增量计算

**SGLang 的 RadixAttention**：
- 使用 Radix Tree（基数树）数据结构管理所有请求的 KV Cache
- 支持更灵活的前缀共享模式（不仅是头部前缀，还支持分支共享）
- 对 Agent 场景（多次工具调用共享上下文）特别友好

```
Radix Tree 示例:

                    [System Prompt]
                    /              \
           [User: 你好]        [User: 写代码]
           /         \              |
    [Asst: 你好!]  [Asst: 嗨!]  [Asst: 好的...]
    /                               |
[User: 谢谢]                  [User: 用Python]
```

**Prefix Caching 的效果**：

| 场景 | TTFT 加速 | 吞吐提升 | 说明 |
|------|----------|---------|------|
| 共享 System Prompt (500 tokens) | 2-3× | 20-30% | 最常见场景 |
| 多轮对话 (历史 2K tokens) | 3-5× | 30-50% | 每轮只需增量计算 |
| Few-Shot Prompt (1K tokens) | 2-4× | 25-40% | 示例部分可缓存 |
| Agent 多步调用 | 3-6× | 40-60% | 上下文高度重叠 |

> **工程建议**：在生产环境中，**务必开启 Prefix Caching**。vLLM 中只需设置 `--enable-prefix-caching`，几乎零成本获得显著的延迟和吞吐改善。

### 4.5.2 结构化输出与约束解码

在生产环境中，LLM 的输出往往需要符合特定格式（如 JSON、XML、SQL），约束解码（Constrained Decoding）技术确保输出的结构正确性。

**核心原理**：在每一步解码时，根据当前已生成的内容和目标格式的语法规则，屏蔽不合法的 token，只允许合法 token 被采样。

```
目标: 生成合法 JSON

已生成: {"name": "Alice", "age":
合法下一个 token: [0-9, 空格, null, true, false, "]  ← 只允许这些
非法 token: [字母, {, [, 其他]                        ← 被屏蔽
```

**主流实现方案**：

| 方案 | 原理 | 框架支持 | 性能开销 |
|------|------|---------|--------|
| JSON Mode | 模型内置 JSON 输出能力 | OpenAI API, vLLM | 极低 |
| Outlines | 基于正则表达式/CFG 的 token masking | vLLM, SGLang | 低 |
| Guidance | 交错控制流与生成 | 独立库 | 中等 |
| LMQL | 声明式约束语言 | 独立框架 | 中等 |
| SGLang | 原生 constrained decoding + 正则约束 | SGLang | 低 |

**Outlines 的工作原理**：
1. 将目标格式（如 JSON Schema）编译为有限状态自动机（FSA）
2. 预计算每个状态下合法的 token 集合（token mask）
3. 解码时，根据当前 FSA 状态应用 token mask
4. 保证输出 100% 符合目标格式

> **工程建议**：对于需要结构化输出的场景，优先使用 vLLM/SGLang 的原生约束解码支持，比在 Prompt 中要求格式更可靠。

### 4.6 成本优化策略

> **实战经验**：推理成本是持续支出，远超一次性训练成本。

**不同模型规模的部署成本参考**（基于云服务 A100-80G 价格）：

| 模型 | GPU 需求 (FP16) | GPU 需求 (4bit) | 月成本 (FP16) | 月成本 (4bit) |
|------|----------------|----------------|-------------|-------------|
| 7B | 1 × A100 | 1 × T4/L4 | ~¥8K | ~¥2K |
| 13B | 1 × A100 | 1 × A10/L40 | ~¥8K | ~¥4K |
| 70B | 4 × A100 (TP=4) | 2 × A100 | ~¥32K | ~¥16K |
| 405B | 16 × A100 (TP=8, PP=2) | 8 × A100 | ~¥128K | ~¥64K |

**优化策略优先级**：

1. **选择合适的模型大小**：8B 模型 + 好的微调 ≈ 70B 通用模型在特定任务上的效果
2. **量化**：AWQ 4-bit 量化，几乎无损但显存减半
3. **推理框架**：vLLM + Continuous Batching，吞吐提升 2-4×
4. **KV Cache 优化**：Prefix Caching（共享系统 prompt 的 KV Cache）
5. **投机解码**：对延迟敏感的场景，加速 2-3×
6. **请求路由**：简单任务走小模型，复杂任务走大模型（路由器本身可以是一个小分类器）

---

## 五、评估体系

### 5.1 基础能力评估

| Benchmark | 评测能力 | 规模 | 评测方式 | 说明 |
|-----------|---------|------|---------|------|
| **MMLU** | 世界知识、推理 | 57 学科, 14K 题 | 4 选 1 | 最广泛使用的综合 benchmark |
| **HumanEval** | 代码生成 | 164 编程题 | pass@k 执行测试 | OpenAI 提出，Python 函数补全 |
| **MBPP** | 代码生成 | 974 题 | pass@k | Google 提出，基础编程能力 |
| **GSM8K** | 数学推理 | 8.5K 小学数学 | 答案精确匹配 | 测试多步推理，CoT 的经典测试集 |
| **MATH** | 高等数学 | 12.5K 竞赛题 | 答案匹配 | 难度远高于 GSM8K |
| **ARC** | 科学推理 | 7787 题 | 4 选 1 | 小学科学考试 |
| **HellaSwag** | 常识推理 | 10K 题 | 4 选 1 | 句子补全，测试常识 |
| **TruthfulQA** | 真实性 | 817 题 | 多选/生成 | 测试模型是否会编造看似合理的假信息 |

**中文 Benchmark**：
- **C-Eval**: 52 学科, 13K 题, 中文 MMLU 等价物
- **CMMLU**: 67 学科, 中文多学科
- **GAOKAO-Bench**: 高考真题
- **AGIEval**: 法律、公务员等专业考试

### 5.2 对齐评估

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| **MT-Bench** | GPT-4 作为裁判，对多轮对话打分 (1-10) | 自动化、可复现；受 GPT-4 bias 影响 |
| **AlpacaEval** | GPT-4 对比两个模型的回复，计算胜率 | 简单直观；存在长度偏好 |
| **Chatbot Arena** | 真实用户盲评，ELO 排名 | 最接近真实评估；成本高、慢 |
| **WildBench** | 真实用户 prompt + GPT-4 评分 | 覆盖真实场景 |

> **Chatbot Arena** 被公认为最有参考价值的评估方式。它让真实用户在两个匿名模型之间选择更好的回复，通过 ELO 评分系统排名。截至 2025 年，它已成为行业事实标准。

**LLM-as-Judge 的注意事项**：
- GPT-4 作为裁判存在多种 bias：位置偏好（偏向第一个回复）、长度偏好、自身风格偏好
- 缓解方案：交换位置做两次评估取平均；设计结构化评分 rubric；多个模型裁判取共识
- 对于关键评估，仍需要人类评估做 calibration

### 5.3 安全评估

| 维度 | 评测内容 | 方法 |
|------|---------|------|
| **有害内容** | 拒绝生成暴力、歧视、非法内容 | Red Teaming（对抗测试）、自动攻击生成 |
| **幻觉** | 编造不存在的事实 | TruthfulQA、HaluEval、对比检索 |
| **隐私泄露** | 是否输出训练数据中的个人信息 | PII 检测、membership inference |
| **越狱攻击** | 绕过安全限制的鲁棒性 | GCG、AutoDAN、multilingual jailbreak |
| **偏见与公平** | 性别、种族、年龄等偏见 | BBQ、WinoBias、对比测试 |

**Red Teaming 流程**：
1. 构建攻击 prompt 库（覆盖各种越狱模板）
2. 自动攻击生成（用一个 LLM 生成攻击 prompt）
3. 评估模型是否被攻破（拒绝率、有害内容评分）
4. 迭代改进安全对齐

### 5.4 实际业务评估方案设计

> 🎯 **面试重点**（区分"会做 benchmark"和"能在业务中落地"的关键）

学术 benchmark 和实际业务需求往往有很大差距。一个完整的业务评估方案：

**Step 1: 定义评估维度**

```
┌─────────────────────────────────────────────┐
│  业务评估维度                                 │
├──────────────┬──────────────┬───────────────┤
│  功能性      │  质量        │  运营指标      │
├──────────────┼──────────────┼───────────────┤
│ 指令遵循     │ 流畅度       │ 延迟 (P50/P99)│
│ 格式正确     │ 相关性       │ 吞吐          │
│ 完成度       │ 准确性       │ 成本/query    │
│ 安全合规     │ 一致性       │ 可用性        │
└──────────────┴──────────────┴───────────────┘
```

**Step 2: 构建评估集**

- 从真实用户查询中采样（覆盖 head/torso/tail query）
- 包含 edge case 和 adversarial case
- 规模：核心场景至少 200+ 条，边缘场景 50+ 条
- 标注 golden answer（多个标注者 + 仲裁）

**Step 3: 自动化评估 Pipeline**

```python
# 评估 pipeline 伪代码
for query in eval_set:
    response = model.generate(query)
    scores = {
        "format_correct": check_format(response),
        "factual_accuracy": fact_check(response, golden),
        "safety": safety_classifier(response),
        "relevance": llm_judge(query, response, rubric),
        "latency": measure_latency(),
    }
    results.append(scores)
report = aggregate(results)
```

**Step 4: A/B 测试**
- 在线上小流量（如 5%）部署新模型
- 对比关键业务指标（CTR、转化率、用户满意度）
- 至少跑 1 周收集足够样本量
- 统计显著性检验（p < 0.05）后全量

---

## 六、技术选型决策指南

### 6.1 模型规模选择

| 场景 | 推荐规模 | 理由 |
|------|---------|------|
| 客服/FAQ 自动回复 | 7B-8B + SFT | 任务明确，小模型足够 |
| 通用对话/助手 | 13B-14B 或 API | 需要一定的世界知识 |
| 代码生成/复杂推理 | 70B 或 API (GPT-4/Claude) | 需要强推理能力 |
| 多语言/多任务 | 70B+ 或 MoE | 多语言需要更大容量 |

### 6.2 训练 vs API 决策

```
是否有强隐私/合规要求？
├── 是 → 必须自建
│   └── 数据量 > 100K? 
│       ├── 是 → 全参 SFT 开源模型
│       └── 否 → LoRA 微调开源模型
└── 否 → 
    └── 日调用量 > 100K/天?
        ├── 是 → 自建更划算 (7B/13B 微调)
        └── 否 → API 更经济
```

### 6.3 完整技术栈选型建议

| 层次 | 推荐方案 | 备选 |
|------|---------|------|
| 基础模型 | LLaMA-3.x / Qwen-2.5 (开源) | GPT-4o / Claude API (闭源) |
| 微调框架 | LLaMA-Factory (易用) / Axolotl (灵活) | HF Transformers + PEFT |
| 微调方法 | QLoRA (资源有限) / Full SFT + DPO (追求效果) | LoRA |
| 量化 | AWQ (生产) / GGUF (本地) | GPTQ |
| 推理框架 | vLLM (在线) / Ollama (本地) | TensorRT-LLM / SGLang |
| 评估 | LLM-as-Judge + 业务评估集 | 开源 Benchmark + Chatbot Arena |

---

## 附录：面试高频考点速查

### 🔴 高频考点（几乎必问）

1. **Transformer 自注意力机制**：Q/K/V 的计算过程、复杂度分析
2. **为什么 Decoder-Only 胜出**：统一范式、scaling 效率、In-Context Learning、KV Cache 友好
3. **RLHF vs DPO**：流程对比、数学原理、优缺点
4. **LoRA 原理**：低秩假设、参数计算、超参选择
5. **KV Cache**：为什么需要、显存计算、PagedAttention
6. **量化**：INT8/INT4 的基本原理、GPTQ/AWQ/GGUF 区别

### 🟡 中频考点（常问）

7. **GQA 原理**：MHA → MQA → GQA 的演进
8. **RoPE**：旋转位置编码的直觉理解
9. **Scaling Laws**：Chinchilla 定律及其影响
10. **分布式训练**：DP/TP/PP 的区别和适用场景
11. **推理优化**：Continuous Batching、Speculative Decoding
12. **MoE 架构**：稀疏激活、负载均衡

### 🟢 低频但加分的考点

13. **Flash Attention**：IO 感知计算的原理
14. **长上下文扩展**：NTK 插值、YaRN、Ring Attention
15. **预训练数据工程**：去重、质量过滤 pipeline
16. **业务评估方案设计**：从 benchmark 到 A/B 测试
17. **SwiGLU / RMSNorm**：具体改进点
18. **GRPO**：DeepSeek-R1 的训练方法，组内相对排序
19. **Adapter vs LoRA vs Prefix Tuning**：各种 PEFT 方法的对比
20. **Prefix Caching / RadixAttention**：推理优化的缓存策略
21. **约束解码**：结构化输出的保证机制

---

## 附录 B：LLM 技术发展时间线

```
2017  Transformer (Attention Is All You Need)
  │
2018  BERT (双向编码器) / GPT-1 (单向解码器)
  │
2019  GPT-2 (1.5B, 零样本能力) / T5 (统一文本到文本)
  │
2020  GPT-3 (175B, In-Context Learning 涌现) / Scaling Laws (Kaplan)
  │
2021  Codex (代码生成) / InstructGPT (RLHF 首次大规模应用)
  │
2022  ChatGPT (对话式 AI 爆发) / Chinchilla (修正 Scaling Laws)
  │    PaLM (540B) / BLOOM (开源 176B) / LLaMA-1 (开源里程碑)
  │
2023  GPT-4 (多模态) / LLaMA-2 (开源商用) / Mistral-7B (小模型标杆)
  │    DPO 论文 / QLoRA / Flash Attention 2 / vLLM
  │
2024  LLaMA-3 (8B/70B/405B) / Qwen-2.5 / DeepSeek-V3 (MoE 671B)
  │    Claude 3.5 Sonnet / Gemini 1.5 Pro (1M 上下文)
  │    GRPO / DeepSeek-R1 (推理能力突破)
  │
2025  开源模型逼近闭源 / 推理模型 (o1/R1) 成为新范式
  │    Agent 框架成熟 / MoE 成为标配 / 长上下文普及
  └──▶ ...
```

---

## 附录 C：常用工具与资源速查

### 训练与微调

| 工具 | 用途 | 特点 |
|------|------|------|
| **LLaMA-Factory** | 一站式微调框架 | 支持 100+ 模型，Web UI，LoRA/QLoRA/全参/RLHF/DPO |
| **Axolotl** | 灵活的微调框架 | 配置驱动，支持复杂训练流程 |
| **Unsloth** | 高速微调 | 2× 训练加速，70% 显存节省 |
| **DeepSpeed** | 分布式训练 | ZeRO 优化器，大规模训练必备 |
| **Megatron-LM** | 大规模预训练 | NVIDIA 官方，TP/PP/DP 全支持 |
| **PEFT** | 参数高效微调库 | HuggingFace 官方，LoRA/Adapter/Prefix Tuning |
| **TRL** | 对齐训练库 | HuggingFace 官方，SFT/DPO/PPO/GRPO |

### 推理与部署

| 工具 | 用途 | 特点 |
|------|------|------|
| **vLLM** | 生产级推理 | PagedAttention，高吞吐，OpenAI 兼容 API |
| **SGLang** | 结构化推理 | RadixAttention，约束解码，Agent 友好 |
| **TensorRT-LLM** | 极致性能推理 | NVIDIA 优化，FP8/INT8 内核 |
| **llama.cpp** | 本地/边缘推理 | CPU 推理，GGUF 格式，跨平台 |
| **Ollama** | 一键本地部署 | 基于 llama.cpp，模型管理简洁 |
| **AutoGPTQ** | GPTQ 量化 | 训练后量化工具 |
| **AutoAWQ** | AWQ 量化 | 激活感知量化，vLLM 原生支持 |

### 评估与测试

| 工具 | 用途 | 特点 |
|------|------|------|
| **lm-evaluation-harness** | 标准 Benchmark 评估 | EleutherAI 出品，支持 MMLU/HumanEval 等 |
| **OpenCompass** | 综合评估平台 | 上海 AI Lab，中英文 Benchmark 全覆盖 |
| **MT-Bench** | 对话质量评估 | GPT-4 作为裁判，多轮对话 |
| **Chatbot Arena** | 人类偏好评估 | ELO 排名，行业金标准 |
| **RAGAS** | RAG 系统评估 | Faithfulness/Relevancy 等维度 |

### 数据工程

| 工具 | 用途 | 特点 |
|------|------|------|
| **Hugging Face Datasets** | 数据集管理 | 海量开源数据集，流式加载 |
| **Label Studio** | 数据标注 | 开源标注平台，支持多种任务类型 |
| **Argilla** | AI 反馈数据管理 | 偏好标注、RLHF 数据管理 |
| **datatrove** | 大规模数据处理 | HuggingFace 出品，去重/过滤/清洗 |

---


---

## 七、Transformer 核心组件深度剖析

> 💡 **面试高频区**：注意力机制的数学推导、位置编码对比、归一化选择是面试中最常被追问的细节。

### 7.1 Self-Attention 完整数学推导 【高频】

**标准 Scaled Dot-Product Attention：**

```
输入：X ∈ R^{n×d}  (n 个 token，d 维)

步骤 1: 线性投影
    Q = X · W_Q    (W_Q ∈ R^{d×d_k})
    K = X · W_K    (W_K ∈ R^{d×d_k})
    V = X · W_V    (W_V ∈ R^{d×d_v})

步骤 2: 计算注意力分数
    Score = Q · K^T / √d_k

    为什么要除以 √d_k？
    ─────────────────────
    假设 Q 和 K 的各分量独立同分布，均值 0，方差 1
    则 Q·K^T 的每个元素是 d_k 个随机变量之和
    方差 = d_k → 标准差 = √d_k
    除以 √d_k 使方差归一化到 1
    → 防止 Softmax 输入过大导致梯度消失

步骤 3: Softmax 归一化
    Attention_weights = Softmax(Score)  ← 每行之和 = 1

步骤 4: 加权求和
    Output = Attention_weights · V
```

**Multi-Head Attention 的本质：**

```
┌──────────────────────────────────────────────────┐
│                Multi-Head Attention               │
│                                                    │
│  X ──┬── Head_1 (Q₁K₁V₁) ──┐                    │
│      ├── Head_2 (Q₂K₂V₂) ──┤                    │
│      ├── Head_3 (Q₃K₃V₃) ──┼── Concat ── W_O ──│── Output
│      ├── ...                 │                    │
│      └── Head_h (QₕKₕVₕ) ──┘                    │
│                                                    │
│  每个 Head: d_k = d_v = d_model / h              │
│  参数量: 4 × d² (Q,K,V 各 d², 输出投影 d²)      │
└──────────────────────────────────────────────────┘

为什么多头而非单头？
────────────────────
1. 不同的头关注不同的语义关系
   - Head 1: 语法依赖（主谓一致）
   - Head 2: 共指消解
   - Head 3: 位置关系
2. 低秩近似的集成效果
   - 单头 d_k = d → 全秩但单一视角
   - 多头 d_k = d/h → 各头低秩，组合后表达力更强
```

**计算复杂度分析：**

| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| QKV 投影 | O(n·d²) | O(n·d) | 线性变换 |
| Attention Score | O(n²·d) | O(n²) | 核心瓶颈 |
| Softmax | O(n²) | O(n²) | 逐行归一化 |
| Attention × V | O(n²·d) | O(n·d) | 加权求和 |
| **总计** | **O(n²·d)** | **O(n²)** | n 是序列长度 |

### 7.2 GQA/MQA 注意力变体深度对比 【高频】

```
标准 MHA (Multi-Head Attention):
──────────────────────────────────
Head 1:  Q₁  K₁  V₁     ← 每个 Head 有独立的 K,V
Head 2:  Q₂  K₂  V₂
Head 3:  Q₃  K₃  V₃
Head 4:  Q₄  K₄  V₄
参数量: 4h·d_k·d = 4d²
KV Cache: 2·h·d_k·n 每层

MQA (Multi-Query Attention):
──────────────────────────────────
Head 1:  Q₁ ─┐
Head 2:  Q₂ ─┤─ K_shared  V_shared  ← 所有 Head 共享同一 K,V
Head 3:  Q₃ ─┤
Head 4:  Q₄ ─┘
参数量: (h+2)·d_k·d ≈ d²+2d·d_k  (远小于 4d²)
KV Cache: 2·d_k·n 每层  (缩小 h 倍!)

GQA (Grouped Query Attention):  ← LLaMA-2/3, Qwen 使用
──────────────────────────────────
Group 1: Q₁,Q₂ ─── K₁ V₁   ← 每组内共享 K,V
Group 2: Q₃,Q₄ ─── K₂ V₂
参数量: 介于 MHA 和 MQA 之间
KV Cache: 2·g·d_k·n 每层  (g = 组数)
```

**实际模型中的选择：**

| 模型 | 注意力类型 | Head 数 | KV Head 数 | 原因 |
|------|-----------|---------|-----------|------|
| GPT-4 (推测) | MHA | 128 | 128 | 最大质量，不差资源 |
| LLaMA-2 70B | GQA | 64 | 8 | 平衡质量与推理效率 |
| LLaMA-3 8B | GQA | 32 | 8 | 小模型也用 GQA |
| Qwen-2.5 72B | GQA | 64 | 8 | 同 LLaMA 策略 |
| Falcon-180B | MQA | 232 | 1 | 极致推理速度 |
| Mistral-7B | GQA | 32 | 8 | 小模型效率优先 |

### 7.3 位置编码技术全面对比 【高频】

```
位置编码演进路线：
────────────────────────────────────────────────────

绝对位置编码                    相对位置编码
    │                              │
    ├── 正弦/余弦 (Vaswani)         ├── RPE (Shaw et al.)
    │   PE(pos,2i) = sin(pos/10000^(2i/d))  │
    │   PE(pos,2i+1) = cos(...)     ├── ALiBi (Press et al.)
    │                              │   Score -= m·|i-j|
    ├── 可学习位置 (BERT/GPT-2)     │   线性偏置，无需训练
    │   E_pos ∈ R^{max_len × d}    │
    │   问题: 无法外推              ├── RoPE (Su et al.) ← 主流
    │                              │   旋转位置编码
    └── 加在输入层                  └── 作用在注意力层
```

**RoPE (Rotary Position Embedding) 深度解析：** 【面试重点】

```
核心思想: 用旋转矩阵编码位置信息

给定 token 位置 m，对 query/key 的第 i 维分量对 (q_{2i}, q_{2i+1})：

    ┌ q'_{2i}   ┐   ┌ cos(m·θ_i)  -sin(m·θ_i) ┐ ┌ q_{2i}   ┐
    │            │ = │                            │·│           │
    └ q'_{2i+1} ┘   └ sin(m·θ_i)   cos(m·θ_i)  ┘ └ q_{2i+1} ┘

其中 θ_i = 10000^{-2i/d}

关键性质: q'_m · k'_n = f(q, k, m-n)
→ 内积只依赖相对位置 m-n，而不是绝对位置
→ 自带远程衰减特性（高频维度衰减快）

长度外推问题:
─────────────
预训练 4K 上下文 → 推理时输入 32K
高频分量的 θ 值超出训练分布 → 注意力崩坏

解决方案:
1. NTK-Aware Scaling: 修改 base (10000 → 更大值)
   → θ_i = base'^{-2i/d}，拉伸频率
2. YaRN: NTK + 注意力温度补偿 + 动态缩放
3. 直接长序列继续训练: 用 32K/128K 数据 fine-tune
```

**位置编码对比总结：**

| 编码方式 | 外推性 | 计算开销 | 代表模型 | 优势 | 劣势 |
|---------|--------|---------|---------|------|------|
| 正弦/余弦 | 差 | 零 | 原始 Transformer | 简单 | 无法学习 |
| 可学习 | 无 | 少量参数 | BERT, GPT-2 | 灵活 | 长度固定 |
| ALiBi | 好 | 零 | BLOOM, MPT | 外推好 | 精度略低 |
| RoPE | 中→好(+YaRN) | 极低 | LLaMA, Qwen, Mistral | 主流选择 | 需长度适配 |

### 7.4 激活函数与归一化选择 【中频】

```
激活函数演进:
─────────────
ReLU → GELU → SwiGLU

ReLU:   f(x) = max(0, x)
        ↓ 问题: 负区间梯度为零 ("Dead ReLU")

GELU:   f(x) = x · Φ(x)  (Φ 是高斯 CDF 的近似)
        ↓ GPT-2/BERT 使用，平滑版 ReLU

SwiGLU: f(x) = Swish(x·W₁) ⊙ (x·W₂)  (⊙ 逐元素乘)
        ↓ Swish(x) = x · σ(x)
        ↓ 门控机制，更强的表达力
        ↓ LLaMA/Qwen/Mistral 标配
        ↓ 注意: FFN 需要 3 个权重矩阵而非 2 个
              → 通常将 FFN 中间维度从 4d 调整为 (8/3)d
              → 保持总参数量不变
```

**归一化方式对比：**

| 特性 | LayerNorm | RMSNorm | 说明 |
|------|-----------|---------|------|
| 公式 | (x - μ) / σ · γ + β | x / RMS(x) · γ | RMS = √(mean(x²)) |
| 参数 | γ, β (2d) | γ (d) | RMSNorm 少一半参数 |
| 计算量 | 需要 mean 和 var | 只需 mean(x²) | RMSNorm 省约 10-15% |
| 位置 | Post-Norm (原始) | Pre-Norm (现代) | Pre-Norm 训练更稳定 |
| 使用 | BERT, GPT-2 | LLaMA, Qwen, Mistral | 现代 LLM 标配 |

```python
# RMSNorm 实现
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

---

## 八、MoE 架构深度解析

> 💡 **面试加分区**：MoE 是 2024-2025 年最热的架构话题，DeepSeek-V3 将其推向工业化。

### 8.1 MoE 工作原理

```
标准 Transformer Block:
    Input → Attention → FFN → Output
    
MoE Transformer Block:
    Input → Attention → Router → [Expert_1, Expert_2, ..., Expert_N] → Output
                         │
                         └── 选择 Top-K 个专家 (通常 K=2)

详细流程:
─────────
                    ┌─────────────┐
     token ───────▶│   Router    │──── g₁,g₂,...,gₙ (门控权重)
     (hidden)      │  (线性层)   │
                    └─────┬───────┘
                          │ 选 Top-K
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │Expert 1 │ │Expert 2 │ │Expert N │
         │  (FFN)  │ │  (FFN)  │ │  (FFN)  │
         └────┬────┘ └────┬────┘ └─────────┘
              │           │
              │  g_i·E_i  │  g_j·E_j       ← 加权求和
              └─────┬─────┘
                    ▼
                  Output = Σ g_k · Expert_k(x)

参数量 vs 计算量:
─────────────────
DeepSeek-V3: 671B 总参数, 37B 激活参数
→ 每个 token 只用 37B 参数计算
→ 相当于用 37B 模型的成本获得 671B 的知识容量
```

### 8.2 Router 设计与负载均衡 【面试重点】

```
Router 计算:
    g = Softmax(x · W_gate)  (W_gate ∈ R^{d × N_experts})
    选 Top-K: 保留 g 中最大的 K 个，其余置零

负载均衡问题:
────────────
如果不加约束 → 某些专家被过度选择 → "专家坍塌"
→ 模型退化为密集模型（只用少数专家）

解决方案:
1. Auxiliary Loss (辅助损失):
   L_balance = α · N · Σᵢ (fᵢ · Pᵢ)
   fᵢ = 分配给专家 i 的 token 比例
   Pᵢ = 专家 i 的平均门控概率
   → 鼓励均匀分配

2. Expert Capacity (容量限制):
   每个专家处理的 token 数 ≤ (总 tokens / N) × capacity_factor
   → 超出的 token 被丢弃或溢出到下一层

3. 噪声注入 (Noisy Top-K):
   g = Softmax(x·W_gate + ε)
   ε ~ N(0, Softplus(x·W_noise))
   → 增加探索性

4. Expert Choice Routing (GShard 改进):
   反转选择方向——让专家选 token 而非 token 选专家
   → 天然保证负载均衡
```

### 8.3 DeepSeek MoE 创新点

```
DeepSeek-V3 关键创新:
─────────────────────

1. 细粒度专家 + 共享专家
   ┌─────────────────────────────────────────┐
   │ 传统 MoE: 8 个大专家，选 2                │
   │ DeepSeek: 256 个小专家，选 8 + 1 共享专家  │
   │                                           │
   │ 共享专家: 每个 token 都会经过              │
   │ → 学习通用知识                            │
   │ 路由专家: 按需激活                         │
   │ → 学习专门知识                            │
   └─────────────────────────────────────────┘

2. 无辅助损失的负载均衡 (Auxiliary-Loss-Free)
   引入 bias 项: g'_i = g_i + b_i
   b_i 根据专家使用频率动态调整
   → 过载专家 b_i 减小，空闲专家 b_i 增大
   → 比辅助损失更优雅，不影响模型质量

3. Multi-Token Prediction (MTP)
   不只预测下一个 token，同时预测后续多个
   → 更丰富的训练信号
   → 推理时可用于 Speculative Decoding
```

### 8.4 MoE 训练与推理挑战

| 挑战 | 原因 | 解决方案 |
|------|------|---------|
| 通信瓶颈 | 专家分布在不同 GPU，All-to-All 通信 | Expert Parallelism + 通信优化 |
| 显存占用 | 所有专家参数都需要加载 | 量化 + Offloading |
| 负载不均 | 训练/推理时专家使用不平衡 | 容量因子 + 动态路由 |
| 训练不稳定 | Router 梯度噪声大 | 渐进式训练 + 梯度裁剪 |
| 推理延迟 | 单 token 需要路由 + 跨设备通信 | Expert Parallelism + 缓存热点专家 |

```python
# 简化的 MoE Layer 实现
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # 路由器
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        
        # 专家 (每个都是标准 FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(n_experts)
        ])
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (B*S, d_model)
        
        # 路由
        router_logits = self.gate(x_flat)  # (B*S, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K 选择
        topk_weights, topk_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 专家计算
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # (B*S,)
            weight = topk_weights[:, k].unsqueeze(-1)  # (B*S, 1)
            
            for i in range(self.n_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[i](expert_input)
                    output[mask] += weight[mask] * expert_output
        
        return output.view(batch_size, seq_len, d_model)
```

---

## 九、长上下文技术全景

### 9.1 上下文窗口扩展技术路线

```
上下文长度演进:
────────────────────────────────────────────────
GPT-3 (2020)      2K tokens
GPT-3.5 (2023)    4K → 16K
GPT-4 (2023)      8K → 32K → 128K
Claude 3 (2024)   200K
Gemini 1.5 (2024) 1M → 2M
LLaMA-3 (2024)    8K → 128K

关键问题: 注意力 O(n²) → 128K 时计算量爆炸
```

### 9.2 长上下文关键技术

**Flash Attention：** 【高频】

```
传统 Attention 的显存问题:
──────────────────────────
S = Q·K^T  ← (n×n) 矩阵，128K 时需要 128K² × 2B = 32GB!
P = Softmax(S)
O = P·V

Flash Attention 核心思想: 分块计算，不存中间矩阵
──────────────────────────────────────────────────
                HBM (显存，慢)          SRAM (片上内存，快)
                ┌──────────┐            ┌──────────┐
  传统:         │ Q,K,V    │ ←读写→    │ 计算     │
                │ S (n×n)  │   5次      │          │
                │ P (n×n)  │            │          │
                │ O        │            │          │
                └──────────┘            └──────────┘

  Flash:        ┌──────────┐            ┌──────────┐
                │ Q,K,V    │ ←读写→    │ Q_block  │
                │ O        │   2次      │ K_block  │
                │          │            │ V_block  │
                │          │            │ O_block  │
                └──────────┘            └──────────┘
                                        在 SRAM 中完成
                                        Softmax 的分块计算
                                        (Online Softmax 算法)

效果:
- 显存: O(n²) → O(n)
- 速度: 2-4× 加速
- 精度: 数学等价，无精度损失
- Flash Attention 2: 进一步优化并行度，再快 2×
- Flash Attention 3: 利用 H100 硬件特性
```

**Ring Attention：**

```
分布式长上下文方案:
───────────────────
将 Q 分块到不同 GPU，K/V 在 GPU 之间"环形传递"

  GPU 0      GPU 1      GPU 2      GPU 3
  Q_0        Q_1        Q_2        Q_3
  ┌───┐      ┌───┐      ┌───┐      ┌───┐
  │K_0│─────▶│K_0│─────▶│K_0│─────▶│K_0│──┐
  │V_0│      │V_0│      │V_0│      │V_0│  │
  └───┘      └───┘      └───┘      └───┘  │
   ▲                                       │
   └───────────────────────────────────────┘
   
   Round 1: 每个 GPU 用本地 KV 计算
   Round 2: KV 传给下一个 GPU
   Round 3: 继续...
   Round N: 所有 KV 块都被看过

优势: 无显存瓶颈，上下文长度 = GPU 数 × 单卡容量
```



---

## 十、对齐技术深度解析

> 💡 **面试高频区**：RLHF → DPO → GRPO 的演进是 2024-2025 面试必考话题。

### 10.1 RLHF 完整流程 【高频】

```
RLHF 三阶段流程:
═══════════════════════════════════════════════════════════

阶段 1: SFT (Supervised Fine-Tuning)
────────────────────────────────────
目标: 让预训练模型学会"按指令回答"

  数据: [(指令, 高质量回答)] × 10K-100K 条
  损失: 标准交叉熵 L = -Σ log P(y_t | y_{<t}, x)
  
  关键: 数据质量 >> 数量
  - InstructGPT: 13K 条人工标注
  - Alpaca: 52K 条 GPT-4 生成（Self-Instruct）
  
  输出: π_SFT (SFT 模型)

阶段 2: 奖励模型训练 (Reward Model)
────────────────────────────────────
目标: 训练一个打分器，评判回答好坏

  数据: [(x, y_win, y_lose)] 偏好对
  - 对同一问题 x，由 π_SFT 生成 K 个回答
  - 人类标注员比较排序: y_1 > y_2 > y_3
  - 生成 C(K,2) 个偏好对
  
  模型: 去掉 LM Head，加一个标量输出头
  损失: Bradley-Terry 模型
  L_RM = -E[log σ(r(x,y_w) - r(x,y_l))]
  
  输出: r(x,y) 奖励模型

阶段 3: PPO 优化
────────────────────────────────────
目标: 用奖励模型信号优化策略

  优化目标:
  max_π E[r(x,y)] - β·KL(π || π_SFT)
                      │
                      └── 防止策略漂移太远
                          ("Reward Hacking")
  
  PPO 具体流程:
  ┌─────────────────────────────────────────┐
  │ for each batch of prompts:              │
  │   1. π_old 生成回答 y                   │
  │   2. 计算 r(x,y) 奖励                  │
  │   3. 计算 Advantage: A = r - V(s)       │
  │   4. PPO Clip 更新:                     │
  │      ratio = π(y|x) / π_old(y|x)       │
  │      L = min(ratio·A, clip(ratio)·A)    │
  │   5. 更新 π 和 V                        │
  └─────────────────────────────────────────┘
  
  需要的模型:
  - Actor (被优化的策略)     ← 训练
  - Critic (价值函数)       ← 训练
  - Reference (π_SFT 副本)  ← 冻结
  - Reward Model            ← 冻结
  → 4 个模型同时在 GPU 上！显存需求巨大
```

### 10.2 DPO：无需奖励模型的对齐 【高频】

```
DPO 核心洞察:
═══════════════
RLHF 的最优解可以解析得到:

  π*(y|x) = π_ref(y|x) · exp(r(x,y) / β) / Z(x)

反解奖励函数:
  r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β·log Z(x)

代入 Bradley-Terry 模型:
  P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))
                    = σ(β·log(π(y_w|x)/π_ref(y_w|x)) 
                        - β·log(π(y_l|x)/π_ref(y_l|x)))

→ 直接用偏好数据训练策略模型，跳过奖励模型!

DPO 损失函数:
  L_DPO = -E[log σ(β·(log π(y_w|x)/π_ref(y_w|x) 
                        - log π(y_l|x)/π_ref(y_l|x)))]
```

**DPO vs RLHF 对比：**

| 维度 | RLHF (PPO) | DPO | 
|------|------------|-----|
| 需要的模型 | 4个（Actor/Critic/Ref/RM） | 2个（Policy/Ref） |
| 显存需求 | 极高（70B 需 4×A100/H100） | 中等（70B 需 2×A100/H100） |
| 训练稳定性 | 低（PPO 超参敏感） | 高（标准分类损失） |
| 超参数 | 多（PPO 有 10+ 超参） | 少（主要就是 β） |
| 效果上限 | 更高（在线采样） | 略低（离线数据） |
| 实现复杂度 | 高 | 低 |
| 代表工作 | InstructGPT, Claude | Zephyr, Tulu |
| 数据需求 | 在线生成 | 离线偏好对 |

### 10.3 GRPO：DeepSeek-R1 的对齐方法 【面试热点】

```
GRPO (Group Relative Policy Optimization):
═══════════════════════════════════════════

动机: DPO 用离线数据，PPO 太复杂
GRPO: 在线采样 + 组内相对排序 → 简洁高效

核心流程:
┌─────────────────────────────────────────────────────┐
│ for each prompt x:                                   │
│                                                       │
│   1. 从当前策略采样 G 个回答:                         │
│      {y₁, y₂, ..., y_G} ~ π_θ(·|x)                 │
│                                                       │
│   2. 用奖励模型/规则打分:                             │
│      {r₁, r₂, ..., r_G}                              │
│                                                       │
│   3. 组内标准化 (Group Relative):                     │
│      Â_i = (r_i - mean(r)) / std(r)                  │
│      → 不需要 Critic 网络!                            │
│      → 用组内统计量代替 Value Function                │
│                                                       │
│   4. 策略梯度更新 (带 clip):                          │
│      ratio = π_θ(y_i|x) / π_old(y_i|x)              │
│      L = -1/G Σ min(ratio·Â_i, clip(ratio)·Â_i)     │
│          + β·KL(π_θ || π_ref)                         │
└─────────────────────────────────────────────────────┘

关键创新:
────────
1. 去掉 Critic: 用组内相对分数代替 Advantage 估计
   → 省一个模型的显存
   → 避免 Critic 估计偏差

2. 在线采样: 从当前策略采样
   → 比 DPO 的离线数据更新鲜
   → 持续改进

3. 兼容规则奖励:
   → DeepSeek-R1 用正确性验证（数学/代码）作为奖励
   → 不一定需要训练奖励模型
```

**对齐技术演进总结：**

```
          简单度增加 →
RLHF ──────────── DPO ──────────── GRPO
(PPO)              │                 │
4个模型            2个模型           2个模型+采样
超参敏感           β 一个参数        PPO 简化版
离线 Critic        离线偏好          在线采样
效果天花板高       效果中等偏上      效果好+简洁
2022              2023              2024-2025
InstructGPT       Zephyr            DeepSeek-R1
```

---

## 十一、量化技术完全指南

> 💡 **面试中频区**：理解量化原理比记忆具体数值更重要。

### 11.1 量化基础

```
为什么需要量化？
───────────────
70B 模型 (FP16):
  参数: 70B × 2B/param = 140GB 显存
  → 至少 2 张 A100-80GB

70B 模型 (INT4):
  参数: 70B × 0.5B/param = 35GB
  → 1 张 A100-80GB 就够了

量化就是: FP16 (16 bit) → INT8 (8 bit) / INT4 (4 bit) / FP8 (8 bit)
```

### 11.2 主流量化方法对比 【中频】

```
量化方法分类:
────────────────────────────
PTQ (Post-Training Quantization):
  训练完再量化，不需要训练数据
  ├── GPTQ  (逐层最优量化，需要校准数据)
  ├── AWQ   (激活感知，保护重要权重)
  ├── GGUF  (llama.cpp 格式，CPU 友好)
  └── SmoothQuant (激活-权重联合平滑)

QAT (Quantization-Aware Training):
  训练时模拟量化效果
  └── QLoRA (量化基础模型 + LoRA 微调)
```

**各方法深度对比：**

| 方法 | 精度(4bit) | 速度 | 校准数据 | 原理 | 适用推理框架 |
|------|-----------|------|---------|------|-------------|
| GPTQ | ★★★★ | 快 | 128条 | 逐层 Hessian 优化 | vLLM, TGI |
| AWQ | ★★★★★ | 快 | 128条 | 保护 1% 显著权重 | vLLM (原生) |
| GGUF | ★★★ | 中 | 不需要 | 分块量化 + 超级块 | llama.cpp, Ollama |
| SmoothQuant | ★★★★ | 最快 | 少量 | 平滑激活异常值 | TensorRT-LLM |
| bitsandbytes | ★★★ | 中 | 不需要 | NF4 + 双重量化 | HuggingFace |

### 11.3 AWQ 原理详解

```
AWQ (Activation-Aware Weight Quantization):
═══════════════════════════════════════════

核心观察: 
  不是所有权重同等重要！
  1% 的"显著通道"对应的激活值特别大
  → 这些通道的量化误差影响最大

策略:
  Step 1: 用校准数据找到"显著通道"
          s_i = max(|Activation[:,i]|)  (每列的最大激活)
  
  Step 2: 对显著通道放大后再量化
          W'[:,i] = W[:,i] × s_i^α   (α ≈ 0.5)
          X'[i,:] = X[i,:] / s_i^α
          → 等价变换: W'×X' = W×X
          → 但量化 W' 时，显著通道被放大，相对误差更小

  Step 3: 搜索最优 α
          对每组权重搜索最优缩放因子
          min_α ||Q(W·diag(s^α))·diag(s^{-α})·X - W·X||
          
结果:
  → INT4 量化几乎无损 (MMLU 下降 < 0.5%)
  → vLLM 原生支持 AWQ 格式
```

### 11.4 QLoRA 微调 【高频】

```
QLoRA 核心思想:
═══════════════
用 4-bit 量化的模型做底座 + LoRA 适配器用 BF16

  ┌─────────────────────────────────────────┐
  │ 原始模型 W (FP16)                       │
  │     ↓ 量化                              │
  │ 量化模型 Q(W) (NF4, 4-bit) ← 冻结      │
  │     +                                    │
  │ LoRA: ΔW = B·A (BF16) ← 可训练         │
  │                                          │
  │ 前向: y = Q(W)·x + B·A·x               │
  │ 反向: 只更新 B,A 的梯度                  │
  │                                          │
  │ 显存节省:                                │
  │   70B FP16 微调: 4-8 × A100-80GB        │
  │   70B QLoRA:    1 × A100-80GB !!!       │
  └─────────────────────────────────────────┘

QLoRA 三大创新:
1. NF4 (NormalFloat 4-bit): 
   假设权重服从正态分布 → 量化桶按正态分位数划分
   → 比均匀 INT4 更精确

2. 双重量化 (Double Quantization):
   量化参数本身也量化
   FP32 scale factor → FP8 scale factor
   → 额外省 0.5 bit/param

3. 分页优化器 (Paged Optimizer):
   优化器状态放 CPU，GPU 显存不够时自动换页
```

```python
# QLoRA 微调实践代码
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch

# Step 1: 4-bit 量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算用 BF16
    bnb_4bit_use_double_quant=True,       # 双重量化
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# Step 2: 添加 LoRA
lora_config = LoraConfig(
    r=64,                    # LoRA rank
    lora_alpha=16,           # 缩放因子
    target_modules=[         # 应用 LoRA 的层
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 83,886,080 || all params: 8,030,261,248
# trainable%: 1.04%

# Step 3: 训练 (用 Trainer 或 TRL)
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=4096,
    # ... 其他训练参数
)
trainer.train()
```

---

## 十二、推理优化进阶

### 12.1 KV Cache 深度解析 【高频】

```
为什么需要 KV Cache？
═══════════════════════
自回归生成: 每生成一个 token 需要看所有之前的 token

无 Cache:
  Step 1: 输入 [A]          → 计算 A 的 K,V → 输出 B
  Step 2: 输入 [A,B]        → 重算 A 的 K,V + 计算 B 的 → 输出 C
  Step 3: 输入 [A,B,C]      → 重算 A,B 的 K,V + 计算 C 的 → 输出 D
  → O(n²) 的总计算量!

有 Cache:
  Step 1: 输入 [A]          → 缓存 K_A,V_A → 输出 B
  Step 2: 只输入 [B]         → 缓存 K_B,V_B → 输出 C  (从 cache 读 K_A,V_A)
  Step 3: 只输入 [C]         → 缓存 K_C,V_C → 输出 D  (从 cache 读 K_A,V_A,K_B,V_B)
  → O(n) 的总计算量!

KV Cache 显存计算:
  每层每 token: 2 × d_model × sizeof(dtype)  (K 和 V 各一份)
  总计: 2 × n_layers × d_model × seq_len × sizeof(dtype) × batch_size
  
  LLaMA-3 70B (FP16, batch=1, seq=4096):
  = 2 × 80 × 8192 × 4096 × 2B = 10.7GB
  → 70B 模型光 KV Cache 就要 10GB+!
```

### 12.2 PagedAttention (vLLM) 【高频】

```
传统 KV Cache 问题:
────────────────────
每个请求预分配 max_seq_len 的连续显存
→ 短请求浪费大量显存 (内部碎片)
→ 显存利用率低 → 并发量上不去

PagedAttention 思想: 像 OS 虚拟内存一样管理 KV Cache
────────────────────────────────────────────────────
  物理显存被划分为固定大小的"Page" (如 16 tokens)
  每个请求有一个 Page Table 映射逻辑块 → 物理块
  
  Request 1: [Block 0] → [Block 1] → [Block 2]
              Page 5      Page 12     Page 3     (物理不连续)
  
  Request 2: [Block 0] → [Block 1]
              Page 8      Page 1
  
  优势:
  1. 按需分配: 短请求只用少量 Page
  2. 无碎片: Page 粒度分配
  3. 共享: 相同 Prompt 的 KV Cache 可共享 Page
     → Prefix Caching: 系统提示只存一份
  4. 显存利用率: ~98% vs 传统 ~50%
  → 吞吐量提升 2-4×
```

### 12.3 Speculative Decoding 【中频】

```
核心思想: 用小模型"猜"，大模型"验"
═══════════════════════════════════

传统生成 (大模型):
  Step 1: [大模型] → token_1     (100ms)
  Step 2: [大模型] → token_2     (100ms)
  Step 3: [大模型] → token_3     (100ms)
  总计: 300ms / 3 tokens

投机解码:
  Step 1: [小模型] → guess_1, guess_2, guess_3   (30ms, 并行猜3个)
  Step 2: [大模型] 验证 guess_1,2,3               (100ms, 一次验证)
          → 假设 guess_1,2 正确, guess_3 错误
          → 大模型给出正确的 token_3
  总计: 130ms / 3 tokens  (2.3× 加速!)

数学保证:
  修正采样确保输出分布与纯大模型完全一致
  → 无质量损失！

适用条件:
  - 小模型接受率 > 60% 才有收益
  - 小模型要足够快 (通常 1-2B)
  - 大小模型的 tokenizer 需一致
  
实际方案:
  - 独立小模型 (如 LLaMA-3-8B 辅助 70B)
  - Self-Speculative (用部分层做草稿)
  - Medusa (多头并行猜测)
  - EAGLE (自回归草稿头)
```

### 12.4 推理框架选型决策树

```
你需要什么样的推理？
────────────────────────────
          需要推理服务
              │
    ┌─────────┼─────────┐
    │         │         │
  云端部署   本地运行   边缘设备
    │         │         │
    │     llama.cpp    MLC-LLM
    │     / Ollama     MediaPipe
    │
    ├── 高吞吐生产环境?
    │     ├── Yes: vLLM (首选)
    │     │        - PagedAttention
    │     │        - Continuous Batching
    │     │        - OpenAI 兼容 API
    │     │
    │     └── NVIDIA GPU 且要极致速度?
    │           └── TensorRT-LLM
    │
    ├── 需要结构化输出/约束解码?
    │     └── SGLang
    │         - RadixAttention (前缀缓存)
    │         - 正则约束解码
    │         - Agent 场景首选
    │
    └── 多模型混合调度?
          └── Triton Inference Server
              + vLLM/TensorRT-LLM backend
```



---

## 十三、分布式训练体系

> 💡 **面试中频区**：理解并行策略的区别和适用场景比背细节更重要。

### 13.1 四大并行策略

```
分布式训练全景:
═══════════════════════════════════════════════════════

1. 数据并行 (Data Parallelism - DP)
   ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
   │GPU 0  │  │GPU 1  │  │GPU 2  │  │GPU 3  │
   │模型副本│  │模型副本│  │模型副本│  │模型副本│
   │Batch 0│  │Batch 1│  │Batch 2│  │Batch 3│
   └───┬───┘  └───┬───┘  └───┬───┘  └───┬───┘
       └──────────┼──────────┼──────────┘
            AllReduce 同步梯度

   适用: 模型放得下单卡
   工具: DDP (PyTorch), FSDP

2. 张量并行 (Tensor Parallelism - TP)
   将每一层的权重矩阵切分到多 GPU

   例: 线性层 Y = X·W,  W ∈ R^{d×4d}
   ┌──────────────────────────────────────┐
   │ GPU 0: Y₀ = X · W₀  (W₀ ∈ d×2d)    │
   │ GPU 1: Y₁ = X · W₁  (W₁ ∈ d×2d)    │
   │ Y = [Y₀, Y₁]  或  Y = Y₀ + Y₁      │
   └──────────────────────────────────────┘

   适用: 单层太大放不下单卡
   特点: 需要节点内高速互联 (NVLink)
   工具: Megatron-LM

3. 流水线并行 (Pipeline Parallelism - PP)
   将不同层分配到不同 GPU

   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ GPU 0   │  │ GPU 1   │  │ GPU 2   │  │ GPU 3   │
   │Layer 0-7│→│Layer 8-15│→│Layer16-23│→│Layer24-31│
   └─────────┘  └─────────┘  └─────────┘  └─────────┘

   问题: "气泡" — 前面的 GPU 等后面的
   解决: Micro-batching (GPipe) / 1F1B 调度
   适用: 模型层数多，跨节点训练

4. 序列并行 (Sequence Parallelism - SP)
   将长序列切分到多 GPU
   每个 GPU 处理序列的一部分
   适用: 超长上下文训练 (128K+)
   通常与 TP 配合使用
```

### 13.2 ZeRO 优化器 【中频】

```
训练时 GPU 显存组成 (FP16 Mixed Precision):
═════════════════════════════════════════════
假设模型参数量 = Φ

  模型参数 (FP16):     2Φ bytes
  梯度 (FP16):         2Φ bytes
  优化器状态 (FP32):   12Φ bytes  ← 最大头!
    - Adam: FP32 参数副本 (4Φ) + 一阶矩 (4Φ) + 二阶矩 (4Φ)
  ──────────────────────────
  总计:                16Φ bytes

  7B 模型: 16 × 7B = 112 GB  (超过 A100-80GB!)

ZeRO (Zero Redundancy Optimizer):
─────────────────────────────────
核心: 将冗余状态分片到多 GPU

  ZeRO-1: 优化器状态分片
    每 GPU: 2Φ + 2Φ + 12Φ/N
    → N=8 时: 4Φ + 1.5Φ = 5.5Φ

  ZeRO-2: + 梯度分片
    每 GPU: 2Φ + 2Φ/N + 12Φ/N
    → N=8 时: 2Φ + 1.75Φ = 3.75Φ

  ZeRO-3 (FSDP): + 参数分片
    每 GPU: 2Φ/N + 2Φ/N + 12Φ/N = 16Φ/N
    → N=8 时: 2Φ = 14GB (7B 模型)
    → 但需要 AllGather 参数，通信量增加

  ┌────────────────────────────────────────────────┐
  │                                                │
  │   每 GPU 显存 (7B 模型, 8 GPU)                 │
  │                                                │
  │   无优化:    112 GB  ███████████████████████    │
  │   ZeRO-1:    44 GB  █████████                  │
  │   ZeRO-2:    30 GB  ██████                     │
  │   ZeRO-3:    14 GB  ███                        │
  │                                                │
  └────────────────────────────────────────────────┘
```

### 13.3 实际训练配置参考

| 模型规模 | GPU 配置 | 并行策略 | 框架 |
|---------|---------|---------|------|
| 7B 全参微调 | 4×A100-80GB | FSDP (ZeRO-3) | HuggingFace + DeepSpeed |
| 7B QLoRA | 1×A100-80GB | 单卡 | PEFT + bitsandbytes |
| 13B 全参微调 | 8×A100-80GB | FSDP | DeepSpeed ZeRO-3 |
| 70B QLoRA | 1-2×A100-80GB | 单卡/DP | PEFT + bitsandbytes |
| 70B 全参微调 | 32×A100-80GB | TP+PP+DP | Megatron-LM |
| 400B+ 预训练 | 1000+×H100 | TP+PP+DP+SP | Megatron-LM + 自研 |

---

## 十四、评估体系深度补充

### 14.1 主流 Benchmark 详解

```
Benchmark 分层体系:
═══════════════════

基础能力 Benchmark:
──────────────────
  MMLU (Massive Multitask Language Understanding)
  ├── 57 个学科，14K 多选题
  ├── STEM / 人文 / 社科 / 其他
  ├── 难度: 高中 → 专业级
  └── 局限: 选择题不能评估生成能力

  HumanEval / MBPP (代码生成)
  ├── HumanEval: 164 个 Python 编程题
  ├── MBPP: 974 个基础编程题
  ├── 评估: pass@k (k 次采样通过率)
  └── 扩展: HumanEval+ (更严格测试用例)

  GSM8K / MATH (数学推理)
  ├── GSM8K: 8.5K 小学数学应用题
  ├── MATH: 12.5K 竞赛级数学题
  └── 评估: 最终答案精确匹配

对齐评估 Benchmark:
──────────────────
  MT-Bench
  ├── 80 个多轮对话题目
  ├── 8 个类别 (写作/角色扮演/推理/...)
  ├── GPT-4 作为裁判打 1-10 分
  └── 问题: 裁判偏好 + 位置偏差

  Chatbot Arena (LMSYS)
  ├── 匿名 A/B 对比，用户投票
  ├── ELO 排名 (类似国际象棋)
  ├── 100K+ 投票累积
  └── 业界公认最权威排名

  AlpacaEval
  ├── 805 个指令，与 GPT-4 对比
  ├── Win Rate 和 Length-Controlled WR
  └── 自动化，可复现
```

### 14.2 评估的陷阱与最佳实践

| 陷阱 | 说明 | 应对 |
|------|------|------|
| Benchmark 过拟合 | 训练数据包含测试题 (数据泄露) | 用 held-out 集 + 动态 Benchmark |
| 选择题 ≠ 真实能力 | MMLU 高分不代表真实任务好 | 结合业务场景评估 |
| 裁判偏见 | GPT-4 倾向长回答、特定风格 | 多裁判 + 人类评估交叉验证 |
| 单指标误导 | 只看平均分忽略分布 | 报告分项指标 + 标准差 |
| 静态 Benchmark 老化 | 公开越久，越容易被"刷榜" | 持续更新，LiveBench |

**业务场景评估框架：**

```python
# 业务评估流水线示例
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EvalCase:
    query: str
    reference: str       # 参考答案 (可选)
    context: str         # 检索上下文 (RAG 场景)
    metadata: Dict       # 场景标签、难度等

class BusinessEvaluator:
    # 面向业务的 LLM 评估框架
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        # ["accuracy", "faithfulness", "latency", "cost"]
    
    def evaluate_accuracy(self, prediction: str, reference: str) -> float:
        # 语义匹配准确率 (用 embedding 相似度或 LLM-as-Judge)
        # 方案1: Embedding 余弦相似度
        sim = cosine_similarity(embed(prediction), embed(reference))
        # 方案2: LLM 裁判
        score = llm_judge(prediction, reference, rubric=self.rubric)
        return score
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        # 忠实度: 答案是否基于提供的上下文 (RAG 关键指标)
        claims = extract_claims(answer)
        supported = [c for c in claims if is_supported(c, context)]
        return len(supported) / len(claims) if claims else 0
    
    def evaluate_safety(self, response: str) -> Dict:
        # 安全评估: 有害内容、隐私泄露、偏见检测
        return {
            "toxicity": toxicity_check(response),
            "pii_leak": pii_detection(response),
            "bias": bias_evaluation(response),
        }
    
    def run_evaluation(self, cases: List[EvalCase], model) -> Dict:
        # 完整评估流水线
        results = []
        for case in cases:
            prediction = model.generate(case.query, case.context)
            result = {
                "accuracy": self.evaluate_accuracy(prediction, case.reference),
                "faithfulness": self.evaluate_faithfulness(prediction, case.context),
                "safety": self.evaluate_safety(prediction),
                "latency_ms": measure_latency(model, case.query),
                "cost_per_query": estimate_cost(model, case.query, prediction),
            }
            results.append(result)
        return aggregate_metrics(results)
```

---

## 十五、模型选型实战指南

### 15.1 开源 vs 闭源决策矩阵

| 考量维度 | 选闭源 (GPT-4/Claude) | 选开源 (LLaMA/Qwen) |
|---------|---------------------|---------------------|
| 数据隐私 | 可接受数据出境 | 严格数据合规 |
| 定制需求 | 通用场景，Prompt 够用 | 需要微调、深度定制 |
| 成本预算 | API 成本可控 (<$10K/月) | 有 GPU 资源，长期成本优化 |
| 延迟要求 | 可接受网络延迟 | 需要 <100ms 本地推理 |
| 技术团队 | 小团队，快速上线 | 有 ML 团队，可运维 |
| 质量要求 | 极致质量 (如 GPT-4o) | 大部分场景够用 |
| 离线需求 | 全在线 | 需要离线/内网部署 |

### 15.2 开源模型选型 (2024-2025)

```
按规模选择:
═══════════
≤ 3B (手机/边缘):
  ├── Phi-3 Mini (3.8B): 微软，推理能力强
  ├── Qwen-2.5-3B: 中文友好
  └── Gemma-2 2B: Google，高效

7-8B (单卡可跑):
  ├── LLaMA-3.1-8B: 社区生态最好
  ├── Qwen-2.5-7B: 中文+代码能力强
  ├── Mistral-7B: 欧洲出品，GQA+滑动窗口
  └── DeepSeek-V2-Lite: MoE，高效

14-32B (甜蜜区):
  ├── Qwen-2.5-32B: 性价比之王
  ├── DeepSeek-V2 (16B 激活): MoE 高效
  └── Mistral Small (22B): 多语言

70B+ (多卡大模型):
  ├── LLaMA-3.1-70B: 综合最强开源
  ├── Qwen-2.5-72B: 中文最强
  └── DeepSeek-V3 (671B/37B激活): MoE 架构

推理模型 (2025 热门):
  ├── DeepSeek-R1: 推理链，数学/代码顶级
  ├── QwQ-32B: 通义推理模型
  └── OpenAI o1/o3: 闭源推理标杆
```

### 15.3 微调方法选择流程

```
你的场景需要微调吗？
────────────────────
          有特定任务
              │
    ┌─────────┼─────────┐
    │                     │
  Prompt 工程能解决?      不能
    │                     │
   Yes                  数据量?
    │               ┌─────┼─────┐
  用 Prompt          │           │
                   <1K 条      1K-100K    >100K
                    │           │           │
                  Few-Shot    LoRA/QLoRA   全参微调
                  + RAG       性价比最高   (有 GPU 集群)
                              │
                         模型规模?
                    ┌─────┼─────┐
                  ≤13B        70B+
                    │           │
                  LoRA        QLoRA
                (1-2 GPU)   (1-2 GPU)
```

---

## 附录 D：扩展面试高频题与深度解析

### D.1 Transformer 相关

**Q: 为什么 Transformer 要用多头注意力而不是单头？**

> 多头注意力的本质是低秩近似的集成。单头注意力用完整的 d_k = d_model 维度计算一个注意力分布，虽然全秩但只有一个"视角"。多头注意力将其分为 h 个 d_k = d_model/h 的子空间，每个头学习不同的语义关系（语法依赖、共指消解、语义相似等），最终 concat 后通过输出投影融合，在参数量不变的前提下获得了更丰富的表达能力。经验上，h=32 时效果显著好于 h=1，但 h 过大（如 h=128 且 d_k 只有 64）会导致单头表达力不足。

**Q: RoPE 为什么能编码相对位置？**

> RoPE 对 Q 和 K 的每对维度施加旋转变换：将位置 m 的向量旋转 m·θ 角度。两个位置 m 和 n 的内积 q_m · k_n 等价于 q · R(m-n) · k，只依赖相对距离 m-n 而不依赖绝对位置。这是因为旋转矩阵的乘法满足 R(m)^T · R(n) = R(n-m) 的性质。不同维度的 θ 值按指数递减（θ_i = 10000^{-2i/d}），低维度高频变化关注局部关系，高维度低频变化关注全局关系。

**Q: Flash Attention 如何做到不存储 n×n 的注意力矩阵？**

> Flash Attention 的核心是 Online Softmax 算法。传统 Softmax 需要先完整计算 QK^T 再归一化，因此必须存储整个 n×n 矩阵。Online Softmax 维护运行中的最大值 m 和求和 l，对每个新的 KV 块增量更新，最终得到精确的 Softmax 结果。计算全部在 GPU SRAM（片上内存）中进行，只需要 O(block_size²) 的临时空间。整个过程是精确计算而非近似，与标准注意力在数学上完全等价。

### D.2 微调与对齐

**Q: DPO 和 RLHF 在什么情况下该选哪个？**

> DPO 适合：团队小、GPU 有限、偏好数据已经收集好（离线场景）、追求训练稳定性。RLHF (PPO) 适合：有足够 GPU (需要同时加载 4 个模型)、需要最高质量输出、奖励信号复杂且需要在线迭代优化。实际选择中，大多数团队应首选 DPO 或 GRPO，因为 PPO 的工程复杂度和超参调优成本很高。DeepSeek-R1 证明了 GRPO + 规则奖励可以达到甚至超过传统 RLHF 的效果。

**Q: LoRA 的 rank 怎么选？**

> 经验法则：r=8 用于简单适配任务（风格迁移、格式调整），r=32-64 用于中等复杂度（领域知识注入），r=128+ 接近全参数微调效果但失去效率优势。更精确的选择方法：从 r=16 开始，观察验证集 loss，逐步增加直到边际收益递减。alpha 通常设为 r 的 1-2 倍。target_modules 通常选择所有 attention 投影层 + FFN 层（即 q/k/v/o_proj + gate/up/down_proj），只选 q/v_proj 是早期做法，覆盖所有层效果更好。

### D.3 推理与部署

**Q: vLLM 和 SGLang 怎么选？**

> vLLM 是通用首选：PagedAttention 架构成熟、社区最大、模型支持最广、OpenAI 兼容 API 开箱即用。SGLang 在以下场景更优：(1) 大量请求共享 System Prompt → RadixAttention 的前缀缓存更高效；(2) 需要约束解码（JSON schema、正则表达式）→ 原生支持且不损性能；(3) Agent 场景多轮调用 → 编程式 API 更灵活。如果不确定，先用 vLLM，遇到瓶颈再评估 SGLang。

**Q: 生产环境如何控制 LLM 推理成本？**

> 成本优化分层策略：(1) 模型选择——不是所有任务都需要 70B，80% 的查询用 7-8B 可以解决，配合路由机制按复杂度分发；(2) 量化——AWQ/GPTQ 4bit 量化几乎无损但显存减半；(3) 缓存——Semantic Cache 缓存相似查询结果，KV Cache 共享系统提示；(4) 批处理——Continuous Batching 提高 GPU 利用率；(5) 混合部署——高频低延迟需求用 GPU，低频长文本用 CPU；(6) Prompt 优化——压缩 System Prompt，减少不必要的上下文。



---

## 十六、Scaling Laws 与训练经济学

> 💡 **面试中频区**：理解 Scaling Laws 对模型规模选择的指导意义。

### 16.1 Kaplan vs Chinchilla Scaling Laws

```
Kaplan Scaling Laws (2020, OpenAI):
═══════════════════════════════════
L(N) ∝ N^{-0.076}    (模型越大，Loss 越低)
L(D) ∝ D^{-0.095}    (数据越多，Loss 越低)
L(C) ∝ C^{-0.050}    (计算越多，Loss 越低)

→ 结论: 优先增大模型，数据次之
→ 导致了 GPT-3 (175B) 的诞生，但只用 300B tokens 训练

Chinchilla Scaling Laws (2022, DeepMind):
═════════════════════════════════════════
修正 Kaplan: 模型和数据应"等比例"扩大

最优比例: D ≈ 20 × N
  → 7B 模型需要 ~140B tokens
  → 70B 模型需要 ~1.4T tokens

实际趋势 (2024-2025):
  → 数据量远超 Chinchilla 建议
  → LLaMA-3 8B 用了 15T tokens (Chinchilla 建议 160B)
  → "过训练" 小模型: 用更多数据让小模型更强
  → 推理成本比训练成本更重要
```

### 16.2 训练成本估算

```
训练 FLOPs 估算公式:
  C ≈ 6 × N × D
  N = 参数量, D = 训练 token 数

示例: 训练 7B 模型，1T tokens
  C = 6 × 7B × 1T = 42 × 10^21 FLOPs = 42 ZFLOPs

A100 GPU:
  BF16 算力: 312 TFLOPS
  实际利用率: ~40-50% (MFU)
  有效算力: ~150 TFLOPS

训练时间:
  单卡: 42e21 / 150e12 = 2.8 × 10^8 秒 ≈ 8.9 年
  64 卡: 8.9 年 / 64 ≈ 50 天
  256 卡: ≈ 12 天

成本 (按 $2/GPU·hour):
  64 × A100 × 50 天 × 24h × $2 = $153,600
  → 7B 模型: ~$15-20 万
  → 70B 模型: ~$150-200 万
  → 400B+ 模型: ~$5000-8000 万
```

---

## 十七、数据工程实战

### 17.1 预训练数据处理流水线

```
完整流水线:
═══════════

  Raw Data (Common Crawl, Books, Code, Wiki...)
      │
  ┌───▼───┐
  │ 格式解析 │ HTML → 纯文本, PDF → 文本
  └───┬───┘
      │
  ┌───▼───┐
  │语言检测│ fastText langid, 保留目标语言
  └───┬───┘
      │
  ┌───▼───┐
  │质量过滤│ 困惑度过滤(KenLM), 长度/特殊字符比例
  └───┬───┘
      │
  ┌───▼───┐
  │ 去重   │ MinHash + LSH (近似去重)
  │       │ Exact substring dedup (精确去重)
  └───┬───┘
      │
  ┌───▼───┐
  │毒性过滤│ 分类器检测有害内容
  │       │ PII 检测 (姓名/邮箱/电话)
  └───┬───┘
      │
  ┌───▼───┐
  │数据混合│ 按比例混合不同来源
  │       │ Web:Book:Code:Wiki = 80:5:10:5
  └───┬───┘
      │
  ┌───▼───┐
  │Tokenize│ SentencePiece / tiktoken
  │+ 打包  │ 拼接成固定长度序列
  └───────┘
```

### 17.2 SFT 数据构造最佳实践

| 原则 | 说明 | 实践 |
|------|------|------|
| 质量 > 数量 | 1K 高质量 > 10K 低质量 | 人工审核 + GPT-4 过滤 |
| 多样性 | 覆盖不同任务类型、难度 | 分类采样，不要偏科 |
| 格式一致性 | 对话格式统一 | 标准化模板 |
| 长度分布 | 避免全是短回答或全是长回答 | 混合不同长度 |
| 安全数据 | 混入拒绝回答样本 | 比例 5-10% |
| 合成数据 | Self-Instruct / Evol-Instruct | GPT-4 生成 + 人工校验 |

```python
# SFT 对话数据格式示例 (Alpaca 格式)
sft_example = {
    "instruction": "请解释什么是梯度消失问题，以及如何解决它。",
    "input": "",  # 可选的额外上下文
    "output": "梯度消失是深度神经网络训练中的常见问题..."
}

# ChatML 格式 (多轮对话)
chat_example = [
    {"role": "system", "content": "你是一位 AI 助手。"},
    {"role": "user", "content": "什么是 Transformer？"},
    {"role": "assistant", "content": "Transformer 是一种基于自注意力机制的..."},
    {"role": "user", "content": "它和 RNN 有什么区别？"},
    {"role": "assistant", "content": "主要区别在于..."},
]
```

---

## 附录 E：LLM 性能基准数据参考

### E.1 主流模型 Benchmark 横评 (2025 Q1)

| 模型 | 参数量 | MMLU | HumanEval | GSM8K | MT-Bench | Arena ELO |
|------|--------|------|-----------|-------|----------|----------|
| GPT-4o | 未知 | 88.7 | 90.2 | 95.3 | 9.3 | 1287 |
| Claude 3.5 Sonnet | 未知 | 88.3 | 92.0 | 96.4 | 9.1 | 1269 |
| DeepSeek-V3 | 671B(37B) | 87.1 | 82.6 | 90.2 | 8.8 | 1252 |
| LLaMA-3.1 70B | 70B | 86.0 | 80.5 | 95.1 | 8.6 | 1218 |
| Qwen-2.5 72B | 72B | 85.3 | 86.4 | 93.2 | 8.7 | 1235 |
| LLaMA-3.1 8B | 8B | 68.4 | 72.2 | 79.6 | 8.0 | 1152 |
| Qwen-2.5 7B | 7B | 74.2 | 79.8 | 82.4 | 7.8 | 1138 |
| Mistral-7B | 7B | 62.5 | 64.4 | 52.2 | 7.6 | 1071 |

> 注: 数据来源于各模型技术报告和 LMSYS Chatbot Arena，部分为估计值。实际性能因评测条件而异。

### E.2 推理性能参考 (单 A100-80GB)

| 模型 | 量化 | Prefill (tokens/s) | Decode (tokens/s) | 显存占用 |
|------|------|-------------------|-------------------|----------|
| LLaMA-3 8B | FP16 | 15000 | 120 | 16GB |
| LLaMA-3 8B | AWQ-4bit | 18000 | 150 | 5GB |
| LLaMA-3 70B | FP16 | 2200 | 18 | 140GB (2卡) |
| LLaMA-3 70B | AWQ-4bit | 4500 | 35 | 36GB |
| Qwen-2.5 72B | GPTQ-4bit | 4200 | 32 | 38GB |
| DeepSeek-V3 | - | 专用集群 | 60 (per query) | 多节点 |

> 测试环境: vLLM 0.5+, batch_size=1, 仅供参考。吞吐量随 batch_size 增大显著提升。

### E.3 训练框架选择速查

```
你要做什么？
─────────────
     预训练 ──── Megatron-LM (大规模) / NanoGPT (学习)
     │
     SFT ─────── LLaMA-Factory (简单) / Axolotl (灵活)
     │
     LoRA ────── PEFT + Transformers / Unsloth (2× 加速)
     │
     DPO/GRPO ── TRL (官方) / LLaMA-Factory
     │
     RLHF (PPO) ─ TRL / OpenRLHF (分布式)
```

---

## 附录 F：常见错误与排查指南

| 症状 | 可能原因 | 排查方法 |
|------|---------|----------|
| Loss 不下降 | 学习率太高/太低; 数据问题 | 检查 LR schedule; 验证数据加载 |
| Loss Spike | 数据异常值; 梯度爆炸 | 检查训练数据; 减小 LR / 加 gradient clip |
| OOM (显存不足) | batch_size 太大; 序列太长 | 减 batch; 用 gradient checkpointing; 用 ZeRO |
| 微调后变笨 | 灾难性遗忘; 过拟合 | 降 LR; 减少 epoch; 增加数据多样性 |
| 推理输出乱码 | tokenizer 不匹配; 量化错误 | 检查 tokenizer 版本; 换量化方法 |
| 推理速度慢 | 未启用 KV Cache; batch=1 | 确认 use_cache=True; 用 Continuous Batching |
| 生成重复内容 | 温度太低; 重复惩罚未设 | temperature ≥ 0.7; repetition_penalty = 1.1 |
| 中文乱码 | tokenizer 不支持中文 | 用支持中文的模型 (Qwen/ChatGLM) |


---

## 附录 G：LLM 架构超参数速查

### 主流模型架构对比

| 模型 | 层数 | 隐藏维度 | Head数 | KV Head | FFN 维度 | 上下文 | 词表大小 |
|------|------|---------|-------|---------|---------|--------|----------|
| LLaMA-3 8B | 32 | 4096 | 32 | 8 (GQA) | 14336 | 8K→128K | 128256 |
| LLaMA-3 70B | 80 | 8192 | 64 | 8 (GQA) | 28672 | 8K→128K | 128256 |
| Qwen-2.5 7B | 28 | 3584 | 28 | 4 (GQA) | 18944 | 128K | 152064 |
| Qwen-2.5 72B | 80 | 8192 | 64 | 8 (GQA) | 29568 | 128K | 152064 |
| Mistral-7B | 32 | 4096 | 32 | 8 (GQA) | 14336 | 32K | 32000 |
| DeepSeek-V3 | 61 | 7168 | 128 | MLA | MoE×256 | 128K | 129280 |
| GPT-4 (推测) | 120 | 12288 | 96 | 未知 | MoE | 128K | ~100K |

### 关键架构组件选择

| 组件 | 现代标配 | 替代方案 | 说明 |
|------|---------|---------|------|
| 注意力 | GQA | MHA, MQA, MLA | GQA 是平衡效率和质量的主流选择 |
| 位置编码 | RoPE | ALiBi | RoPE + YaRN 可扩展长上下文 |
| 归一化 | RMSNorm (Pre) | LayerNorm | Pre-Norm 训练更稳定 |
| 激活函数 | SwiGLU | GELU | SwiGLU 需要 3 个权重矩阵 |
| FFN 维度 | 8/3 × d | 4 × d | SwiGLU 用 8/3d 保持参数量 |
| Bias | 无 Bias | 有 Bias | 去掉 Bias 不影响效果但省参数 |
| 词表大小 | 100K-150K | 30K-50K | 大词表提升多语言和代码能力 |
| Embedding | 共享输入/输出 | 独立 | 共享节省参数但限制灵活性 |

### 训练超参数经验值

| 超参数 | 7B 模型 | 70B 模型 | 说明 |
|--------|---------|---------|------|
| 学习率 (峰值) | 3e-4 | 1.5e-4 | 模型越大 LR 越小 |
| 权重衰减 | 0.1 | 0.1 | 标准值 |
| Warmup Steps | 2000 | 2000 | 通常固定 |
| LR Schedule | Cosine | Cosine | 衰减到峰值的 10% |
| Batch Size (tokens) | 4M | 8-16M | 大模型用大 Batch |
| 梯度裁剪 | 1.0 | 1.0 | 防止梯度爆炸 |
| Precision | BF16 | BF16 | H100 可用 FP8 |
| Dropout | 0 | 0 | 预训练通常不用 Dropout |

### 微调超参数经验值

| 超参数 | SFT | LoRA | QLoRA |
|--------|-----|------|-------|
| 学习率 | 1-2e-5 | 1-3e-4 | 1-3e-4 |
| Epoch | 2-3 | 2-5 | 3-5 |
| Batch Size | 128 | 128 | 64-128 |
| LoRA rank | - | 32-64 | 32-64 |
| LoRA alpha | - | 64-128 | 64-128 |
| Warmup Ratio | 0.03 | 0.03 | 0.03 |
| 序列长度 | 2048-4096 | 2048-4096 | 2048-4096 |
| 梯度累积 | 按需 | 按需 | 按需 |

### 推理参数调优指南

| 参数 | 创意写作 | 事实问答 | 代码生成 | 数据提取 |
|------|---------|---------|---------|----------|
| temperature | 0.8-1.0 | 0.1-0.3 | 0.2-0.4 | 0 |
| top_p | 0.9-0.95 | 0.8-0.9 | 0.85-0.95 | 1.0 |
| top_k | 40-60 | 10-20 | 20-40 | 1 |
| repetition_penalty | 1.1-1.2 | 1.0-1.05 | 1.0 | 1.0 |
| max_tokens | 2000+ | 500-1000 | 2000+ | 按需 |
| frequency_penalty | 0.3-0.5 | 0 | 0 | 0 |

> 💡 **关键原则**: temperature=0 用于确定性输出 (数据提取、分类)；temperature=0.7 是通用默认值；>1.0 增加创意但可能产生胡言乱语。



### 推理模型 (Reasoning Models) 趋势

| 模型 | 方法 | 特点 | 适用场景 |
|------|------|------|---------|
| OpenAI o1/o3 | 隐式思维链 | 内部推理过程不可见，效果最强 | 数学/竞赛/复杂推理 |
| DeepSeek-R1 | GRPO + 长思维链 | 开源，可见推理过程，67B 参数 | 数学/代码/逻辑推理 |
| QwQ-32B | 长思维链 | 通义出品，32B 高效推理 | 通用推理任务 |
| Claude 3.5 | 扩展思考 | Extended Thinking 模式 | 分析/写作/推理 |

**推理模型 vs 标准模型的核心区别：**



> 💡 **选择建议**: 简单任务 (分类/提取/翻译) 用标准模型；复杂任务 (数学/代码/多步推理) 用推理模型。可以用路由机制按任务难度自动分发。



---

> 本文档共覆盖 LLM 技术栈 17 个核心主题，7 个附录，涵盖从架构原理到生产部署的完整链路。持续更新中。


> 📌 **最后的建议**：LLM 技术栈的知识面很广，但面试中最核心的还是 **Transformer 基础 + 微调技术 + 推理部署** 这三块。把这三块吃透，再向两端（预训练、评估）扩展。理解原理 > 记忆细节，能讲清楚 trade-off > 能背公式。对于 2024-2025 的面试，**GRPO/DeepSeek-R1、MoE 架构、推理优化（vLLM/SGLang）** 是新增的高频考点，务必掌握。
