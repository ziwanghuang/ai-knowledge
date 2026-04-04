# RAG 完整技术栈深度解析

> 🎯 **定位**：从 PoC 到生产级的 RAG 系统全景深度解析，面向有一定基础的工程师和架构师。
> 本文覆盖架构设计、文档处理、Embedding、向量数据库、检索策略、生产挑战、评估体系七大板块。
> 🔴 = 面试高频 | 🟡 = 面试中频 | 🟢 = 加分项

---

## 目录

- [一、RAG 架构设计](#一rag-架构设计)
- [二、文档解析与分块](#二文档解析与分块)
- [三、Embedding 模型](#三embedding-模型)
- [四、向量数据库](#四向量数据库)
- [五、检索策略](#五检索策略)
- [六、生产级 RAG 十大挑战](#六生产级-rag-十大挑战)
- [七、RAG 评估](#七rag-评估)
- [附录：面试高频考点速查](#附录面试高频考点速查)

---

## 一、RAG 架构设计

### 1.1 为什么需要 RAG？🔴

LLM 的三大固有缺陷催生了 RAG：

| 问题 | 表现 | RAG 如何解决 |
|------|------|-------------|
| **知识截断** | 训练数据有截止日期，无法获取最新信息 | 实时检索外部知识库 |
| **幻觉** | 一本正经地编造不存在的事实 | 基于检索到的真实文档生成，可验证 |
| **领域知识不足** | 对私有/垂直领域知识一无所知 | 索引企业内部文档，注入专有知识 |

**核心洞察**：RAG 本质上是一种**外部记忆扩展机制**——不改变模型参数，而是在推理时动态注入相关知识。这类似于人类"开卷考试"：不需要记住一切，只需要在需要时找到正确的参考资料。

### 1.2 RAG 演进路线 🔴

```
Naive RAG ──────→ Advanced RAG ──────→ Modular RAG ──────→ Agentic RAG
(2020-2022)        (2023)               (2024)              (2024-2025)
```

#### Naive RAG（朴素 RAG）

最简单的"检索 + 拼接 + 生成"三步走：

```
用户问题 → Embedding → 向量检索 Top-K → 拼接到 Prompt → LLM 生成答案
```

**问题**：
1. 检索质量差 — 用户问题和文档的语义鸿沟（vocabulary mismatch）
2. 分块质量差 — 机械切分导致上下文断裂
3. 生成质量差 — 检索到的内容冗余或矛盾，模型不知如何取舍
4. 无反馈机制 — 一条路走到黑，不知道检索结果好不好

#### Advanced RAG（增强 RAG）

在 Naive RAG 的每个环节加入优化：

```
Pre-Retrieval                    Retrieval                    Post-Retrieval
┌─────────────────┐    ┌───────────────────────┐    ┌──────────────────────┐
│ • Query Rewrite │    │ • Hybrid Search       │    │ • Reranking          │
│ • HyDE          │ →  │ • Multi-Query         │ →  │ • Context Compression│
│ • Step-back     │    │ • Parent-Child        │    │ • Dedup & Filter     │
│ • Query Routing │    │ • Sentence Window     │    │ • Lost-in-Middle Fix │
└─────────────────┘    └───────────────────────┘    └──────────────────────┘
```

#### Modular RAG（模块化 RAG）

将 RAG 拆解为可编排的模块，按需组合：

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Router  │ → │ Retriever│ → │ Reranker │ → │Generator │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
      ↓              ↓              ↓              ↓
 路由模块可选    多种检索策略    多种重排方案    多种生成策略
 Web/DB/KG     Dense/Sparse   Cross-Enc/LLM  直接/CoT/Multi-hop
```

**核心思想**：不同类型的问题走不同的检索和生成路径。

#### Agentic RAG（Agent 驱动 RAG）

最新范式 — Agent 自主决策何时检索、检索什么、如何验证：

```
Agent Loop:
  1. 分析用户问题 → 判断是否需要检索
  2. 如需检索 → 自主构造查询、选择数据源
  3. 评估检索结果 → 不满意则重试（换查询/换数据源）
  4. 生成答案 → 自我验证 → 输出
```

代表实现：Self-RAG、CRAG、Adaptive RAG（基于 LangGraph 的状态图实现）。

### 1.3 RAG vs Fine-Tuning vs Long Context 🔴

这是面试必考的决策题：

| 维度 | RAG | Fine-Tuning | Long Context |
|------|-----|-------------|-------------|
| **知识更新** | ✅ 实时，改文档即生效 | ❌ 需重新训练 | ⚠️ 需每次都塞入上下文 |
| **成本** | 中等（向量库 + 检索） | 高（训练 GPU + 数据标注） | 高（每次调用的 token 成本） |
| **幻觉控制** | ✅ 可引用来源验证 | ❌ 无法追溯 | ⚠️ 上下文过长时容易迷失 |
| **私有数据** | ✅ 数据留在本地 | ⚠️ 数据进入模型参数 | ⚠️ 每次通过 API 发送 |
| **知识量** | ✅ 无上限 | ❌ 受模型容量限制 | ❌ 受窗口大小限制 |
| **行为改变** | ❌ 不能改变模型风格 | ✅ 可以改变回答风格/格式 | ❌ 不能改变模型风格 |
| **延迟** | ⚠️ 增加检索延迟 | ✅ 无额外延迟 | ⚠️ 输入越长延迟越高 |

**决策树**：

```
需要最新/私有知识？
  ├─ 是 → 知识量 > 上下文窗口？
  │         ├─ 是 → RAG ✅
  │         └─ 否 → Long Context（简单场景）或 RAG（需要可追溯性）
  └─ 否 → 需要改变模型行为/风格？
            ├─ 是 → Fine-Tuning
            └─ 否 → 直接用基础模型
```

**最佳实践**：三者不互斥。生产中常见 RAG + Fine-Tuning 的组合：用 SFT 让模型学会"如何使用检索到的文档"（格式、引用习惯），用 RAG 提供具体知识。

### 1.4 系统架构设计 🟡

生产级 RAG 系统的典型双链路架构：

```
═══════════════ 离线索引链路 (Offline Indexing Pipeline) ═══════════════

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 数据源    │ → │ 文档解析  │ → │ 文本分块  │ → │ Embedding│ → │ 向量存储  │
│ S3/DB/API│   │ PDF/HTML  │   │ 多策略    │   │ 批量计算  │   │ Milvus   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
      │                                                            │
      └── CDC / 定时任务 触发增量更新 ──────────────────────────────→ │

═══════════════ 在线检索链路 (Online Retrieval Pipeline) ═══════════════

┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 用户查询  │ → │ Query    │ → │ 多路检索  │ → │ 重排序    │ → │ Prompt   │
│          │   │ 预处理    │   │ Dense+   │   │ Cross-   │   │ 构建     │
│          │   │ 改写/扩展  │   │ Sparse   │   │ Encoder  │   │ + 生成   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

**关键设计决策**：

1. **索引流水线的幂等性**：同一文档重复处理不应产生重复索引（用文档 hash 做去重）
2. **增量 vs 全量更新**：小规模文档变更用增量索引（CDC 监听）；Embedding 模型升级时需要全量重建
3. **检索链路的延迟控制**：总延迟 = Query预处理 + 向量检索 + 重排序 + LLM生成。向量检索通常 <50ms，瓶颈在 LLM 生成
4. **缓存层**：语义缓存（semantic cache）— 如果新问题和历史问题语义相似度 >0.95，直接返回缓存结果

---

## 二、文档解析与分块

### 2.1 文档解析的核心挑战 🟡

文档解析是 RAG 中最脏最累但最关键的环节 — "Garbage In, Garbage Out"。

#### 不同文档类型的解析策略

| 文档类型 | 难度 | 核心挑战 | 推荐工具链 |
|---------|------|---------|-----------|
| **原生 PDF** | ⭐⭐ | 多栏排版、页眉页脚、脚注 | pdfplumber → Marker → MinerU |
| **扫描 PDF** | ⭐⭐⭐⭐ | OCR 精度、版面分析、倾斜校正 | PaddleOCR + PP-Structure / MinerU |
| **Word (.docx)** | ⭐⭐ | 嵌套表格、嵌入图片、修订痕迹 | python-docx + Unstructured |
| **HTML** | ⭐⭐⭐ | 动态渲染(SPA)、广告噪声、模板剥离 | Trafilatura / Jina Reader / Playwright |
| **Markdown** | ⭐ | 最友好的格式 | 直接解析层级结构 |
| **PPT (.pptx)** | ⭐⭐⭐ | 图文混排、动画、SmartArt | python-pptx + VLM 描述 |
| **表格 (Excel/CSV)** | ⭐⭐⭐ | 合并单元格、跨 Sheet 引用、数据含义推断 | pandas + LLM 生成描述 |

#### PDF 解析技术栈演进

```
第一代：规则解析               第二代：深度学习版面分析          第三代：多模态解析
PyPDF2, pdfplumber          LayoutLM, PP-Structure          VLM (GPT-4V, Claude 3)
提取纯文本，结构信息丢失      识别标题/段落/表格/图片区域     直接"看"PDF 截图理解内容
                            Marker, MinerU                  精度最高但成本也最高
```

**实战建议**：

```python
# 推荐的多解析器降级策略（伪代码）
def parse_pdf(pdf_path):
    try:
        # 首先尝试 MinerU（开源最强 PDF 解析）
        result = minerU_parse(pdf_path)
        if quality_score(result) > 0.8:
            return result
    except:
        pass
    
    try:
        # 降级到 Marker
        result = marker_parse(pdf_path)
        if quality_score(result) > 0.6:
            return result
    except:
        pass
    
    # 最终降级到 OCR
    images = pdf_to_images(pdf_path)
    return paddle_ocr(images)
```

#### 表格解析深度方案

表格是 RAG 的硬骨头。三种主流方案：

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **结构化提取** | 将表格转为 Markdown/JSON | 保留结构，LLM 容易理解 | 复杂表格（合并单元格）容易出错 |
| **表格描述生成** | 用 LLM/VLM 生成表格的自然语言描述 | 语义丰富，检索友好 | 信息可能丢失，成本高 |
| **表格截图 + 多模态** | 将表格渲染为图片，用 VLM 理解 | 保留视觉信息，处理复杂排版 | 多模态模型成本高，延迟大 |

**最佳实践**：结构化提取作为主方案，表格描述生成作为补充索引，复杂表格（如财报）用截图 + VLM。

### 2.2 分块策略深度对比 🔴

分块（Chunking）是 RAG 效果的最大杠杆之一 — 块太小丢失上下文，块太大引入噪声。

#### 六种主流分块策略

**1. 固定大小分块（Fixed-size Chunking）**

```
参数：chunk_size=512, overlap=50
文本 → 每 512 token 切一刀，相邻块重叠 50 token
```

- ✅ 实现简单、速度快
- ❌ 粗暴切割语义，可能在句子中间断开
- 适用：快速原型验证

**2. 递归字符分块（Recursive Character Splitting）**

```
分隔符优先级：["\n\n", "\n", "。", ".", " ", ""]
先按段落分 → 超过 chunk_size 的段落再按句子分 → 还超再按字符分
```

- ✅ 尊重自然语言结构，LangChain 默认方案
- ⚠️ 需要针对不同语言调整分隔符
- 适用：通用场景的首选

**3. 语义分块（Semantic Chunking）**

```
计算相邻句子的 Embedding 相似度
相似度突然下降的位置 = 语义边界 → 在此处切分
```

```python
# 语义分块核心逻辑（伪代码）
sentences = split_into_sentences(text)
embeddings = embed(sentences)

breakpoints = []
for i in range(1, len(embeddings)):
    sim = cosine_similarity(embeddings[i-1], embeddings[i])
    if sim < threshold:  # threshold 通常 0.5-0.7
        breakpoints.append(i)

chunks = split_at_breakpoints(sentences, breakpoints)
```

- ✅ 语义完整性最好
- ❌ 需要额外的 Embedding 计算，速度慢
- ❌ 块大小不可控（有的块可能很大）
- 适用：对检索质量有高要求的场景

**4. 基于文档结构的分块（Document-based Chunking）**

```markdown
# 标题 1           → Chunk 1
## 标题 1.1        → Chunk 2
正文内容...        → 属于 Chunk 2
## 标题 1.2        → Chunk 3
正文内容...        → 属于 Chunk 3
```

- ✅ 完全尊重文档作者的逻辑组织
- ❌ 依赖文档有良好的层级结构（很多文档没有）
- 适用：Markdown 文档、技术文档、法律法规

**5. 滑动窗口分块（Sliding Window）**

```
Window size=500, Step=250
[0:500] [250:750] [500:1000] ...
每个 chunk 与前后 chunk 有 50% 重叠
```

- ✅ 最大程度保留上下文连续性
- ❌ 产生大量冗余数据（索引膨胀）
- 适用：上下文高度连贯的长文本（如叙事、法律条文）

**6. Late Chunking（延迟分块）**🟢

2024 年的新方案 — 先用长上下文模型对整个文档做 Embedding，再在 Embedding 空间中切分：

```
传统：文档 → 切分为 chunks → 各自独立 Embedding
Late Chunking：文档 → 长上下文模型计算 token 级 Embedding → 按位置切分 → 池化得到 chunk Embedding
```

- ✅ 每个 chunk 的 Embedding 包含了全文档的上下文信息
- ❌ 需要支持长上下文的 Embedding 模型（如 Jina 8K）
- 适用：前沿方案，尚未广泛使用

#### 分块参数调优 🟡

分块没有最优解，只有最适合的参数：

```
chunk_size 的经验值：
  - QA 场景：256-512 tokens（精确匹配，避免噪声）
  - 文档总结：1024-2048 tokens（需要完整上下文）
  - 代码文档：按函数/类自然分块

overlap 的经验值：
  - 通常为 chunk_size 的 10-20%
  - 高重叠 → 召回率高但冗余大
  - 低重叠 → 效率高但可能漏掉跨 chunk 的信息
```

**实验方法论**：

1. 准备评测集（50-100 个问题 + 标准答案 + 相关文档标注）
2. 用网格搜索测试不同参数组合
3. 用 Recall@5 和 RAGAS Faithfulness 作为核心指标
4. 选择 Recall 和 Faithfulness 都达标的最小 chunk_size（控制成本）

### 2.3 元数据提取与富化 🟡

元数据是提升检索精度的免费午餐 — 为每个 chunk 附加结构化信息：

```json
{
  "chunk_id": "doc_001_chunk_003",
  "text": "RAG 系统在金融领域的应用...",
  "metadata": {
    "source": "RAG_Best_Practices.pdf",
    "page": 15,
    "section": "3.2 行业应用",
    "doc_type": "技术白皮书",
    "author": "张三",
    "created_at": "2024-06-15",
    "language": "zh",
    "keywords": ["RAG", "金融", "风控"],
    "parent_chunk_id": "doc_001_chunk_002"  // 用于 Parent-Child 策略
  }
}
```

**LLM 辅助元数据生成**：

```python
prompt = """
请从以下文本中提取元数据：
- 主题分类（从预定义列表中选择）
- 3-5 个关键词
- 内容摘要（一句话）
- 实体（人名、组织、技术名词）

文本：{chunk_text}
"""
```

---

## 三、Embedding 模型

### 3.1 文本嵌入的工作原理 🔴

Embedding 模型的本质：将文本映射到高维稠密向量空间，使得**语义相似的文本在向量空间中距离相近**。

#### 演进路线

```
Word2Vec (2013)        → 词级别，无上下文
  ↓
ELMo/BERT (2018)       → 上下文化表示，但句子级表示需要额外处理
  ↓
Sentence-BERT (2019)   → 首个高效的句子级嵌入，twin-tower 对比学习
  ↓
E5/BGE (2023)          → 指令式嵌入，区分 query 和 document
  ↓
GTE/Jina v3 (2024)     → 支持长上下文（8K+）、多语言、Matryoshka 维度
  ↓
ColBERT/ColPali (2024)  → Late Interaction，token 级别细粒度匹配
```

#### 对比：Bi-Encoder vs Cross-Encoder vs Late Interaction 🔴

```
Bi-Encoder (双塔)                Cross-Encoder (交叉)           Late Interaction
┌────────┐ ┌────────┐          ┌──────────────────┐           ┌────────┐ ┌────────┐
│ Query  │ │  Doc   │          │  Query [SEP] Doc │           │ Query  │ │  Doc   │
│Encoder │ │Encoder │          │    BERT/T5       │           │Encoder │ │Encoder │
└───┬────┘ └───┬────┘          └───────┬──────────┘           └───┬────┘ └───┬────┘
    ↓          ↓                       ↓                      token向量  token向量
   vec_q     vec_d              relevance score                   ↓ MaxSim 交互 ↓
    ↓ cosine ↓                                                 relevance score
  similarity score

速度：最快（离线计算doc）       速度：最慢（每对都要联合）      速度：中等
精度：中等                     精度：最高                     精度：接近 Cross-Encoder
用途：初筛（检索阶段）         用途：精排（重排阶段）         用途：兼顾精度和效率
```

### 3.2 对称 vs 非对称检索 🔴

这是 Embedding 选型的核心概念：

| | 对称检索 | 非对称检索 |
|---|---------|-----------|
| **场景** | 找相似的文本（如去重、聚类） | 用短查询找长文档 |
| **Query 和 Doc** | 长度和风格相近 | Query 短（问题），Doc 长（段落） |
| **代表模型** | Sentence-BERT | E5, BGE |
| **示例** | "Python 教程" 找 "Python 入门指南" | "什么是 RAG？" 找 "RAG 是一种将外部知识..." |

**E5/BGE 的指令前缀机制**：

```python
# BGE 模型的典型用法
query = "Represent this sentence for searching relevant passages: 什么是 RAG？"
doc = "RAG（Retrieval-Augmented Generation）是一种将外部知识检索与大语言模型..."

# 通过前缀让模型知道当前是 query 还是 document，调整编码策略
```

### 3.3 主流 Embedding 模型对比 🟡

截至 2025 年初的 MTEB 排行榜状况：

| 模型 | 维度 | 最大输入 | 中文效果 | 开源 | 特点 |
|------|------|---------|---------|------|------|
| **OpenAI text-embedding-3-large** | 3072 (可截断) | 8191 | ⭐⭐⭐⭐ | ❌ | 商用最强之一，支持维度截断 |
| **BGE-M3** | 1024 | 8192 | ⭐⭐⭐⭐⭐ | ✅ | 多语言+多粒度+多功能，中文首选 |
| **BGE-large-zh** | 1024 | 512 | ⭐⭐⭐⭐⭐ | ✅ | 中文专精，短文本场景优选 |
| **GTE-Qwen2** | 1536 | 8192 | ⭐⭐⭐⭐⭐ | ✅ | 阿里出品，长上下文+中英双语 |
| **Jina Embeddings v3** | 1024 (可截断) | 8192 | ⭐⭐⭐⭐ | ✅ | Matryoshka，LoRA task adapter |
| **Cohere Embed v3** | 1024 | 512 | ⭐⭐⭐⭐ | ❌ | 内置 int8/binary 压缩 |
| **E5-Mistral-7B** | 4096 | 32768 | ⭐⭐⭐ | ✅ | 基于 LLM 的嵌入，超长上下文 |

**选型建议**：

```
中文场景 + 开源 → BGE-M3（首选）或 GTE-Qwen2
英文场景 + 开源 → E5-large-v2 或 BGE-M3
商用 API → OpenAI text-embedding-3-large
长文本 (>512 tokens) → BGE-M3 / GTE-Qwen2 / Jina v3
极致性能 → E5-Mistral-7B（需 GPU）
```

### 3.4 Embedding 模型微调 🟡

当开箱即用的模型在你的领域效果不佳时，微调是最有效的提升手段。

#### 微调数据格式

```json
// 对比学习三元组：(query, positive, negative)
{
  "query": "如何配置 Kafka 的消费者组？",
  "positive": "Kafka 消费者组通过 group.id 配置，同一组内消费者共同消费分区...",
  "negative": "Kafka 生产者的 acks 参数控制消息确认机制..."
}
```

#### 微调关键技术

| 技术 | 说明 |
|------|------|
| **对比学习 (Contrastive Learning)** | InfoNCE loss，拉近正例、推远负例 |
| **硬负样本挖掘 (Hard Negative Mining)** | 选择和 query 相似但不相关的 doc 作为负例 — 比随机负例效果提升 5-10% |
| **知识蒸馏 (Distillation)** | 用 Cross-Encoder 的打分指导 Bi-Encoder 训练 |
| **Matryoshka 训练** | 在多个维度上同时训练，支持运行时截断 |

#### 微调的投入产出比

```
场景                        | 建议
通用知识问答                | 不需要微调，用 BGE-M3 开箱即用
特定领域（医疗/法律/金融）   | 强烈建议微调，收集 1000+ 三元组，效果提升显著
代码检索                    | 建议微调，代码和自然语言的语义差距大
跨语言场景                  | 微调多语言模型比分别训练单语模型效果更好
```

### 3.5 维度选择与性能权衡 🟡

```
高维 (1024-3072)：
  - 更丰富的语义表达
  - 更高的检索精度
  - 更大的存储开销和检索延迟
  - 适合：追求精度的生产系统

低维 (256-512)：
  - 更快的检索速度
  - 更低的存储成本
  - 精度有一定下降（通常 1-3% 的 Recall 损失）
  - 适合：大规模向量（亿级）或延迟敏感场景
```

**Matryoshka Representation Learning（套娃嵌入）**🟢：

训练时在多个维度上同时优化，使得向量可以被截断到任意维度而保持良好性能：

```
原始维度 1024：[0.12, 0.45, -0.33, ..., 0.67]
截断到 512：  [0.12, 0.45, -0.33, ...]          # 仍然有效！精度下降很小
截断到 256：  [0.12, 0.45, ...]                  # 还能用，适合粗筛
```

**实用策略**：用 1024 维做精排，256 维做粗筛，同一个模型搞定两级检索。

---

## 四、向量数据库

### 4.1 向量数据库核心概念 🟡

向量数据库解决的核心问题：**在百万/亿级向量中快速找到最相似的 Top-K**。

暴力搜索（遍历所有向量计算余弦相似度）的时间复杂度是 O(N×D)，其中 N 是向量数量，D 是维度。当 N=1亿、D=1024 时，单次查询需要约 1000 亿次浮点运算 — 这不可接受。

所以需要**近似最近邻搜索（ANN, Approximate Nearest Neighbor）**：用精度换速度，在 O(logN) 或 O(1) 的时间内找到"近似"最相似的结果。

### 4.2 索引算法深度解析 🔴

#### HNSW（Hierarchical Navigable Small World）— 最常用

**核心思想**：构建多层跳表式的图结构。上层稀疏（大步跳）、下层稠密（精细搜索）。

```
Layer 3:  A ─────────────────── D                  (稀疏，快速导航)
Layer 2:  A ──── B ──────────── D ──── E           (中等密度)
Layer 1:  A ── B ── C ── D ── E ── F ── G          (稠密，精确搜索)
Layer 0:  A B C D E F G H I J K L M N O P Q R S T  (全量数据)
```

搜索过程：从最高层的入口点出发 → 在当前层贪心搜索最近的邻居 → 无法更近时下降到下一层 → 在最底层找到精确结果。

**关键参数**：
- `M` (每个节点的最大连接数)：M 越大，图越稠密，精度越高但内存越大。默认 16
- `ef_construction` (建索引时的搜索宽度)：越大索引质量越好但建索引越慢。默认 200
- `ef_search` (查询时的搜索宽度)：越大精度越高但查询越慢。默认 50-200

```
性能特征：
  - 查询速度：⭐⭐⭐⭐⭐（毫秒级）
  - 索引构建：⭐⭐⭐（较慢）
  - 内存占用：⭐⭐（高 — 需要把图结构全部加载到内存）
  - 适合：数据量 <5000万，对延迟要求极高
```

#### IVF（Inverted File Index）

**核心思想**：先用 K-Means 将向量聚类为 `nlist` 个簇 → 查询时只搜索最近的 `nprobe` 个簇。

```
建索引：
  所有向量 → K-Means 聚类为 1000 个簇 → 每个簇维护一个倒排列表

查询：
  Query → 找到最近的 10 个簇 → 只在这 10 个簇内搜索 → 大幅减少搜索范围
```

**关键参数**：
- `nlist` (簇数量)：通常 √N 到 4√N。N=100万时 nlist=1000-4000
- `nprobe` (查询时搜索的簇数)：越大精度越高但越慢。通常 nlist 的 1-10%

```
性能特征：
  - 查询速度：⭐⭐⭐⭐（取决于 nprobe）
  - 索引构建：⭐⭐⭐⭐（较快，只需一次 K-Means）
  - 内存占用：⭐⭐⭐⭐（低 — 只存储聚类中心 + 原始向量）
  - 适合：大数据量（亿级），可接受略低精度
```

#### DiskANN

**核心思想**：将向量索引存储在 SSD 上而非内存中，通过精心设计的数据布局和缓存策略实现高效检索。

```
内存中：压缩的导航图 + 聚类中心（几 GB）
SSD 上：完整的向量数据 + 精细索引（几百 GB 到 TB）

查询流程：
  1. 在内存中的压缩图上导航到候选区域
  2. 从 SSD 批量读取候选向量（利用 SSD 的高随机读性能）
  3. 精确计算距离，返回 Top-K
```

```
性能特征：
  - 查询速度：⭐⭐⭐（毫秒到十几毫秒，受 SSD 速度影响）
  - 索引构建：⭐⭐⭐（需要较长时间）
  - 内存占用：⭐⭐⭐⭐⭐（极低 — 大量数据放在磁盘上）
  - 适合：超大数据量（十亿+），内存预算有限
```

#### 索引选型决策

```
数据量 < 100K         → FLAT（暴力搜索，100% 精度）
数据量 100K - 50M     → HNSW（速度和精度的最佳平衡）
数据量 50M - 1B       → IVF + PQ（平衡内存和精度）
数据量 > 1B           → DiskANN 或 IVF + SQ（磁盘索引）
内存极度受限          → IVF + PQ 或 DiskANN
延迟极度敏感 (<1ms)   → HNSW + GPU 加速
```

### 4.3 主流向量数据库对比 🔴

| 特性 | Milvus | Qdrant | Weaviate | Pinecone | pgvector | Chroma |
|------|--------|--------|----------|----------|----------|--------|
| **部署模式** | 分布式/单机 | 单机/分布式 | 单机/分布式 | 全托管 | PG 插件 | 嵌入式 |
| **开源** | ✅ Apache 2.0 | ✅ Apache 2.0 | ✅ BSD 3 | ❌ | ✅ | ✅ |
| **混合搜索** | ✅ | ✅ | ✅ | ✅ | ⚠️ 需手动 | ⚠️ 基础 |
| **最大规模** | 万亿级 | 百亿级 | 百亿级 | 十亿级 | 千万级 | 百万级 |
| **标量过滤** | ✅ 丰富 | ✅ 丰富 | ✅ | ✅ | ✅ SQL | ⚠️ 基础 |
| **多租户** | ✅ Partition | ✅ Collection | ✅ Tenant | ✅ Namespace | ✅ Schema | ❌ |
| **GPU 加速** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **适用场景** | 大规模生产 | 中等规模生产 | 全功能需求 | 快速上手 | 已有 PG | 原型开发 |

**选型决策树**：

```
已有 PostgreSQL 基础设施？
  ├─ 是 → 数据量 < 1000 万？
  │         ├─ 是 → pgvector（零额外运维）
  │         └─ 否 → Milvus/Qdrant（专用向量库性能更好）
  └─ 否 → 只是快速验证原型？
           ├─ 是 → Chroma（嵌入式，一行代码搞定）
           └─ 否 → 需要零运维云服务？
                     ├─ 是 → Pinecone（全托管）
                     └─ 否 → 需要分布式？
                               ├─ 是 → Milvus（分布式最成熟）
                               └─ 否 → Qdrant（Rust 性能好）或 Weaviate
```

### 4.4 混合搜索：Dense + Sparse 🔴

单纯的向量搜索对**关键词/专有名词/编号**不友好：

```
Query: "RFC 7540 的主要内容"
Dense Search: 可能匹配到关于 HTTP 协议的通用文档（语义相关但不精确）
Sparse Search (BM25): 精确匹配 "RFC 7540"，找到正确文档
```

**混合搜索架构**：

```
         Query
        ┌──┴──┐
   Dense│     │Sparse
  (向量) │     │(BM25)
        ↓     ↓
  结果集A   结果集B
        └──┬──┘
      融合排序 (RRF)
           ↓
       统一结果
```

**倒数排序融合（RRF, Reciprocal Rank Fusion）**：

```python
# RRF 公式：score(doc) = Σ 1 / (k + rank_i(doc))
# k 是常数（通常 60），rank_i 是 doc 在第 i 路检索中的排名

def rrf_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**RRF 的优势**：不需要归一化不同检索系统的分数（Dense 的余弦相似度和 BM25 的分数量纲不同），只需要排名。

### 4.5 过滤策略：Pre-filter vs Post-filter 🟡

```
Pre-filter（先过滤再搜索）：
  所有向量 → 按元数据过滤 → 在过滤后子集中做向量搜索
  ✅ 保证返回结果都满足过滤条件
  ❌ 过滤后数据太少时，ANN 索引效率下降

Post-filter（先搜索再过滤）：
  所有向量 → 向量搜索 Top-K*N → 按元数据过滤 → 取 Top-K
  ✅ 向量搜索效率高
  ❌ 过滤后可能不够 K 个结果
```

**最佳实践**：大多数向量数据库支持 "filtered search"，在搜索过程中同时应用过滤 — 比纯 Pre/Post 都更高效。

---

## 五、检索策略

### 5.1 Pre-Retrieval：查询预处理 🔴

用户的原始查询往往不是好的检索查询 — 需要预处理。

#### Query Rewriting（查询改写）

用 LLM 将用户口语化的问题改写为更适合检索的查询：

```
用户原始问题："为啥我的 RAG 效果这么差？"
改写后："RAG 系统检索效果差的常见原因和优化方法"
```

```python
rewrite_prompt = """
你是一个搜索查询优化专家。
请将以下用户问题改写为更适合在知识库中搜索的查询。
要求：消除口语化表达、补充关键术语、保持核心意图不变。

用户问题：{query}
优化后的搜索查询：
"""
```

#### HyDE（Hypothetical Document Embeddings）🟡

**核心思想**：先让 LLM 生成一个"假设性的答案文档"，然后用这个假设文档的 Embedding 去检索（而非用原始 query 的 Embedding）。

```
Query: "什么是 HNSW 算法？"
  ↓ LLM 生成假设答案
HyDE Doc: "HNSW（Hierarchical Navigable Small World）是一种基于图的近似最近邻
           搜索算法。它通过构建多层图结构实现高效检索..."
  ↓ Embedding
用 HyDE Doc 的向量去检索
```

**为什么有效**：假设答案和真实答案的语义分布更接近（都是"文档风格"），而原始 query 是"问题风格" — 缩小了 query-document 的语义鸿沟。

**局限**：如果 LLM 生成的假设答案方向完全错误，检索反而会偏。适合模型有一定领域知识的场景。

#### Multi-Query（多查询检索）

从不同角度生成多个查询变体，扩大召回面：

```
原始问题："RAG 和微调哪个更好？"
  ↓ LLM 生成多个变体
Query 1: "RAG 和 Fine-Tuning 的优缺点对比"
Query 2: "什么场景用 RAG 什么场景用微调"
Query 3: "RAG vs Fine-Tuning performance comparison"
  ↓ 各自独立检索
  ↓ 合并去重结果
```

#### Step-back Prompting（后退提问）

将具体问题抽象为更高层的概念问题：

```
具体问题："vLLM 的 PagedAttention 如何减少显存碎片？"
后退问题："大语言模型推理时的显存管理机制有哪些？"

用后退问题检索到更全面的背景知识，结合原始具体问题一起输入 LLM。
```

#### Query Routing（查询路由）

根据问题类型将查询路由到不同的数据源或检索策略：

```python
def route_query(query):
    """根据问题类型选择最佳检索路径"""
    query_type = classify_query(query)  # LLM 分类
    
    if query_type == "factual":
        return vector_search(query)      # 事实性问题 → 向量检索
    elif query_type == "analytical":
        return graph_rag(query)          # 分析性问题 → 图谱检索
    elif query_type == "recent_events":
        return web_search(query)         # 时事问题 → 网络搜索
    elif query_type == "code":
        return code_search(query)        # 代码问题 → 代码语义搜索
    else:
        return hybrid_search(query)      # 默认 → 混合检索
```

### 5.2 Retrieval：核心检索模式 🔴

#### Parent-Child Retrieval（小块检索，大块返回）

这是 Advanced RAG 中最实用的模式之一：

```
索引结构：
  Parent Chunk (2000 tokens): "第三章 分布式系统设计..."
    ├─ Child Chunk (200 tokens): "CAP 定理是分布式系统的基础约束..."
    ├─ Child Chunk (200 tokens): "一致性哈希算法的核心思想是..."
    └─ Child Chunk (200 tokens): "Raft 共识算法通过选举机制..."

检索流程：
  1. 用 Child Chunk 做向量检索（小块 → 精确匹配）
  2. 命中 Child Chunk 后，返回其 Parent Chunk（大块 → 完整上下文）
  3. Parent Chunk 作为 LLM 的上下文
```

**为什么有效**：小块检索精度高（信噪比好），大块返回保证了生成所需的完整上下文。

#### Sentence Window Retrieval（句子窗口检索）

与 Parent-Child 类似，但更细粒度：

```
索引单位：单个句子
检索命中："RAG 系统的核心挑战是检索质量。"
返回窗口：前后各扩展 3-5 个句子

"... 在生产环境中，RAG 系统面临多种挑战。RAG 系统的核心挑战是检索质量。
如果检索到的文档不相关，再强的 LLM 也无法生成正确答案。常见的优化方法包括..."
```

#### Recursive Retrieval（递归检索）

针对需要多跳推理的复杂问题：

```
问题："DeepSeek-V3 使用了什么注意力机制？这种机制相比标准 MHA 有什么优势？"

第一轮检索：找到 "DeepSeek-V3 使用了 MLA (Multi-head Latent Attention)"
  ↓ 提取到新信息 "MLA"
第二轮检索：用 "MLA Multi-head Latent Attention" 检索 → 找到 MLA 的详细原理
  ↓ 提取到对比信息
第三轮检索：用 "MLA vs MHA 对比" 检索 → 找到两者的对比分析
  ↓ 综合三轮检索结果生成最终答案
```

### 5.3 Post-Retrieval：检索后处理 🔴

#### Reranking（重排序）

检索返回的 Top-K 结果中，排序可能不够精确。重排序用更强的模型重新打分：

```
向量检索 Top-20 → Cross-Encoder 重排序 → 取 Top-5 输入 LLM

Cross-Encoder 的打分方式：
  输入：[query] [SEP] [document]
  输出：相关性分数 (0-1)
  
比 Bi-Encoder 精度高 10-15%，但速度慢得多（所以只重排 Top-20 而非全量）
```

**主流 Reranker 对比**：

| Reranker | 类型 | 特点 |
|----------|------|------|
| **BGE-Reranker-v2-m3** | Cross-Encoder | 开源最强之一，多语言，中文效果好 |
| **Cohere Rerank** | API 服务 | 商用，简单好用，效果稳定 |
| **ColBERT** | Late Interaction | 精度接近 Cross-Encoder，但可以预计算 doc 端 |
| **LLM-based Reranker** | LLM 打分 | 用 GPT-4 等直接打分，精度最高但成本高 |
| **FlashRank** | 轻量 Cross-Encoder | 超轻量级，适合延迟敏感场景 |

#### Context Compression（上下文压缩）

检索到的 chunks 可能包含大量与问题无关的内容。压缩可以减少噪声：

```python
# 方案 1：LLM 提取
compress_prompt = """
从以下检索到的文档中，只提取与问题直接相关的信息。

问题：{query}
文档：{retrieved_chunk}
提取的相关信息：
"""

# 方案 2：LLMLingua — 基于 PPL 的 token 级别压缩
# 移除对回答贡献最小的 token，将 2000 token 的 chunk 压缩到 500 token
```

#### Lost-in-the-Middle 效应处理 🟡

研究发现（Stanford, 2023）：LLM 在处理长上下文时，最容易忽略**中间位置**的信息，对开头和结尾的信息利用率最高。

```
上下文中的位置：  [开头]  [中间]  [结尾]
信息利用率：       高      低      高

优化策略：将最相关的 chunk 放在开头和结尾，次相关的放中间

排列前：[Chunk3(高)] [Chunk1(最高)] [Chunk5(中)] [Chunk2(高)] [Chunk4(最高)]
排列后：[Chunk1(最高)] [Chunk3(高)] [Chunk5(中)] [Chunk2(高)] [Chunk4(最高)]
```

### 5.4 高级检索模式 🟡

#### Self-RAG（自反思 RAG）

模型自主决定何时检索、评估检索质量、决定是否重新检索：

```
输入问题
  ↓
模型生成 [Retrieve] token → 需要检索
  ↓ 检索文档
模型生成 [IsRel] token → 评估检索是否相关
  ├─ [IsRel=Yes] → 继续生成
  └─ [IsRel=No] → 丢弃，重新检索或直接生成
  ↓
模型生成 [IsSup] token → 评估答案是否被文档支持
  ├─ [IsSup=Yes] → 答案可信
  └─ [IsSup=No] → 需要修正
```

#### CRAG（Corrective RAG）

```
检索文档 → 评估质量
  ├─ Correct（高质量）→ 知识精炼 → 生成
  ├─ Ambiguous（不确定）→ Web 搜索补充 → 生成
  └─ Incorrect（低质量）→ 丢弃，完全依赖 Web 搜索 → 生成
```

#### Adaptive RAG

根据问题的难度动态选择策略：

```
问题难度评估
  ├─ 简单（模型已知）→ 直接生成，不检索
  ├─ 中等 → 单轮 RAG
  └─ 复杂 → 多轮迭代 RAG + 重排序
```

#### Graph RAG（图谱增强 RAG）🟡

微软开源方案的核心架构：

```
离线阶段：
  文档 → LLM 抽取实体和关系 → 构建知识图谱
       → 社区发现算法（Leiden）将图谱分为社区
       → 为每个社区生成摘要

在线阶段：
  全局查询（"总结所有...的趋势"）→ Map-Reduce 遍历社区摘要
  局部查询（"X 和 Y 的关系"）→ 从实体出发，遍历图谱邻域
```

**Graph RAG vs 传统 RAG**：

```
传统 RAG：擅长回答"某个具体事实是什么"（点查询）
Graph RAG：擅长回答"X 和 Y 有什么关系"、"总结某领域的趋势"（关联/全局查询）
```

### 5.5 Agentic RAG 实现模式 🟢

基于 LangGraph 的 Agentic RAG 状态图：

```
                    ┌──────────────┐
                    │  Start Node  │
                    │  分析问题     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
          ┌────────┤  Router Node  ├────────┐
          │        │  路由决策      │        │
          │        └──────┬───────┘        │
     直接回答         需要检索          需要Web搜索
          │              │                  │
   ┌──────▼──────┐ ┌────▼────┐      ┌─────▼─────┐
   │  Generate   │ │Retrieve │      │ Web Search│
   │  直接生成   │ │ 向量检索 │      │ 网络搜索  │
   └──────┬──────┘ └────┬────┘      └─────┬─────┘
          │              │                  │
          │       ┌──────▼───────┐         │
          │       │   Evaluate   │         │
          │       │  评估检索质量 │         │
          │       └──┬──────┬───┘         │
          │      合格│      │不合格        │
          │          │  ┌───▼────┐        │
          │          │  │Rewrite │        │
          │          │  │重写查询 ├───→ 回到 Retrieve
          │          │  └────────┘
          │   ┌──────▼───────┐
          └──→│   Generate   │←─────────┘
              │  生成最终答案  │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │  Self-Check  │
              │  答案自检     │
              └──────┬───────┘
                     │
                  输出答案
```

---

## 六、生产级 RAG 十大挑战

> 🔴 面试重点区域 — 能回答这些问题说明你有真实的生产经验。

### 挑战 1：内容缺失 — 知识库覆盖不全

**问题表现**：用户问的知识根本不在知识库里，但系统仍然强行从不相关的文档中"挤"出一个答案 → 幻觉。

**解决方案**：

```
1. 检索置信度阈值
   - 设置最小相似度阈值（如 cosine > 0.65）
   - 低于阈值 → 返回"抱歉，未找到相关信息"而非胡说

2. 意图分类前置
   - 用 LLM 先判断问题是否在知识库范围内
   - 范围外 → 引导用户换个问法或升级人工

3. 覆盖率监控
   - 记录所有"未命中"的查询（hit rate <阈值）
   - 定期分析未命中查询 → 补充知识库

4. 多数据源兜底
   - 知识库检索失败 → 降级到 Web 搜索 → 再降级到模型自身知识（明确标注）
```

### 挑战 2：检索质量 — 错过正确文档

**问题表现**：正确文档在知识库里，但检索没有把它排到 Top-K。

**根因分析**：

```
1. Query-Document 语义鸿沟
   用户说"系统变慢了"，文档写的是"性能劣化的诊断方法"→ 向量相似度不够

2. Embedding 模型领域适配不足
   通用 Embedding 模型对专业术语的编码效果差

3. 分块策略不当
   关键信息被切分到两个 chunk 中，单独一个 chunk 的语义不完整

4. 数据噪声
   知识库中有大量低质量、过时、重复的文档
```

**解决方案组合**：

```python
# 检索质量优化的"组合拳"
pipeline = (
    QueryRewrite()           # 1. 查询改写 → 缩小语义鸿沟
    | MultiQuery(n=3)        # 2. 多角度查询 → 提高召回
    | HybridSearch(           # 3. 混合检索 → 关键词+语义互补
        dense_weight=0.7,
        sparse_weight=0.3
    )
    | Reranker(               # 4. 重排序 → 提升 Top-K 精度
        model="bge-reranker-v2-m3",
        top_k=5
    )
)
```

### 挑战 3：上下文窗口溢出 — 检索到了但放不下

**问题表现**：需要参考 20 个 chunk 才能完整回答，但 LLM 上下文窗口只够放 5 个。

**解决方案**：

```
1. 上下文压缩
   - 用 LLM 对每个 chunk 做信息提取，只保留与 query 相关的部分
   - 2000 token 的 chunk → 压缩到 200-500 token

2. Map-Reduce 策略
   - Map：对每个 chunk 独立生成子答案
   - Reduce：将所有子答案合并为最终答案
   - 适合总结类问题

3. Refine 策略
   - 逐个 chunk 迭代精化答案
   - 第1个 chunk → 初始答案 → 第2个 chunk → 精化答案 → ...

4. 分块策略优化
   - 使用更小的 chunk_size + Parent-Child 策略
   - 只有最相关的小块被选中，返回其 parent 保证上下文
```

### 挑战 4：未能利用上下文 — 检索到了但模型忽略了

**问题表现**：相关文档确实在上下文中，但 LLM 没有基于它生成答案（尤其是答案在上下文中间位置时）。

**解决方案**：

```
1. 位置优化（对抗 Lost-in-the-Middle）
   - 最相关的 chunk 放开头和结尾
   
2. Prompt 强化指令
   - "请严格基于以下参考文档回答，如果文档中有答案，必须引用"
   - "如果参考文档中没有相关信息，请明确说明"

3. 引用强制
   - 要求模型在回答中标注引用来源 [1][2]
   - 后处理验证引用是否存在于上下文中

4. 上下文长度控制
   - 减少输入的 chunk 数量，提高信噪比
   - 宁可输入 3 个高质量 chunk，不要输入 10 个含大量噪声的 chunk
```

### 挑战 5：格式错误 — 输出格式不符合要求

**解决方案**：

```python
# 1. System Prompt 明确格式要求
system_prompt = """
请以以下 JSON 格式输出：
{
  "answer": "...",
  "sources": [{"doc": "...", "page": ...}],
  "confidence": 0.0-1.0
}
"""

# 2. 结构化输出（JSON Mode / Function Calling）
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[...]
)

# 3. 后处理修复
def fix_format(response):
    try:
        return json.loads(response)
    except:
        # 用 LLM 修复格式
        return llm_fix_json(response)
```

### 挑战 6：回答的具体性不当

**问题**：回答过于笼统（"可以通过多种方式优化"）或过于具体（用户问概述但得到了细节）。

**解决方案**：

```
1. 问题意图识别
   - 概览类问题 → 控制回答的详细程度
   - 具体问题 → 给出精确答案

2. Prompt 中的具体性指令
   - "请直接给出具体的配置参数和命令"
   - "请先给出总结，再展开细节"

3. Few-shot 示例
   - 在 System Prompt 中给出期望的回答示例
```

### 挑战 7：回答不完整

**问题**：多要点问题只回答了部分。如"RAG 的检索、分块和评估有哪些最佳实践？"只回答了检索。

**解决方案**：

```
1. 查询分解
   复杂问题 → LLM 拆解为子问题 → 各自检索 → 合并答案
   
2. Checklist Prompt
   "请确保你的回答覆盖了以下所有方面：1) 检索 2) 分块 3) 评估"

3. 自检机制
   生成答案后 → LLM 自检"是否完整回答了所有部分" → 补充遗漏
```

### 挑战 8：数据摄入可扩展性

**问题**：10 万+文档的索引需要几天，影响数据新鲜度。

**解决方案**：

```
1. 异步流水线
   - 用 Celery/RQ 做异步任务队列
   - 文档解析、分块、Embedding 分别并行

2. 批量 Embedding
   - 使用 GPU 批量推理而非逐条调用 API
   - OpenAI Batch API 成本降低 50%

3. 增量索引
   - 用文档 hash 判断是否已索引
   - 只处理新增/修改的文档
   - CDC (Change Data Capture) 监听数据源变化

4. 索引预热
   - 新索引构建完成后，先在影子环境验证质量
   - 质量达标后原子切换（蓝绿部署）
```

### 挑战 9：结构化数据处理

**问题**：表格、数据库不适合传统的"分块 → 向量化"流程。

**解决方案**：

```
1. Text-to-SQL
   用户自然语言 → LLM 生成 SQL → 执行查询 → 将结果作为上下文

2. 表格序列化
   - Markdown 表格格式：保留结构信息
   - 逐行描述：将每行数据转为自然语言句子

3. 混合架构
   结构化数据 → SQL 查询
   非结构化数据 → 向量检索
   → Router 根据问题类型选择路径
```

### 挑战 10：多数据源融合

**问题**：不同数据源的信息可能矛盾（内部文档 vs 公开文档 vs API 数据）。

**解决方案**：

```
1. 数据源优先级
   内部权威文档 > 近期更新的文档 > 公开文档 > 模型自身知识

2. 时间戳排序
   多个 chunk 回答同一问题时，优先使用最新的

3. 矛盾检测
   LLM 检测检索到的多个 chunk 是否有矛盾
   有矛盾 → 按优先级取舍 或 如实告知用户存在不一致

4. 元数据标注
   每个回答标注数据来源和时间，让用户自行判断
```

---

## 七、RAG 评估

### 7.1 评估维度 🔴

RAG 系统的评估需要覆盖检索和生成两个阶段：

```
┌──────────────────────────────────────────────┐
│                 RAG 评估维度                   │
├──────────────────┬───────────────────────────┤
│    检索阶段       │       生成阶段              │
├──────────────────┼───────────────────────────┤
│ Context Recall   │ Faithfulness (忠实度)      │
│ Context Precision│ Answer Relevancy (相关性)  │
│ Context Relevancy│ Answer Correctness (正确性)│
│ Hit Rate         │ Answer Completeness (完整性)│
│ MRR              │ Harmlessness (无害性)      │
│ NDCG@K           │                           │
└──────────────────┴───────────────────────────┘
```

#### 关键指标详解

**1. Faithfulness（忠实度）**— 最核心的 RAG 指标

```
定义：生成的答案是否忠实于检索到的上下文？（不编造信息）

计算方法（RAGAS）：
  1. 将答案拆分为多个独立的声明（claims）
  2. 检验每个声明是否能从上下文中找到支持
  3. Faithfulness = 有支持的声明数 / 总声明数

示例：
  上下文："RAG 系统由检索和生成两部分组成，检索使用向量相似度。"
  答案："RAG 由检索和生成组成，检索基于向量相似度，生成使用 GPT-4。"
  
  声明1: "RAG 由检索和生成组成" → ✅ 有支持
  声明2: "检索基于向量相似度" → ✅ 有支持
  声明3: "生成使用 GPT-4" → ❌ 上下文未提及
  
  Faithfulness = 2/3 = 0.67
```

**2. Context Recall（上下文召回率）**

```
定义：标准答案中的关键信息有多少被检索到的上下文覆盖？

计算方法：
  1. 将标准答案拆分为独立声明
  2. 检验每个声明是否被检索到的上下文覆盖
  3. Context Recall = 被覆盖的声明数 / 标准答案总声明数

高 Recall → 检索到了正确信息
低 Recall → 检索遗漏了关键文档
```

**3. Context Precision（上下文精确度）**

```
定义：检索到的上下文中，有多少与问题真正相关？

高 Precision → 检索结果质量高，噪声少
低 Precision → 检索到了很多无关内容（影响生成质量和 token 成本）
```

**4. Answer Relevancy（答案相关性）**

```
定义：答案是否真正回答了用户的问题？

计算方法（RAGAS）：
  1. 从答案中用 LLM 反向生成 N 个问题
  2. 计算生成的问题与原始问题的 Embedding 相似度
  3. Relevancy = 平均相似度

高 Relevancy → 答案紧扣问题
低 Relevancy → 答案跑题了
```

### 7.2 评估框架 🟡

#### RAGAS

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 准备评估数据
eval_data = {
    "question": ["什么是 RAG？", ...],
    "answer": ["RAG 是一种将检索与生成结合的技术...", ...],
    "contexts": [["RAG（Retrieval-Augmented Generation）...", ...], ...],
    "ground_truth": ["RAG 是检索增强生成的缩写...", ...]
}

dataset = Dataset.from_dict(eval_data)

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.90, 
#  'context_precision': 0.75, 'context_recall': 0.80}
```

**RAGAS 的优势**：
- LLM-based 评估，无需大量人工标注
- 覆盖检索和生成全链路
- 每个指标都有直觉的含义

**RAGAS 的局限**：
- 依赖 Judge LLM 的能力（用弱模型评估可能不准）
- 评估本身有成本（每次评估都要调用 LLM）
- 评估结果有随机性（需要多次运行取平均）

#### TruLens

```
与 RAGAS 类似，但更侧重：
  1. 回答忠实度（Groundedness）
  2. 答案相关性（Answer Relevance）  
  3. 上下文相关性（Context Relevance）

优势：与 LangChain/LlamaIndex 深度集成，开箱即用的 Dashboard
```

#### 对比

| 维度 | RAGAS | TruLens |
|------|-------|---------|
| 指标丰富度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 开箱即用 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 可视化 | ⚠️ 需自建 | ✅ 内置 Dashboard |
| 定制性 | ✅ 可自定义指标 | ⚠️ 定制空间较小 |
| 社区活跃度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 7.3 评估数据集构建 🟡

#### 方法 1：人工标注（金标准）

```
流程：
  1. 从知识库中抽样文档
  2. 领域专家针对文档编写问题
  3. 标注每个问题的：
     - 标准答案 (ground truth)
     - 相关文档列表 (relevant chunks)
  4. 交叉验证（多人标注同一问题，计算一致性 κ>0.7）

规模：50-200 个问答对即可做有效评估
成本：高（需要领域专家时间）
质量：最高
```

#### 方法 2：LLM 辅助生成

```python
generate_prompt = """
你是一个 QA 评测数据集生成器。
请根据以下文档内容，生成 3 个问答对。

要求：
1. 问题应该是用户真实会问的（自然语言，非学术化）
2. 答案必须能从文档中找到依据
3. 包含不同难度：简单事实型 / 推理型 / 对比型

文档内容：
{document_chunk}

请以 JSON 格式输出：
[
  {
    "question": "...",
    "answer": "...",
    "relevant_chunk_ids": ["..."],
    "difficulty": "easy/medium/hard"
  }
]
"""
```

**质量控制**：LLM 生成的数据需要人工审核，通常有 20-30% 需要修正。

#### 方法 3：生产数据回收

```
从线上真实查询中挖掘评测数据：
  1. 收集高频用户查询
  2. 收集用户反馈标记的 Bad Case
  3. 人工审核并添加标准答案
  4. 转化为回归测试用例

优势：最接近真实场景
劣势：需要产品已上线才有数据
```

### 7.4 端到端 vs 模块级评估 🟡

```
端到端评估（End-to-End）：
  问题 → [RAG 系统黑盒] → 答案
  评估指标：Faithfulness, Answer Relevancy, Correctness
  适用：验证系统整体效果，定期回归测试

模块级评估（Component-level）：
  ┌────────────────┐
  │ 检索模块评估     │ Recall@K, Precision@K, MRR, NDCG
  ├────────────────┤
  │ 重排序模块评估   │ MAP, NDCG (重排前后对比)
  ├────────────────┤
  │ 生成模块评估     │ Faithfulness, Relevancy (固定上下文)
  └────────────────┘
  适用：定位问题环节，优化特定模块

建议：
  - 日常优化 → 模块级评估（定位瓶颈）
  - 发布前 → 端到端评估（确保整体不退化）
  - 上线后 → 用户反馈 + 自动化端到端回归
```

### 7.5 生产评估最佳实践

**三级评测体系**：

```
Level 1: 离线评估（每次变更触发）
  - 自动化 CI/CD 集成
  - RAGAS 评测套件
  - 100+ 问答对覆盖核心场景
  - 指标阈值：Faithfulness > 0.85, Context Recall > 0.80

Level 2: 在线评估（实时收集）
  - 用户 👍👎 反馈
  - 响应延迟 P50/P95/P99
  - 检索命中率 (hit rate)
  - LLM 审核采样（抽检 5% 的回答用 GPT-4 打分）

Level 3: 回归评估（定期执行）
  - 所有历史 Bad Case 作为回归用例
  - 新增场景自动扩充评测集
  - 月度人工深度审查
```

**Bad Case 闭环**：

```
发现 Bad Case
  ↓
根因分析：是检索问题还是生成问题？
  ├─ 检索问题 → 文档缺失？分块不当？Embedding 不够好？
  └─ 生成问题 → Prompt 不够好？上下文太长/太短？模型幻觉？
  ↓
修复
  ↓
转化为评测用例（加入回归测试集）
  ↓
回归验证（确认修复生效且未引入新问题）
```

---

## 附录：面试高频考点速查

### 🔴 高频（必须掌握）

| # | 考点 | 核心要点 |
|---|------|---------|
| 1 | RAG vs Fine-Tuning vs Long Context | 三者的适用场景决策树，能说清楚 trade-off |
| 2 | RAG 架构演进 | Naive → Advanced → Modular → Agentic 每代的改进 |
| 3 | 分块策略 | 6 种分块方法的优缺点，chunk_size 如何调优 |
| 4 | 混合检索 | Dense + Sparse + RRF 的完整流程 |
| 5 | 重排序 | Bi-Encoder vs Cross-Encoder vs Late Interaction 的区别 |
| 6 | HNSW 原理 | 多层图结构、查询流程、关键参数 |
| 7 | Faithfulness 评估 | RAGAS 的评估方法论 |
| 8 | Parent-Child 检索 | 小块检索大块返回的原理和实现 |
| 9 | 生产级挑战 | 至少能说出 5 个挑战及解决方案 |
| 10 | 向量数据库选型 | Milvus/Qdrant/pgvector 的选型依据 |

### 🟡 中频（加分项）

| # | 考点 | 核心要点 |
|---|------|---------|
| 1 | HyDE | 假设文档嵌入的原理和适用场景 |
| 2 | Query Rewriting/Routing | 查询预处理的方法论 |
| 3 | Self-RAG / CRAG | 自适应检索的状态机设计 |
| 4 | Graph RAG | 微软方案的核心架构和适用场景 |
| 5 | Embedding 微调 | 对比学习、硬负样本挖掘 |
| 6 | Lost-in-the-Middle | 上下文位置对 LLM 注意力的影响 |
| 7 | 评估数据集构建 | 人工 + LLM 辅助 + 生产回收 |
| 8 | 多租户权限 | 检索时的权限过滤方案 |

### 🟢 加分项（展示深度）

| # | 考点 | 核心要点 |
|---|------|---------|
| 1 | Late Chunking | 延迟分块保留全文档上下文 |
| 2 | ColBERT/ColPali | Late Interaction 范式 |
| 3 | Matryoshka Embedding | 套娃嵌入支持维度截断 |
| 4 | DiskANN | 磁盘索引的原理和适用场景 |
| 5 | Agentic RAG | LangGraph 实现的状态图 |
| 6 | 语义缓存 | 用向量相似度做查询缓存 |

---

## 参考资源

### 经典论文
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — RAG 原始论文 (2020)
- [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) — Self-RAG (2023)
- [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884) — CRAG (2024)
- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) — Microsoft Graph RAG (2024)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 评估框架 (2023)
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Word2Vec (2013)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — 长上下文中间信息丢失问题 (2023)

### 开源项目
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- [LlamaIndex](https://docs.llamaindex.ai/) — 专注 RAG 的框架
- [Milvus](https://milvus.io/docs) — 分布式向量数据库
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — 开源最强中文 Embedding
- [RAGAS](https://docs.ragas.io/) — RAG 评估框架
- [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard) — Embedding 模型评测

### 学习课程
- [DeepLearning.AI - Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/)
- [LangChain Academy - Introduction to LangGraph](https://academy.langchain.com/)
