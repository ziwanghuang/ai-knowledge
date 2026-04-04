# 🔍 RAG — 检索增强生成

## 概述

RAG（Retrieval-Augmented Generation，检索增强生成）是一种将外部知识检索与大语言模型生成能力相结合的技术范式。通过在生成前检索相关文档，RAG 有效缓解了 LLM 的幻觉问题，并使模型能够利用最新的、领域特定的知识。

## 核心知识体系

### 1. RAG 基础架构

```
用户查询 → 查询处理 → 检索器(Retriever) → 重排序(Reranker) → 上下文构建 → LLM 生成 → 响应
                              ↑
                        向量数据库 / 知识库
```

- **Indexing（索引）**：文档加载 → 分块 → Embedding → 存储
- **Retrieval（检索）**：查询向量化 → 相似度搜索 → 结果过滤
- **Generation（生成）**：上下文拼接 → Prompt 构建 → LLM 生成

### 2. 文档解析与预处理 ⭐

#### 2.1 各类文档解析
| 文档类型 | 工具 | 难点 |
|---------|------|------|
| PDF | PyPDF2、pdfplumber、Unstructured、Marker、MinerU | 扫描件需 OCR、复杂排版、表格提取、公式识别 |
| HTML/网页 | BeautifulSoup、Trafilatura、Jina Reader | 动态渲染（需 Playwright/Selenium）、广告噪音过滤 |
| 表格数据 | Camelot、Tabula、pandas | 跨行跨列合并、嵌套表格 |
| 图片/OCR | Tesseract、PaddleOCR、EasyOCR、GPT-4V | 扫描文档、截图文字、图表理解 |
| Office | python-docx、openpyxl、python-pptx | 格式多样、嵌入对象 |

#### 2.2 文本分块策略详解
- **Fixed-size Chunking**：按固定字符/token 数切分，简单但可能破坏语义
- **Recursive Character Splitting**：按段落 → 句子 → 字符层级递归切分，LangChain 默认方案
- **Semantic Chunking**：基于 Embedding 相似度判断语义边界，效果更好但计算量大
- **Document-based Chunking**：利用文档本身的结构（标题、章节）切分
- **Agentic Chunking**：用 LLM 判断最佳切分点
- **关键参数**：chunk_size（块大小）、chunk_overlap（重叠区域），需根据场景调参

### 3. Embedding 模型深入 ⭐

#### 3.1 主流文本 Embedding 模型
| 模型 | 维度 | 特点 |
|------|------|------|
| OpenAI text-embedding-3 | 256-3072 | 商用，效果好 |
| BGE 系列（BAAI） | 768-1024 | 开源，中文效果最好之一 |
| M3E | 768 | 中文开源，轻量高效 |
| Jina Embeddings | 768 | 开源，支持超长上下文（8192 token） |
| Cohere Embed v3 | 1024 | 商用，多语言，内置压缩 |
| E5 系列（Microsoft） | 768-1024 | 指令式 Embedding，可通过 prompt 控制 |
| GTE 系列（Alibaba） | 768-1024 | 通用文本 Embedding，中英文效果好 |
| Sentence-BERT | 768 | 经典开源模型，学术研究常用 |

#### 3.2 Embedding 选型考量
- **维度大小**：越高精度越好，但存储和计算成本越大（常见 384/768/1024/1536/3072）
- **最大输入长度**：短文本（512 token）vs 长文本（8192+ token）
- **多语言支持**：是否支持中文、跨语言检索
- **Matryoshka（套娃）Embedding**：支持截断维度而保持性能
- **评估基准**：MTEB（Massive Text Embedding Benchmark）排行榜

#### 3.3 多模态 Embedding
- **CLIP**（OpenAI）：文本 + 图像联合嵌入
- **ImageBind**（Meta）：六种模态统一嵌入
- **CLAP**：文本 + 音频嵌入

#### 3.4 微调 Embedding 模型
- 对比学习（Contrastive Learning）：正负样本对
- 硬负采样（Hard Negative Mining）
- 蒸馏（Distillation）：大模型蒸馏到小模型

### 4. 向量数据库深入 ⭐

#### 4.1 主流向量数据库对比
| 数据库 | 特点 |
|--------|------|
| **Milvus / Zilliz** | 开源分布式，支持万亿级向量，适合大规模生产 |
| **Pinecone** | 全托管云服务，零运维，适合快速上手 |
| **Weaviate** | 支持混合搜索（向量 + 关键词），内置模块化 Embedding |
| **Chroma** | 轻量级、嵌入式，适合本地开发和原型验证 |
| **Qdrant** | Rust 编写，高性能，支持丰富的过滤条件 |
| **FAISS** | Meta 开源向量检索库（非数据库），适合研究和嵌入式场景 |
| **pgvector** | PostgreSQL 插件，在已有 PG 基础上增加向量能力 |

#### 4.2 索引算法
- **HNSW**（Hierarchical Navigable Small World）：多层图结构，查询速度快，内存占用较高
- **IVF**（Inverted File Index）：聚类 + 倒排索引，适合大数据量
- **PQ**（Product Quantization）：向量压缩量化，降低存储和计算成本
- **ScaNN**（Google）：各指标均衡，适合大规模检索
- **DiskANN**：支持磁盘存储的 ANN 算法，超大数据量下表现优异

#### 4.3 工程实践要点
- **元数据过滤（Metadata Filtering）**：结合标签、时间、类别等结构化条件缩小搜索范围
- **混合检索（Hybrid Search）**：Dense 向量检索 + Sparse 关键词检索（BM25）联合排序
- **多租户隔离**：按用户/组织隔离数据（Namespace / Collection / Partition）
- **一致性与持久化**：WAL、快照、副本
- **性能调优**：索引参数调整（ef_construction、M、nprobe）、批量写入、预热
- **数据生命周期管理**：TTL、归档策略、增量更新 vs 全量重建

#### 4.4 在 Agent 中的角色
- 长期记忆存储：对话历史、用户偏好、学到的知识
- RAG 的检索引擎：从知识库中召回相关上下文
- 工具/API 语义路由：从大量可用工具中选出最相关的
- 去重与相似检测：避免重复内容

### 5. 检索策略

- **稀疏检索（Sparse）**：BM25、TF-IDF
- **密集检索（Dense）**：基于 Embedding 的语义检索
- **混合检索（Hybrid Search）**：结合稀疏与密集检索的优势
- **重排序（Reranking）**：Cross-Encoder、Cohere Rerank、ColBERT（Late Interaction）、BGE Reranker

### 6. 高级 RAG 技术

- **Query Transformation**：查询改写、HyDE、多查询扩展
- **Query Decomposition**：将复杂问题拆分为多个子查询
- **Multi-Query RAG**：为同一问题生成多种查询变体，扩大召回覆盖
- **Self-Query RAG**：LLM 自动从自然语言中提取结构化过滤条件
- **RAG-Fusion**：倒数排序融合（RRF），合并多路检索结果
- **Self-RAG**：模型自主决定是否需要检索
- **Corrective RAG（CRAG）**：检索后自动评估文档质量，必要时重新检索或回退
- **Adaptive RAG**：根据问题难度动态选择检索策略
- **Graph RAG**：基于知识图谱的检索增强（微软开源方案）
- **Agentic RAG**：Agent 驱动的自适应检索策略
- **多模态 RAG**：图片/表格/视频的 RAG 处理
- **上下文压缩（Context Compression）**：对检索结果进行摘要/压缩，减少无关噪音
- **递归检索**：多轮迭代检索，逐步深入

### 7. RAG 系统工程化 ⭐

- **API 接口设计**：FastAPI/Flask 后端、RESTful API、WebSocket、SSE 流式输出
- **性能优化**：推理速度优化、高并发支持、Redis 缓存策略
- **稳定性保障**：错误处理与降级、重试机制、熔断器模式（Circuit Breaker）
- **资源成本优化**：Token 计数与限制、API 调用优化、成本监控
- **系统监控**：日志系统（ELK/Loki）、监控（Prometheus + Grafana）、告警机制
- **用户反馈闭环**：反馈收集、效果评估、A/B 测试、持续优化迭代
- **外部数据源集成**：数据库连接、API 接口集成、实时数据同步

### 8. 生产级 RAG 十大难题 ⭐⭐

> Demo 里的 RAG：100 篇文档 → 向量化 → 检索 → 拼到 Prompt → 搞定。生产级 RAG 面临的是 10 万+ 文档 × 多格式 × 多语言 × 权限控制 × 时效性 × 实时性。

#### 8.1 文档解析的鲁棒性
- 真实世界文档千奇百怪：扫描件 PDF、加密 PDF、表格跨页、图文混排、SPA 页面、嵌套表格
- **解决**：多解析器组合（Marker for PDF、Trafilatura for HTML、PaddleOCR for 扫描件）+ 解析质量检测 + 降级策略

#### 8.2 切片策略的"没有银弹"
- 切太细（200 tokens）→ 上下文断裂；切太粗（2000 tokens）→ 噪声太多
- **解决**：多粒度索引（Parent-Child 策略）——小粒度做检索提高精准度，命中后返回大粒度做生成保证上下文完整

#### 8.3 检索质量：精度 vs 召回的平衡
- 语义检索对专有名词不友好，关键词检索无法理解语义
- **解决**：混合检索 + RRF 融合 + Cross-Encoder 重排 + Query 改写

#### 8.4 数据新鲜度与一致性
- 知识库文档更新了但索引还是旧的，导致回答过时信息
- **解决**：增量索引 + CDC 监听变化 + 双索引原子切换 + TTL 机制 + 元数据时间戳过滤

#### 8.5 多租户权限隔离
- 不同用户能看到不同文档，索引不做权限过滤会导致机密数据泄露
- **解决**：检索时过滤（元数据权限条件）+ 索引隔离（Namespace/Collection）+ pgvector 混合查询（向量检索 + SQL 权限过滤）

#### 8.6 幻觉与证据链
- Agent 回答了问题但答案来源不可追溯
- **解决**：强制引用（输出包含来源片段 ID 和文件名）+ 置信度评分（低于阈值不回答）+ Faithfulness 检测

#### 8.7 多模态检索
- 知识库中有图片、表格、流程图，传统 RAG 无法检索
- **解决**：图片用 VLM 生成描述文本索引 + 多模态 Embedding（CLIP）+ 表格结构化提取

#### 8.8 Embedding 模型更新
- 换更好的 Embedding 模型需要全量重建索引（数十万文档 × GPU 成本）
- **解决**：版本化索引 + 增量迁移（新文档用新模型，旧文档后台异步迁移）+ Matryoshka Embedding

#### 8.9 检索后处理
- 检索到的片段可能有重复、矛盾、Lost-in-the-Middle 效应
- **解决**：去重合并 + 时间排序取最新 + 位置优化（最相关放开头和结尾）+ 上下文压缩

#### 8.10 评测与持续优化
- **三级评测体系**：
  - 离线评测：RAGAS 自动评估忠实度、相关性、上下文精准度
  - 在线评测：用户反馈（👍👎）、Bad Case 收集、查询日志分析
  - 回归评测：每次变更后自动跑回归确保不退化
- **Bad Case 闭环**：发现 → 根因分析 → 修复 → 转化为评测用例 → 回归验证

### 9. 评估体系

- **检索质量**：Recall@K、MRR、NDCG
- **生成质量**：Faithfulness（忠实度）、Relevancy（相关性）、Answer Correctness
- **自动评估指标**：BLEU、ROUGE、Perplexity、BERTScore
- **人工评估标准与流程**
- **领域评估基准**：C-Eval（中文）、CMMLU（中文多任务）
- **评估框架**：RAGAS、TruLens、LangSmith

## 深度学习文档

- 📖 [RAG 完整技术栈深度解析](./RAG完整技术栈深度解析.md) — 系统深入讲解架构设计、文档解析与分块、Embedding 模型、向量数据库、检索策略、生产级十大挑战、评估体系七大板块，含面试重点标注

## 学习路线建议

1. 理解 RAG 的基本流程与核心组件
2. 动手搭建一个简单的 RAG 应用（LangChain + Chroma）
3. 学习文档解析工具与不同的分块策略
4. 深入理解 Embedding 模型选型与向量数据库索引算法
5. 掌握高级 RAG 技术（查询改写、重排序、混合检索、RAG-Fusion）
6. 学习 RAG 系统的工程化与评估优化方法

## 推荐资源

- 📄 [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — RAG 原始论文
- 📄 [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)
- 📘 [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- 📘 [Milvus 文档](https://milvus.io/docs)
- 📘 [MTEB 排行榜](https://huggingface.co/spaces/mteb/leaderboard)
- 🎓 [DeepLearning.AI - Building and Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/)
