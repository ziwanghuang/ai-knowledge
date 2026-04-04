# 🔗 知识图谱 — Knowledge Graph

## 概述

知识图谱（Knowledge Graph）是一种以图结构组织和表示知识的技术，通过实体、关系和属性构建结构化的知识网络。在 AI Agent 系统中，知识图谱是实现精确推理、减少幻觉、支撑 Graph RAG 的核心基础设施。

## 核心知识体系

### 1. 图数据库

| 数据库 | 特点 |
|--------|------|
| **Neo4j** | 最流行的原生图数据库，Cypher 查询语言，社区生态完善 |
| **ArangoDB** | 多模型数据库（图 + 文档 + KV），AQL 查询语言 |
| **Amazon Neptune** | AWS 托管图数据库，支持 Gremlin 和 SPARQL |
| **TigerGraph** | 高性能分布式图数据库，适合大规模图分析 |

### 2. 知识表示与推理

- **三元组（Subject-Predicate-Object）**：知识的基本单元
- **RDF / OWL**：语义网标准，适合严格本体建模
- **Property Graph**：属性图模型，节点和边都可以有属性，更灵活
- **知识推理**：
  - 规则推理（SWRL）
  - 基于嵌入的推理（TransE、RotatE、ComplEx）
  - 路径推理

### 3. 知识图谱构建

- **实体识别（NER）**：识别文本中的人名、地名、机构等
- **关系抽取（RE）**：提取实体间的关系
- **事件抽取**：识别事件及其参与者和属性
- **用 LLM 自动构建知识图谱**：Prompt 驱动的三元组抽取 ⭐
- **知识融合与对齐**：多源知识去重、实体对齐

### 4. 图神经网络（GNN）

- **GCN（Graph Convolutional Network）**：图卷积网络
- **GAT（Graph Attention Network）**：图注意力网络
- **GraphSAGE**：归纳式学习，处理动态图
- **应用**：节点分类、链接预测、图分类

### 5. 知识图谱 + LLM 结合 ⭐

- **Graph RAG**：用知识图谱结构化信息增强 LLM 检索（微软开源方案）
- **KG-enhanced LLM**：将知识图谱嵌入 LLM 上下文减少幻觉
- **LLM-driven KG Construction**：用 LLM 自动从非结构化文本构建知识图谱
- **Text-to-Cypher / Text-to-SPARQL**：自然语言转图查询语言

## 学习路线建议

1. 学习图数据库基础，掌握 Neo4j 和 Cypher 查询
2. 理解知识表示方法（三元组、属性图）
3. 实践用 LLM 自动构建知识图谱
4. 学习 Graph RAG 技术
5. 了解 GNN 基础与应用

## 深度学习文档

- 📖 [知识图谱完整技术体系深度解析](知识图谱完整技术体系深度解析.md) — 覆盖基础理论、图数据库、知识表示学习、KG构建、GNN、KG+LLM融合

## 推荐资源

- 📘 [Neo4j 官方文档](https://neo4j.com/docs/)
- 📘 [Microsoft Graph RAG](https://github.com/microsoft/graphrag)
- 📄 [Knowledge Graphs](https://arxiv.org/abs/2003.02320) — 综述论文
- 🎓 [Stanford CS224W - Machine Learning with Graphs](http://web.stanford.edu/class/cs224w/)
- 📖 [《知识图谱：方法、实践与应用》](https://book.douban.com/subject/34889060/)
