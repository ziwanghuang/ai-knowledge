# ai-knowledge

系统化整理 AI 领域的学习笔记、论文阅读、技术实践与最佳实践。

## 涵盖方向

- 📐 基础理论 — ML / DL / 强化学习 / 概率统计 / 认知科学
- 🧠 LLM — 大语言模型原理、微调、部署
- 🤖 AI Agent — LangGraph / LangChain / Multi-Agent 架构
- 🔍 RAG — 检索增强生成、向量数据库、Embedding 策略
- 🔗 知识图谱 — 图数据库、知识表示与推理、Graph RAG
- 🎭 多模态 — VLM、语音技术、视频理解、图像生成
- 🔌 MCP — Model Context Protocol 接入与工具集成
- ✍️ Prompt Engineering — 提示词设计模式与优化技巧
- 🛠️ 工程实践 — AI 应用落地中的架构、性能、可观测性
- 🌐 AI 搜索 — Perplexity 式搜索、Web Browsing Agent、实时信息获取
- 📊 数据工程 — 数据采集、标注、合成数据、数据飞轮
- 🧪 AI 测试 — LLM 测试、Prompt 回归、Red Teaming、Agent 端到端测试
- 🔐 AI 伦理与治理 — 偏见公平性、可解释性、法规合规、负责任 AI
- 💰 AI 产品与商业化 — 产品设计、定价策略、增长模型、竞品分析
- 🔬 前沿研究 — 自主 Agent、Agent OS、具身智能、科学发现
- 🏭 行业应用 — 金融、医疗、法律、教育等垂直领域落地

## 目录结构

```
docs/
├── fundamentals/           # 📐 基础理论
├── llm/                    # 🧠 大语言模型
├── ai-agent/               # 🤖 AI Agent 智能体架构
├── rag/                    # 🔍 检索增强生成
├── knowledge-graph/        # 🔗 知识图谱
├── multimodal/             # 🎭 多模态技术
├── mcp/                    # 🔌 Model Context Protocol
├── prompt-engineering/     # ✍️ 提示词工程
├── engineering-practice/   # 🛠️ 工程实践
├── ai-search/              # 🌐 AI 搜索与信息获取
├── data-engineering/       # 📊 数据工程与数据飞轮
├── ai-testing/             # 🧪 AI 测试与质量保障
├── ai-ethics/              # 🔐 AI 伦理与治理
├── ai-product/             # 💰 AI 产品与商业化
├── frontier-research/      # 🔬 前沿研究
└── industry-applications/  # 🏭 行业应用
```


## 🚢 部署与同步

### 推送到远端服务器

使用 rsync 将项目推送到远端服务器，自动排除不需要同步的文件：

```bash
rsync -avz \
    --exclude='.git/' \
    --exclude='.github/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='venv' \
    /Users/ziwh666/GitHub/ai-knowledge \
    root@182.43.22.165:/data/github/
```

### 从远端拉取最新代码

```bash
git fetch origin && git reset --hard origin/main
```

