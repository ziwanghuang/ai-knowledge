# 🌐 AI 搜索与信息获取 — AI Search

## 概述

AI 搜索是将大语言模型与传统搜索引擎深度融合的新范式，代表了从"给链接"到"给答案"的信息获取方式变革。以 Perplexity、SearchGPT 为代表的 AI 搜索产品正在重塑用户获取信息的方式，同时 Web Browsing Agent 也成为 Agent 系统获取实时信息的关键能力。

## 核心知识体系

### 1. AI 搜索产品范式

| 产品 | 特点 |
|------|------|
| **Perplexity** | AI 原生搜索引擎，搜索 + RAG + 生成融合，带引用来源 |
| **SearchGPT / ChatGPT Search** | OpenAI 搜索产品，对话式搜索体验 |
| **Google AI Overviews** | Google 搜索结果中的 AI 摘要 |
| **Bing Chat / Copilot** | 微软 AI 搜索，集成 GPT-4 |
| **秘塔 AI 搜索** | 国内 AI 搜索产品，中文效果好 |
| **Kimi 搜索** | 月之暗面，支持长文档搜索 |

### 2. 搜索 + RAG + 生成 融合架构

```
用户查询 → 查询理解与改写 → 多源搜索（Web/知识库/数据库）
    → 结果聚合与去重 → 重排序与过滤 → 上下文构建
    → LLM 生成（带引用标注）→ 结构化回答
```

- **查询理解**：意图识别、查询分类（事实型/探索型/导航型）
- **查询改写与扩展**：同义词扩展、子查询分解、多语言翻译
- **多源检索融合**：Web 搜索 API + 本地知识库 + 数据库，RRF 排序融合
- **引用与溯源**：生成内容中标注信息来源，支持用户验证

### 3. Web Browsing Agent ⭐

- **网页浏览能力**：
  - 搜索引擎 API 调用（Google、Bing、Brave Search）
  - 网页内容提取（Jina Reader、Trafilatura、Firecrawl）
  - 动态页面渲染（Playwright、Selenium、Puppeteer）
  - JavaScript 执行与交互
- **浏览器操作 Agent**：
  - 表单填写、按钮点击、页面导航
  - 截图理解 + 操作决策
  - 代表项目：WebArena、WebVoyager、BrowserGym
- **反爬虫与合规**：
  - robots.txt 遵守
  - 请求频率控制
  - 用户代理伪装与 IP 轮换

### 4. 实时信息获取

- **搜索 API 集成**：
  | API | 特点 |
  |-----|------|
  | Google Custom Search | 最全面，需付费 |
  | Bing Search API | 微软，性价比高 |
  | Brave Search API | 隐私友好，免费额度 |
  | SerpAPI | 聚合多个搜索引擎 |
  | Tavily | 专为 AI Agent 设计的搜索 API |
- **RSS / 新闻聚合**：实时追踪信息源
- **社交媒体监控**：Twitter/X API、Reddit API
- **数据流订阅**：WebSocket 实时数据推送

### 5. 搜索质量评估

- **搜索相关性指标**：NDCG、MRR、Precision@K
- **回答质量指标**：准确性、完整性、时效性、引用准确率
- **用户体验指标**：响应时间、交互满意度

## 学习路线建议

1. 了解 AI 搜索产品的设计理念与用户体验
2. 学习搜索 API 的集成与使用（推荐 Tavily / Brave Search）
3. 实践 Web Browsing Agent（Playwright + LLM）
4. 构建搜索 + RAG 融合的信息获取系统
5. 优化搜索质量与引用准确性

## 推荐资源

- 📘 [Perplexity](https://www.perplexity.ai/) — AI 搜索标杆产品
- 📘 [Tavily API](https://tavily.com/) — AI Agent 专用搜索 API
- 📘 [Firecrawl](https://github.com/mendableai/firecrawl) — 网页抓取与内容提取
- 📘 [Playwright](https://playwright.dev/) — 浏览器自动化
- 📘 [WebArena](https://webarena.dev/) — Web Agent 评估基准
- 📄 [WebGPT: Browser-assisted question-answering](https://arxiv.org/abs/2112.09332)
