# AI 搜索深度指南：从传统检索到生成式搜索的完整技术体系

> 本文面向有 RAG/Agent 基础、了解搜索引擎基本概念但希望系统掌握 AI 搜索技术全貌的开发者。
> 约 14000 字，覆盖架构演进、产品剖析、Web Browsing Agent、Deep Research、查询理解、搜索质量优化、工程实现和前沿方向。

---

## 一、AI 搜索是什么

**一句话**：AI 搜索是将大语言模型与信息检索深度融合的新范式 —— 从"给你十个链接，自己找"变成"直接给你答案，并告诉你答案从哪来"。

### 类比理解

传统搜索引擎像一个**图书管理员**：你说"我要关于量子计算的书"，他把书架指给你，你自己一本本翻。

AI 搜索像一个**研究助理**：你说"帮我搞清楚量子计算对密码学的影响"，他去翻 20 本书、3 篇论文，然后给你写一份带引用的摘要报告。

```
传统搜索（给链接）：
  用户 → "量子计算 密码学 影响" → 搜索引擎 → 10 个蓝色链接 → 用户自己阅读整合

AI 搜索（给答案）：
  用户 → "量子计算会如何影响现有的加密体系？"
    → 查询理解：识别为技术解释型问题
    → 子查询分解：量子计算原理 + RSA 破解 + 后量子密码学
    → 多源检索：搜 5 个权威网页 + 2 篇论文
    → 重排序 + 内容提取
    → LLM 综合生成，每句话标注引用 [1][2][3]
    → 结构化回答 + 来源列表
```

### 三次范式转变

| 范式 | 时代 | 核心技术 | 用户获得 |
|------|------|---------|---------|
| **关键词匹配** | 1998-2015 | 倒排索引 + PageRank + BM25 | 链接列表 |
| **语义理解** | 2015-2022 | BERT/向量检索 + 知识图谱 | 精选摘要 + 知识卡片 |
| **生成式搜索** | 2023-至今 | LLM + RAG + Agent | 直接答案 + 引用溯源 |

第三次范式转变不只是技术升级，它改变了信息获取的根本交互模型：用户不再需要"翻译"自己的需求为关键词，也不再需要自己做信息整合。

---

## 二、AI 搜索的核心架构

所有 AI 搜索系统都建立在一个共同的骨架上：**搜索 + RAG + 生成** 的三段式融合架构。但魔鬼在细节里。

### 2.1 全局数据流

```
                    ┌─────────────────────────────────────────┐
                    │           AI 搜索系统全局数据流          │
                    └─────────────────────────────────────────┘

用户查询                                                    结构化回答
  │                                                            ▲
  ▼                                                            │
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 查询理解  │→│  多源检索  │→│ 重排序/   │→│ 上下文    │→│ LLM 生成 │
│ & 改写    │  │  & 扇出   │  │  过滤    │  │ 构建     │  │ + 引用   │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
    │              │              │              │              │
  意图识别      Web API        Cross-        Chunk 拼接      带引用的
  实体抽取      知识图谱       Encoder        Token 预算      流式输出
  子查询分解    垂直索引       重排序         元数据注入      事实验证
  查询改写      实时爬取       E-E-A-T       上下文窗口管理   幻觉控制
```

### 2.2 各层关键设计决策

#### 查询理解层

这是 AI 搜索质量的**第一道闸门**。用户的自然语言查询往往模糊、多义、或过于口语化。

```python
# 查询理解的核心任务

class QueryUnderstanding:
    def process(self, raw_query: str) -> SearchPlan:
        # 1. 意图分类：事实型 / 探索型 / 导航型 / 对比型 / 操作型
        intent = self.classify_intent(raw_query)
        
        # 2. 实体识别：提取人名、地名、产品名、时间等
        entities = self.extract_entities(raw_query)
        
        # 3. 查询改写：消除歧义、补充上下文
        rewritten = self.rewrite_query(raw_query, intent, entities)
        
        # 4. 子查询分解：复杂问题拆成多个搜索任务
        sub_queries = self.decompose(rewritten, intent)
        
        # 5. 搜索策略选择：决定用哪些数据源、搜索深度
        strategy = self.plan_strategy(intent, sub_queries)
        
        return SearchPlan(
            intent=intent,
            sub_queries=sub_queries,
            strategy=strategy
        )
```

**查询改写（Query Rewriting）** 是质量提升最大的单点优化：

| 改写类型 | 原始查询 | 改写后 | 效果 |
|---------|---------|--------|------|
| 消歧 | "苹果最新消息" | "Apple 公司 2026 年最新产品发布" | 消除实体歧义 |
| 扩展 | "React 性能优化" | "React 性能优化 虚拟化 memo useMemo 代码分割" | 扩大召回 |
| HyDE | "为什么天空是蓝色的" | （生成假设性回答文档再用于检索） | 提升语义匹配 |
| 分解 | "对比 Go 和 Rust 做 Web 开发" | Q1: "Go Web 开发优势" + Q2: "Rust Web 开发优势" | 多角度召回 |

#### 检索层

AI 搜索的检索层面临传统搜索不曾有的复杂度 —— 需要**同时查多个异构数据源**，并在毫秒级做融合。

**Google 的"查询扇出"（Query Fan-out）** 是典型设计：

```
用户查询: "半程马拉松训练计划"
    │
    ├→ 子查询 1: "半程马拉松训练日程表" → Web 索引
    ├→ 子查询 2: "初跑者训练建议" → Web 索引
    ├→ 子查询 3: "马拉松营养策略" → 垂直索引
    ├→ 子查询 4: "半程马拉松" → 知识图谱（实体卡片）
    └→ 子查询 5: "马拉松训练视频" → YouTube 字幕索引
         │
         ▼
    并行检索 → 结果聚合 → 去重 → 按 E-E-A-T 过滤
```

**多源融合排序** 通常使用 **Reciprocal Rank Fusion (RRF)**：

```python
def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60) -> list:
    """
    将多个排序列表融合为统一排名。
    k 是平滑参数，防止高排名文档权重过于集中。
    """
    scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            scores[doc.id] += 1.0 / (k + rank)
    
    # 按融合分数降序排列
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 示例：融合 Web 搜索和知识库检索的结果
web_results = search_web("量子计算密码学影响")      # [doc_A, doc_B, doc_C]
kb_results = search_knowledge_base("量子计算密码学")  # [doc_C, doc_D, doc_A]

fused = reciprocal_rank_fusion([web_results, kb_results])
# doc_A 和 doc_C 在两个列表中都出现，融合后排名提升
```

#### 生成层

生成层的核心挑战不是"生成文本"，而是**带引用的忠实生成** —— 每个关键论断都必须有来源支撑，否则就是幻觉。

```python
GENERATION_PROMPT = """
你是一个 AI 搜索引擎的回答生成模块。

基于以下检索到的信息来源回答用户问题。

【严格要求】：
1. 每个事实性陈述必须用 [n] 标注引用来源
2. 如果信息源之间有冲突，明确指出分歧
3. 如果信息源不足以回答，明确说"根据现有检索结果，无法确认..."
4. 不要添加任何检索结果中不存在的信息
5. 优先引用时效性更强的来源

【信息来源】：
{sources}

【用户问题】：{query}

【回答格式】：
先给出简洁的核心回答（2-3句），然后展开详细分析。
"""
```

---

## 三、主流 AI 搜索产品技术剖析

### 3.1 四大玩家的架构对比

基于 iPullRank 2025 年的深度拆解报告，各产品的技术路线差异显著：

| 维度 | Google AI Overviews | Bing Copilot | Perplexity AI | ChatGPT Search |
|------|-------------------|-------------|--------------|----------------|
| **检索模型** | 查询扇出（多子查询并行） | 双通道（BM25 + 向量）+ 段落重排序 | 多引擎 API 调用（Google + Bing） | LLM 生成查询 + Bing API |
| **索引类型** | 全网索引 + KG + 垂直索引 | Bing 全网索引 | 无原生索引（实时检索） | 无持久化索引（实时抓取） |
| **核心优势** | 覆盖面最广，多意图覆盖 | 经典 SEO 兼容性强 | 透明度最高，引用清晰 | 对话式体验最自然 |
| **引用方式** | 内联链接 + 侧边栏卡片 | 内联上标链接 | 答案前列出来源 + 内联引用 | 内联或文末引用 |
| **生成模型** | Gemini | GPT-4 级别 | 内部模型（切换多种） | GPT-4o |

### 3.2 Perplexity AI —— AI 原生搜索的标杆

Perplexity 的设计哲学最能体现"AI 搜索"与"传统搜索加 AI"的区别：

**核心架构特点**：
1. **无自有索引** —— 完全依赖实时调用 Google/Bing API，这意味着它把精力全部放在"理解 + 综合"而非"爬取 + 索引"
2. **答案提取性优先** —— 极度偏好结构清晰、开头就给核心答案的页面
3. **透明引用** —— 每个事实陈述都标注来源编号，用户可以点击验证

**Perplexity 的处理流程**：
```
用户提问 → 查询改写（针对搜索引擎优化）
  → 并行调用 Google + Bing API
  → 候选页面的词汇 + 语义双重评估
  → 选择 Top-K 页面进行内容提取
  → 构建上下文（来源标注）
  → LLM 生成带引用的回答
  → 展示：引用来源列表 + 正文（内联 [1][2]...）
```

**为什么 Perplexity 体验好？** 三个关键设计：
- **先展示来源，再展示答案** —— 建立信任前置
- **引用粒度到句子级** —— 每个论断可验证
- **Follow-up 问题建议** —— 引导深入探索

### 3.3 Google AI Overviews —— 巨头的防守反击

Google 的优势在于**基础设施的深度** —— 20 年积累的全网索引、知识图谱、垂直搜索，这些是任何新玩家短期无法复制的。

**关键技术差异**：
- **E-E-A-T 过滤**（Experience, Expertise, Authoritativeness, Trustworthiness）—— Google 会根据内容创作者的专业度和权威性过滤搜索结果，这在医疗、法律、金融等 YMYL（Your Money Your Life）领域尤为重要
- **查询扇出的深度** —— 一个用户查询可能被拆解为 5-8 个子查询，覆盖用户可能的全部意图
- **片段提取性** —— Google 非常依赖能从页面中干净提取独立段落的能力，如果内容结构混乱，即使相关性高也很难被引用

### 3.4 产品选型决策框架

```
你需要什么？
    │
    ├─ 最全面的信息覆盖 → Google AI Overviews
    │    适合：通用知识查询、需要权威来源的场景
    │
    ├─ 最透明的引用体验 → Perplexity AI
    │    适合：研究型查询、需要验证每个论据的场景
    │
    ├─ 最自然的对话体验 → ChatGPT Search
    │    适合：探索型对话、多轮深入讨论
    │
    └─ 最好的中文体验 → 秘塔 AI / Kimi
         适合：中文内容查询、国内信息源
```

---

## 四、Web Browsing Agent

Web Browsing Agent 是 AI 搜索的"手和脚" —— 它让 AI 不只是调用搜索 API 拿结果，而是能像人一样**浏览网页、点击按钮、提交表单、阅读页面内容**。

### 4.1 架构层次

```
┌─────────────────────────────────────────────┐
│              Web Browsing Agent              │
├─────────────────────────────────────────────┤
│  决策层: LLM（决定做什么）                    │
│    "我需要搜索这个查询" / "我需要点击这个链接"    │
├─────────────────────────────────────────────┤
│  感知层: 网页内容理解                          │
│    HTML 解析 / 截图理解 / 文本提取             │
├─────────────────────────────────────────────┤
│  执行层: 浏览器自动化                          │
│    Playwright / Puppeteer / Selenium          │
├─────────────────────────────────────────────┤
│  工具层: 搜索 API + 内容提取                   │
│    Google API / Bing API / Tavily             │
│    Jina Reader / Trafilatura / Firecrawl      │
└─────────────────────────────────────────────┘
```

### 4.2 决策循环

一个成熟的 Web Browsing Agent 遵循 **搜索 → 阅读 → 判断 → 深入或终止** 的循环：

```python
class WebBrowsingAgent:
    """Web Browsing Agent 的核心决策循环"""
    
    def research(self, question: str, max_steps: int = 15) -> str:
        context = ResearchContext(question=question)
        
        for step in range(max_steps):
            # 1. 决策：下一步做什么？
            action = self.llm.decide(
                question=question,
                gathered_info=context.gathered_info,
                visited_urls=context.visited_urls,
                remaining_gaps=context.knowledge_gaps
            )
            
            if action.type == "SEARCH":
                # 搜索新查询
                results = self.search_api.search(action.query)
                context.add_search_results(results)
                
            elif action.type == "READ_PAGE":
                # 深入阅读某个页面
                content = self.browser.extract_content(action.url)
                relevant_info = self.llm.extract_relevant(
                    content, question, context.knowledge_gaps
                )
                context.add_info(relevant_info, source=action.url)
                
            elif action.type == "FOLLOW_LINK":
                # 跟踪页面内的链接深入
                content = self.browser.click_and_read(action.link)
                context.add_info(content, source=action.link)
                
            elif action.type == "REFINE_QUERY":
                # 根据已有信息改写查询
                new_query = self.llm.refine_query(
                    original=question,
                    what_we_know=context.gathered_info,
                    what_we_need=context.knowledge_gaps
                )
                results = self.search_api.search(new_query)
                context.add_search_results(results)
                
            elif action.type == "SUFFICIENT":
                # 信息足够，可以生成回答
                break
            
            # 2. 更新知识差距评估
            context.update_knowledge_gaps(self.llm)
        
        # 3. 综合生成带引用的回答
        return self.llm.synthesize(context)
```

### 4.3 网页内容提取：三种策略

| 策略 | 工具 | 适用场景 | 优缺点 |
|------|------|---------|--------|
| **API 提取** | Jina Reader, Firecrawl | 静态页面、文章类内容 | ✅ 快速、成本低 ❌ 无法处理动态页面 |
| **浏览器渲染** | Playwright, Puppeteer | 动态页面、SPA 应用 | ✅ 能执行 JS ❌ 慢、资源消耗大 |
| **截图 + 视觉理解** | Playwright + VLM | 复杂布局、交互式页面 | ✅ 理解能力最强 ❌ 最慢、成本最高 |

```python
# 使用 Playwright 进行动态页面渲染和内容提取
from playwright.async_api import async_playwright

async def extract_dynamic_content(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # 设置合理的超时和 User-Agent
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
        })
        
        await page.goto(url, wait_until="networkidle", timeout=15000)
        
        # 等待关键内容加载
        await page.wait_for_selector("article, main, .content", timeout=5000)
        
        # 提取正文内容（排除导航、广告等）
        content = await page.evaluate("""
            () => {
                // 移除无关元素
                const removeSelectors = ['nav', 'header', 'footer', 
                    '.ad', '.sidebar', '.cookie-banner'];
                removeSelectors.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                });
                
                // 获取主要内容
                const main = document.querySelector('article') 
                          || document.querySelector('main')
                          || document.querySelector('.content')
                          || document.body;
                return main.innerText;
            }
        """)
        
        await browser.close()
        return content
```

### 4.4 网页正文提取算法

从 HTML 中精确提取正文是搜索质量的基础工作。主流算法：

- **Trafilatura** —— 基于启发式规则 + 机器学习的混合方法，对新闻文章效果极好，Python 生态首选
- **Readability (Mozilla)** —— Firefox 阅读模式背后的算法，JavaScript 实现，关注可读性评分
- **Firecrawl** —— 云服务形式的网页抓取+提取，支持 Markdown 输出，专为 AI 消费设计

```python
# Trafilatura 提取示例 —— 简洁高效
import trafilatura

def extract_article(url: str) -> dict:
    downloaded = trafilatura.fetch_url(url)
    
    # 提取正文 + 元数据
    result = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        output_format="json",
        with_metadata=True
    )
    
    return {
        "text": result["text"],
        "title": result["title"],
        "date": result["date"],
        "author": result["author"],
        "url": url
    }
```

### 4.5 反爬虫与合规

Web Browsing Agent 必须在**能力**和**合规**之间找到平衡：

| 合规要求 | 实现方式 |
|---------|---------|
| 遵守 robots.txt | 请求前检查 robots.txt，尊重 Disallow 规则 |
| 请求频率控制 | 同一域名间隔 ≥ 1 秒，使用指数退避 |
| 身份标识 | User-Agent 明确标识为 Bot，不伪装为浏览器 |
| 缓存复用 | 相同 URL 短期内复用缓存，减少重复请求 |
| 数据最小化 | 只提取回答问题所需的内容，不存储全量页面 |

---

## 五、Deep Research —— AI 搜索的终极形态

Deep Research（深度研究）是 2025 年 AI 搜索领域最重要的突破方向。它不是简单的"搜一下、答一下"，而是一个**自主研究 Agent**，能花 5-30 分钟完成一个人类研究员可能需要数小时的调研任务。

### 5.1 什么是 Deep Research

传统 AI 搜索与 Deep Research 的本质区别：

| 维度 | 传统 AI 搜索 | Deep Research |
|------|------------|--------------|
| 检索深度 | 1 轮搜索，5-10 个结果 | 多轮迭代，50-200 个结果 |
| 执行时间 | 2-5 秒 | 5-30 分钟 |
| 推理复杂度 | 简单综合 | 多步推理、假设验证、矛盾检测 |
| 输出形式 | 简短回答 | 结构化研究报告 |
| 适用场景 | 事实查询、快速问答 | 竞品分析、文献综述、深度调研 |

### 5.2 四阶段架构

根据 2025 年的综述论文（Zhang et al., arXiv:2508.12752），Deep Research Agent 的架构可以抽象为四个阶段：

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  规划     │ →  │  问题生成     │ →  │  网络探索     │ →  │  报告生成     │
│ Planning  │    │  Question     │    │  Web          │    │  Report       │
│           │    │  Developing   │    │  Exploration  │    │  Generation   │
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
    │                  │                   │                    │
 将研究问题         根据子目标          通过 API 或          综合检索信息
 分解为结构化       生成具体、          浏览器主动           生成带引用的
 的子目标           多样的检索查询      交互收集信息         结构化报告
```

### 5.3 规划策略的深度

Deep Research 的规划不是简单的"拆解子问题"。它引入了**动态规划**和**反思调整**：

```python
class DeepResearchAgent:
    """Deep Research Agent 的核心流程"""
    
    async def research(self, topic: str) -> ResearchReport:
        # Phase 1: 初始规划
        plan = await self.create_research_plan(topic)
        # 例如: topic = "AI Agent 在金融行业的应用前景"
        # plan = {
        #   "sub_goals": [
        #     "AI Agent 在金融领域的现有应用案例",
        #     "金融 AI Agent 的技术架构特点",
        #     "监管合规对金融 AI Agent 的约束",
        #     "市场规模与增长预测",
        #     "主要玩家与竞争格局"
        #   ],
        #   "priority_order": [0, 2, 1, 4, 3],
        #   "estimated_depth": "deep"
        # }
        
        gathered_sections = []
        
        for sub_goal in plan.prioritized_sub_goals:
            # Phase 2: 为每个子目标生成检索查询
            queries = await self.generate_queries(sub_goal, gathered_sections)
            # 多角度生成 3-5 个不同查询
            
            # Phase 3: 执行检索 + 深入探索
            section_info = await self.explore(queries)
            gathered_sections.append(section_info)
            
            # 关键：动态调整计划
            plan = await self.reflect_and_adjust(
                plan, gathered_sections, topic
            )
            # 可能发现新的重要子目标，或调整优先级
        
        # Phase 4: 生成结构化报告
        report = await self.generate_report(topic, gathered_sections)
        
        return report
    
    async def reflect_and_adjust(self, plan, gathered, topic):
        """
        反思机制：根据已收集的信息动态调整计划。
        这是 Deep Research 区别于简单多步搜索的关键。
        """
        reflection = await self.llm.reflect(
            prompt=f"""
            研究主题: {topic}
            原始计划: {plan.sub_goals}
            已收集信息摘要: {[s.summary for s in gathered]}
            
            请评估：
            1. 哪些子目标已经充分覆盖？
            2. 发现了哪些新的重要方向需要补充？
            3. 是否存在信息冲突需要进一步验证？
            4. 当前的优先级是否需要调整？
            
            输出调整后的计划。
            """
        )
        return reflection.adjusted_plan
```

### 5.4 推理增强检索（Reasoning-Enhanced Retrieval）

Deep Research 的核心创新是把**推理深度嵌入检索过程**：

**Search-in-the-Chain**：在推理链中动态插入搜索步骤

```
推理步骤 1: "要分析量子计算对 RSA 的威胁，需要知道 Shor 算法的原理"
  → 触发搜索: "Shor's algorithm RSA factoring complexity"
  → 获取结果，整合到推理上下文

推理步骤 2: "Shor 算法需要足够多的量子比特，当前量子计算机的规模是？"
  → 触发搜索: "quantum computer qubit count 2026 latest"
  → 获取结果，继续推理

推理步骤 3: "当前最大约 1000+ 逻辑量子比特，破解 RSA-2048 需要约 4000 个..."
  → 形成中间结论，评估是否需要更多信息
```

**冲突检测与解决**：当不同来源给出矛盾信息时，系统需要额外推理：

```
来源 A: "GPT-5 将于 2026 年 Q1 发布"
来源 B: "OpenAI 尚未确认 GPT-5 的发布时间"
来源 C: "GPT-5 已于 2025 年 12 月内测"

→ 冲突检测：发布时间存在矛盾
→ 可信度评估：来源 B（官方博客）> 来源 A（科技博客）> 来源 C（未知来源）
→ 综合结论："截至目前，OpenAI 官方尚未确认 GPT-5 的具体发布时间 [B]。
   有未经证实的报道称... [A][C]"
```

### 5.5 OpenAI Deep Research 的实现思路

OpenAI 在 2025 年 2 月发布的 Deep Research 是这个方向的标志性产品。其核心特点：

- **基于 o3 模型**：利用 o3 的强推理能力做规划和综合
- **多步浏览**：平均浏览 20-50 个网页，花费 5-30 分钟
- **结构化报告**：输出带目录、分节、引用的完整报告
- **Agent 终极形态**：OpenAI Deep Research 团队明确表示"Agent 的终极形态是所有任务都能完成"

---

## 六、搜索质量优化

### 6.1 查询理解深度解析

#### 意图识别

搜索查询的意图分类是优化的第一步：

| 意图类型 | 示例 | 最佳处理策略 |
|---------|------|------------|
| **事实型** | "地球到太阳的距离" | 直接回答，单一权威来源即可 |
| **探索型** | "如何学习机器学习" | 多源综合，提供结构化路径 |
| **导航型** | "GitHub 登录" | 直接给出 URL，无需综合 |
| **对比型** | "React vs Vue 2026" | 多维度对比表格 |
| **操作型** | "帮我查一下明天的航班" | 调用工具执行 |
| **时效型** | "今天的科技新闻" | 优先最新结果，时间过滤 |

#### 查询改写的高级技术

**HyDE (Hypothetical Document Embeddings)**：

```python
def hyde_search(query: str, search_engine, llm) -> list:
    """
    HyDE：先让 LLM 生成一个"假设性回答文档"，
    再用这个文档的 embedding 去检索真实文档。
    
    原理：假设文档和真实文档在语义空间更接近，
    比用户的短查询更能匹配目标文档。
    """
    # Step 1: 生成假设性文档
    hypothetical_doc = llm.generate(
        f"请写一段约 200 字的文章来回答这个问题：{query}"
    )
    
    # Step 2: 用假设文档做 embedding 检索
    results = search_engine.semantic_search(hypothetical_doc)
    
    # Step 3: 也用原始查询检索，取并集
    original_results = search_engine.semantic_search(query)
    
    return merge_and_deduplicate(results, original_results)
```

**Multi-Query 检索**：

```python
def multi_query_search(query: str, llm, search_engine) -> list:
    """
    从多个角度生成查询，扩大召回面。
    """
    # 让 LLM 从不同角度改写查询
    queries = llm.generate(f"""
        原始问题: {query}
        
        请从 3 个不同角度改写这个问题，使每个版本关注不同方面：
        1. 关注定义和概念
        2. 关注实际应用和案例
        3. 关注技术细节和实现
        
        输出 3 个独立的搜索查询。
    """)
    
    # 并行检索
    all_results = []
    for q in queries:
        results = search_engine.search(q)
        all_results.append(results)
    
    # RRF 融合
    return reciprocal_rank_fusion(all_results)
```

### 6.2 结果重排序

检索拿回来的结果排序往往不够精确，**重排序（Reranking）** 是提升搜索质量最有效的单点优化。

#### Cross-Encoder 重排序

```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list:
        """
        Cross-Encoder 对 (query, doc) 对做联合编码，
        比 Bi-Encoder 的分离编码更准确，但更慢。
        
        适用场景：在初始检索（召回 50-100 个）之后做精排（取 Top-5）。
        """
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        
        # 按分数排序
        ranked = sorted(
            zip(documents, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked[:top_k]
```

#### LLM 重排序

用 LLM 做重排序是 2025 年的新趋势，尤其适合需要理解复杂语义关系的场景：

```python
def llm_rerank(query: str, documents: list[dict], llm) -> list:
    """
    让 LLM 对文档进行相关性评分。
    优点：理解深层语义，能处理复杂查询意图。
    缺点：慢、贵。适合少量候选的精排。
    """
    prompt = f"""
    用户查询: {query}
    
    请对以下文档按与查询的相关性从高到低排序。
    考虑因素：内容相关性、信息质量、时效性、权威性。
    
    文档列表:
    {format_documents(documents)}
    
    输出格式: 排序后的文档 ID 列表，如 [3, 1, 5, 2, 4]
    """
    
    ranking = llm.generate(prompt)
    return reorder_by_ranking(documents, ranking)
```

#### 重排序方案对比

| 方案 | 延迟 | 精度 | 成本 | 适用场景 |
|------|------|------|------|---------|
| BM25 基线 | <10ms | 中 | 极低 | 关键词匹配 |
| Bi-Encoder (BGE) | 10-50ms | 中高 | 低 | 语义召回 |
| Cross-Encoder (BGE-Reranker) | 50-200ms | 高 | 中 | 精排 Top-50→Top-5 |
| LLM Reranking | 1-5s | 最高 | 高 | 复杂意图的少量精排 |
| Cohere Rerank API | 100-300ms | 高 | 中 | 云服务，开箱即用 |

### 6.3 引用与事实验证

AI 搜索的可信度建立在**引用准确性**上。引用标注有两种主要策略：

**生成时标注**（Inline Citation）：

```
生成 Prompt 中要求：
"每个事实陈述后用 [n] 标注来源编号"

LLM 输出：
"量子计算机使用量子比特进行计算 [1]，目前最先进的量子处理器
已达到 1000+ 量子比特 [2]，但距离实际破解 RSA-2048 加密
仍需约 4000 个逻辑量子比特 [3]。"
```

**生成后验证**（Post-hoc Verification）：

```python
def verify_citations(answer: str, sources: list[dict]) -> dict:
    """
    生成后验证：检查每个引用是否与来源内容一致。
    """
    claims = extract_claims_with_citations(answer)
    
    results = []
    for claim in claims:
        source = sources[claim.citation_id]
        
        # 用 NLI (Natural Language Inference) 模型验证
        verification = nli_model.predict(
            premise=source.content,
            hypothesis=claim.text
        )
        
        results.append({
            "claim": claim.text,
            "citation": claim.citation_id,
            "verdict": verification.label,  # SUPPORT / REFUTE / NEUTRAL
            "confidence": verification.score
        })
    
    return {
        "total_claims": len(claims),
        "supported": sum(1 for r in results if r["verdict"] == "SUPPORT"),
        "refuted": sum(1 for r in results if r["verdict"] == "REFUTE"),
        "details": results
    }
```

### 6.4 幻觉控制策略

AI 搜索中的幻觉比纯 LLM 对话更危险 —— 因为用户信任搜索结果是有来源支撑的。

```
幻觉控制的五道防线：

1. 检索质量 → 确保 Top-K 结果确实相关
   ↓
2. 上下文约束 → Prompt 明确要求"只基于检索结果回答"
   ↓
3. 引用强制 → 每个论断必须标注来源
   ↓
4. 生成后验证 → NLI 模型检查引用一致性
   ↓
5. 置信度表达 → 不确定时明确说"无法确认"
```

---

## 七、搜索 API 工程实现

### 7.1 主流搜索 API 对比

基于 2026 年 3 月的最新评测（Rhumb AN Score），四大 AI 搜索 API 的详细对比：

| 维度 | Exa (8.7) | Tavily (8.6) | Serper (8.0) | Brave Search (7.1) |
|------|-----------|-------------|-------------|-------------------|
| **定位** | 神经语义搜索 | Agent 专用搜索 | Google 结果封装 | 隐私优先独立索引 |
| **检索方式** | 神经嵌入检索 | 多引擎聚合 | Google API 封装 | 自有索引 |
| **结构化数据** | ✅ 内容直接返回 | ✅ answer + results | ✅ 知识图谱 + 富结果 | ⚠️ 基础结构 |
| **内容提取** | ✅ 搜索时同步返回 | ✅ include_raw_content | ❌ 需额外抓取 | ⚠️ extra_snippets |
| **异步支持** | ✅ | ✅ 原生 async | ✅ | ✅ |
| **免费额度** | 有 | 有 | 有 | 有 |
| **最佳场景** | 语义研究、概念探索 | Agent 通用搜索 | 时事新闻、精确查询 | 隐私合规场景 |

### 7.2 选型决策树

```
你的 Agent 需要什么？
    │
    ├─ 语义/概念研究（"找与 X 相关的高质量文章"）
    │   → Exa — 神经嵌入检索，找到关键词搜索遗漏的内容
    │
    ├─ 通用 Agent 搜索（快速获取结构化结果）
    │   → Tavily — 专为 Agent 设计，search_depth 参数控制质量/速度权衡
    │
    ├─ 时事新闻/精确查询（"昨天的 AI 新闻"）
    │   → Serper — Google 结果最新最全，新闻端点支持时间过滤
    │
    ├─ 隐私合规/多源验证（需要非 Google 来源）
    │   → Brave Search — 独立索引，结果与 Google/Bing 不同
    │
    └─ 中文搜索优先
        → 博查 AI / 百度千帆 — 国内信息源覆盖更好
```

### 7.3 实际集成代码

```python
# Tavily — Agent 最常用的搜索 API 集成示例

from tavily import TavilyClient
import asyncio

class AISearchEngine:
    def __init__(self, tavily_api_key: str):
        self.client = TavilyClient(api_key=tavily_api_key)
    
    def quick_search(self, query: str) -> dict:
        """快速搜索：低延迟，适合简单事实查询"""
        return self.client.search(
            query=query,
            search_depth="basic",   # basic: 快但浅  advanced: 慢但深
            max_results=5,
            include_answer=True,    # 返回合成摘要
            include_raw_content=False
        )
    
    def deep_search(self, query: str) -> dict:
        """深度搜索：高质量，适合复杂研究型查询"""
        return self.client.search(
            query=query,
            search_depth="advanced",  # 消耗双倍积分
            max_results=10,
            include_answer=True,
            include_raw_content=True,  # 包含全文提取
            include_images=False
        )
    
    def search_with_context(self, query: str, context: str) -> dict:
        """上下文感知搜索：结合对话历史优化检索"""
        enhanced_query = self._enhance_query(query, context)
        return self.deep_search(enhanced_query)
    
    def _enhance_query(self, query: str, context: str) -> str:
        """根据对话上下文增强查询"""
        # 实际实现中可以用 LLM 做查询改写
        return f"{query} {context[:200]}"


# Exa — 语义搜索集成示例

from exa_py import Exa

class SemanticSearchEngine:
    def __init__(self, exa_api_key: str):
        self.exa = Exa(api_key=exa_api_key)
    
    def semantic_search(self, query: str, num_results: int = 5) -> list:
        """语义搜索：找概念相关的内容，不仅是关键词匹配"""
        results = self.exa.search_and_contents(
            query=query,
            type="neural",           # 神经搜索模式
            num_results=num_results,
            text={"max_characters": 3000},  # 直接返回内容
            highlights=True          # 返回关键段落高亮
        )
        
        return [{
            "title": r.title,
            "url": r.url,
            "text": r.text,
            "highlights": r.highlights,
            "score": r.score
        } for r in results.results]
    
    def find_similar(self, url: str) -> list:
        """相似内容发现：给一个 URL，找类似的文章"""
        results = self.exa.find_similar_and_contents(
            url=url,
            num_results=5,
            text={"max_characters": 2000}
        )
        return results.results
```

### 7.4 缓存策略设计

搜索 API 调用有成本（金钱和延迟），缓存是必要的工程优化：

```python
import hashlib
import json
import time

class SearchCache:
    """搜索结果缓存，支持语义相似查询的缓存命中"""
    
    def __init__(self, redis_client, ttl_seconds: int = 3600):
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    def get(self, query: str, search_depth: str) -> dict | None:
        """精确匹配缓存"""
        key = self._make_key(query, search_depth)
        cached = self.redis.get(key)
        if cached:
            data = json.loads(cached)
            # 检查时效性
            if time.time() - data["cached_at"] < self.ttl:
                return data["results"]
        return None
    
    def set(self, query: str, search_depth: str, results: dict):
        """存入缓存"""
        key = self._make_key(query, search_depth)
        data = {
            "results": results,
            "cached_at": time.time()
        }
        self.redis.setex(key, self.ttl, json.dumps(data))
    
    def _make_key(self, query: str, depth: str) -> str:
        """生成缓存键"""
        normalized = query.strip().lower()
        hash_val = hashlib.md5(f"{normalized}:{depth}".encode()).hexdigest()
        return f"search_cache:{hash_val}"
    
    # 进阶：语义缓存
    def semantic_get(self, query: str, embedding_model) -> dict | None:
        """
        语义缓存：如果之前搜过语义相似的查询，复用结果。
        例如："Python 性能优化" 和 "如何让 Python 跑得更快" 应该命中同一缓存。
        """
        query_embedding = embedding_model.encode(query)
        
        # 在缓存的查询 embedding 中找最相似的
        similar = self.vector_store.search(query_embedding, threshold=0.92)
        
        if similar:
            return self.get(similar[0].original_query, "basic")
        return None
```

### 7.5 成本控制

| 优化手段 | 效果 | 实现复杂度 |
|---------|------|-----------|
| 精确缓存 | 减少 30-50% API 调用 | 低 |
| 语义缓存 | 额外减少 10-20% | 中 |
| 查询分流（简单查询用 basic，复杂查询用 advanced） | 减少 40% 成本 | 中 |
| 搜索必要性判断（不是所有问题都需要搜索） | 减少 20-30% 无效搜索 | 中 |
| 结果复用（多轮对话中复用之前的搜索结果） | 减少 30% 重复搜索 | 低 |

```python
class SmartSearchRouter:
    """智能搜索路由：决定是否需要搜索、用什么深度搜索"""
    
    def should_search(self, query: str, conversation_context: list) -> dict:
        """
        不是所有查询都需要搜索。
        - 纯闲聊 → 不搜
        - LLM 知识范围内的通用问题 → 不搜
        - 需要实时信息 / 具体数据 / 最新事件 → 搜
        """
        decision = self.llm.classify(f"""
            用户查询: {query}
            
            判断这个查询是否需要进行网络搜索：
            - 如果是闲聊、常识性问题、或 LLM 训练数据能充分回答的 → NO
            - 如果涉及实时信息、具体数据、最新事件、特定产品/人物 → YES
            - 如果需要搜索，判断深度：basic（事实查询）或 advanced（研究型）
            
            输出: {{"need_search": bool, "depth": "basic"|"advanced", "reason": "..."}}
        """)
        
        return decision
```

### 7.6 可观测性设计

AI 搜索系统的可观测性比普通 Web 应用更复杂，需要追踪完整的"查询 → 检索 → 生成"链路：

```python
import time
from dataclasses import dataclass

@dataclass
class SearchTrace:
    """搜索追踪记录"""
    trace_id: str
    query: str
    
    # 查询理解
    intent: str
    sub_queries: list[str]
    query_understanding_ms: float
    
    # 检索
    search_api: str
    search_depth: str
    results_count: int
    search_ms: float
    cache_hit: bool
    
    # 重排序
    reranked_results: int
    rerank_ms: float
    
    # 生成
    context_tokens: int
    generation_tokens: int
    generation_ms: float
    
    # 质量指标
    citations_count: int
    citation_accuracy: float  # 引用验证通过率
    
    # 成本
    search_api_cost: float
    llm_cost: float
    total_cost: float
    
    # 总延迟
    total_ms: float


# 关键监控指标
METRICS = {
    "搜索质量": [
        "citation_accuracy",     # 引用准确率 (目标 > 90%)
        "answer_relevancy",      # 回答相关性 (人工评估)
        "hallucination_rate",    # 幻觉率 (目标 < 5%)
    ],
    "性能": [
        "p50_latency_ms",        # 50 分位延迟 (目标 < 3000ms)
        "p99_latency_ms",        # 99 分位延迟 (目标 < 8000ms)
        "search_cache_hit_rate", # 缓存命中率 (目标 > 30%)
    ],
    "成本": [
        "cost_per_query",        # 每次查询成本 (目标 < $0.05)
        "search_api_calls_saved", # 缓存节省的 API 调用数
        "unnecessary_search_rate", # 不必要搜索占比 (目标 < 15%)
    ]
}
```

---

## 八、搜索质量评估体系

### 8.1 评估维度

AI 搜索的评估不能只看"答案对不对"，需要多维度评估：

| 评估维度 | 定义 | 评估方法 | 目标值 |
|---------|------|---------|--------|
| **检索召回率** | 相关文档被检索到的比例 | Recall@K | >80% |
| **检索精确率** | 检索结果中相关文档的比例 | Precision@K | >60% |
| **排序质量** | 相关文档排序的合理性 | NDCG@K, MRR | >0.7 |
| **回答相关性** | 回答与问题的匹配程度 | LLM-as-Judge | >4/5 |
| **引用准确率** | 引用内容与来源一致的比例 | NLI 验证 | >90% |
| **忠实度** | 回答是否完全基于检索内容 | Faithfulness Score | >85% |
| **完整性** | 回答是否覆盖了问题的全部要点 | LLM-as-Judge | >3.5/5 |
| **时效性** | 引用来源的时间新鲜度 | 时间戳检查 | 视场景而定 |
| **响应延迟** | 从查询到回答的总时间 | P50/P95 | <3s/<8s |

### 8.2 NDCG（归一化折损累积增益）—— 搜索排序的标准指标

```python
import numpy as np

def ndcg_at_k(relevance_scores: list[int], k: int) -> float:
    """
    NDCG@K: 衡量排序质量的标准指标。
    
    relevance_scores: 按排序顺序的相关性评分列表
                      (0=不相关, 1=部分相关, 2=高度相关, 3=完美匹配)
    
    例子：
      搜索 "Python 性能优化"
      结果排序: [文章A(3分), 文章B(2分), 文章C(0分), 文章D(1分), 文章E(3分)]
      
      DCG@5 = 3/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5) + 3/log2(6)
            = 3.0 + 1.26 + 0 + 0.43 + 1.16 = 5.85
      
      理想排序: [3, 3, 2, 1, 0]
      IDCG@5 = 3/log2(2) + 3/log2(3) + 2/log2(4) + 1/log2(5) + 0/log2(6)
             = 3.0 + 1.89 + 1.0 + 0.43 + 0 = 6.32
      
      NDCG@5 = DCG/IDCG = 5.85/6.32 = 0.926 (很好的排序质量)
    """
    def dcg(scores, k):
        scores = scores[:k]
        return sum(
            score / np.log2(idx + 2) 
            for idx, score in enumerate(scores)
        )
    
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True), k)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
```

### 8.3 LLM-as-Judge 评估搜索质量

```python
def evaluate_search_quality(query: str, answer: str, sources: list, judge_llm) -> dict:
    """使用 LLM 作为评判者评估搜索回答质量"""
    
    evaluation = judge_llm.evaluate(f"""
    你是一个 AI 搜索质量评估专家。请评估以下搜索回答的质量。
    
    【用户查询】: {query}
    
    【AI 回答】: {answer}
    
    【检索来源】:
    {format_sources(sources)}
    
    请从以下 5 个维度评分（1-5 分）并给出理由：
    
    1. **相关性** (Relevancy): 回答是否直接回应了用户的问题？
    2. **准确性** (Accuracy): 回答中的事实是否正确？
    3. **引用质量** (Citation Quality): 引用是否准确对应来源内容？
    4. **完整性** (Completeness): 是否覆盖了问题的全部要点？
    5. **表达质量** (Presentation): 回答是否清晰、有结构、易理解？
    
    输出格式:
    {{
        "relevancy": {{"score": N, "reason": "..."}},
        "accuracy": {{"score": N, "reason": "..."}},
        "citation_quality": {{"score": N, "reason": "..."}},
        "completeness": {{"score": N, "reason": "..."}},
        "presentation": {{"score": N, "reason": "..."}},
        "overall": N,
        "summary": "..."
    }}
    """)
    
    return evaluation
```

---

## 九、AI 搜索 vs RAG：协同与差异

AI 搜索和 RAG 经常被混为一谈，但它们有本质区别：

### 9.1 核心差异

| 维度 | RAG | AI 搜索 |
|------|-----|---------|
| **数据源** | 预索引的私有知识库 | 实时互联网 + 多源 |
| **时效性** | 取决于索引更新频率 | 实时 |
| **数据规模** | 百万~十亿级文档 | 整个互联网 |
| **检索控制** | 完全可控（你的数据、你的索引） | 部分可控（依赖外部 API） |
| **成本模型** | 基础设施成本（向量库 + 计算） | API 调用成本 |
| **典型场景** | 企业知识问答、文档助手 | 通用信息查询、研究调研 |
| **幻觉控制** | 相对容易（来源可控） | 更难（来源不可控） |

### 9.2 融合架构

实际生产系统往往**同时使用**两者：

```
用户查询
    │
    ├─ 路由判断: 这个问题应该查内部知识库还是搜互联网？
    │    │
    │    ├─ "我们公司的报销流程是什么？" → RAG（内部知识库）
    │    ├─ "最新的 GPT 模型有什么改进？" → AI 搜索（互联网）
    │    └─ "用 LangGraph 实现我们的审批流程" → 两者都用
    │
    ├─ RAG 路径:
    │    向量检索 → 重排序 → 生成
    │
    ├─ AI 搜索路径:
    │    查询改写 → 搜索 API → 内容提取 → 重排序 → 生成
    │
    └─ 融合路径:
         RAG 结果 + 搜索结果 → RRF 融合 → 统一生成
```

```python
class HybridSearchSystem:
    """RAG + AI 搜索的融合系统"""
    
    def __init__(self, rag_engine, search_engine, router_llm):
        self.rag = rag_engine
        self.search = search_engine
        self.router = router_llm
    
    async def answer(self, query: str) -> dict:
        # Step 1: 路由决策
        route = await self.router.classify(query)
        
        if route == "internal_only":
            results = await self.rag.retrieve(query)
        elif route == "web_only":
            results = await self.search.search(query)
        else:  # "hybrid"
            rag_results, web_results = await asyncio.gather(
                self.rag.retrieve(query),
                self.search.search(query)
            )
            results = self.fuse(rag_results, web_results)
        
        # Step 2: 生成回答
        answer = await self.generate(query, results)
        
        return {
            "answer": answer,
            "sources": results,
            "route": route
        }
```

---

## 十、前沿方向与未来趋势

### 10.1 Agentic Search —— 搜索即 Agent

2025-2026 年的核心趋势是搜索与 Agent 的深度融合。搜索不再是一个被动的"工具"，而是一个有推理能力的自主系统：

```
当前: 用户 → 搜索 → 结果 → 用户整合
    （搜索是工具）

未来: 用户 → Agent → [搜索 ← 推理 → 验证 → 再搜索 → 综合]
    （搜索是 Agent 的认知过程）
```

**关键技术方向**：
- **强化学习优化搜索策略**：用 RL 训练 Agent 学会"什么时候搜、搜什么、搜几次"
- **Search-in-the-Chain**：在推理链中动态插入搜索步骤
- **自适应搜索深度**：根据问题复杂度自动决定搜索深度

### 10.2 多模态搜索

搜索不再局限于文本：

| 模态 | 技术方向 | 典型场景 |
|------|---------|---------|
| 文本 → 文本 | 传统语义搜索 | 知识查询 |
| 图像 → 文本 | 视觉搜索 + 描述生成 | "这张照片是哪里？" |
| 文本 → 图像 | 搜索 + 图像生成 | "给我看量子计算机的结构图" |
| 语音 → 文本 | 语音搜索 + 回答生成 | 智能助手 |
| 视频 → 文本 | 视频内容搜索 | "这个视频讲了什么？" |

### 10.3 个性化 AI 搜索

同一个查询，不同用户应该得到不同的回答：

- **专业度适配**："什么是 Transformer" → 给 ML 研究者讲数学推导，给产品经理讲类比
- **上下文感知**：结合用户的对话历史和偏好
- **记忆搜索**：把用户之前的搜索和发现作为未来搜索的背景

### 10.4 实时流式搜索

传统"等搜索完再生成"的模式正在被"边搜边生成"的流式模式替代：

```
传统模式（串行）: 
  搜索(3s) → 处理(1s) → 生成(3s) = 用户等 7s 才看到内容

流式模式（并行）:
  t=0s: 开始搜索，同时展示"正在搜索..."
  t=1s: 第一批结果到达，开始生成第一段
  t=2s: 用户已经能看到部分回答
  t=3s: 更多搜索结果到达，补充生成
  t=5s: 全部完成，用户已经读了一半
```

### 10.5 搜索的可验证性

未来的 AI 搜索将更加强调**每一步都可审计**：

- **检索链路透明**：用户可以看到搜索了哪些查询、访问了哪些网页
- **推理过程可见**：Deep Research 的每一步思考过程可展开查看
- **引用双向链接**：从回答可以跳转到来源，从来源也可以看到它支撑了哪些结论

---

## 十一、面试重点标注 ⭐

### 必须能讲清楚的概念

1. **AI 搜索 vs 传统搜索的本质区别** —— 从"给链接"到"给答案"，RAG 架构是核心
2. **查询理解与改写** —— 意图识别、HyDE、Multi-Query 的原理和适用场景
3. **重排序** —— Cross-Encoder vs Bi-Encoder 的区别，为什么重排序能显著提升质量
4. **引用与幻觉控制** —— 生成时引用标注 + 生成后 NLI 验证
5. **RRF 融合** —— 多源搜索结果的融合排序算法
6. **AI 搜索产品对比** —— Perplexity vs Google AI Overviews 的架构差异
7. **搜索 API 选型** —— Tavily/Exa/Serper 各自的优劣势

### 能加分的深度知识

1. **Deep Research 的四阶段架构** —— 规划-问题生成-网络探索-报告生成
2. **Search-in-the-Chain** —— 推理增强检索的具体实现
3. **NDCG 的计算方法** —— 搜索排序质量的标准评估指标
4. **缓存策略设计** —— 精确缓存 + 语义缓存的工程实现
5. **成本优化** —— 搜索必要性判断、查询分流、结果复用

---

## 参考资料

1. iPullRank. *AI Search Architecture Deep Dive: Teardowns of Leading Platforms*. 2025.
2. Zhang, W. et al. *Deep Research: A Survey of Autonomous Research Agents*. arXiv:2508.12752. 2025.
3. Rhumb. *Exa vs Tavily vs Serper vs Brave Search for AI Agents — AN Score Comparison*. DEV Community, 2026.
4. OpenAI. *Introducing Deep Research*. 2025.
5. Gao, Y. et al. *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997. 2024.
6. Tavily Documentation. https://tavily.com/
7. Exa Documentation. https://docs.exa.ai/
8. Perplexity AI. https://www.perplexity.ai/
