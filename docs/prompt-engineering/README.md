# ✍️ Prompt Engineering — 提示词工程

## 概述

Prompt Engineering（提示词工程）是指通过精心设计输入提示词来引导大语言模型产生期望输出的技术与方法论。它是使用 LLM 最直接、最重要的技能，好的提示词设计能显著提升模型的输出质量和任务完成效果。

## 核心知识体系

### 1. 基础提示技巧

- **角色设定（Role Prompting）**：为模型指定专家角色，引导专业化输出
- **明确指令**：清晰、具体地描述任务要求
- **输出格式控制**：指定 JSON、Markdown、表格等输出格式
- **分隔符使用**：用 `---`、`"""` 等分隔不同内容区域
- **示例提供**：通过示例展示期望的输入输出模式

### 2. 高级提示策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| Zero-shot | 不提供示例，直接描述任务 | 简单、明确的任务 |
| Few-shot | 提供少量示例 | 需要特定格式或风格 |
| Chain-of-Thought (CoT) | 引导逐步推理 | 复杂推理、数学问题 |
| Tree-of-Thought (ToT) | 探索多条推理路径 | 需要创造性思维 |
| Self-Consistency | 多次采样取一致结果 | 提升推理准确性 |
| ReAct | 推理与行动交替 | Agent 场景 |

### 3. Chain-of-Thought 深入

- **标准 CoT**：在 Few-shot 示例中展示推理过程
- **Zero-shot CoT**：添加 "Let's think step by step" 触发推理
- **Auto-CoT**：自动生成推理链
- **CoT 的局限**：简单任务可能反而降低效果，小模型效果有限

### 4. 提示词设计模式

- **CRISPE 框架**：Capacity（角色）、Request（请求）、Insight（背景）、Style（风格）、Personality（个性）、Experiment（实验）
- **CO-STAR 框架**：Context（上下文）、Objective（目标）、Style（风格）、Tone（语气）、Audience（受众）、Response（响应格式）
- **结构化提示**：使用 XML 标签、Markdown 标题等组织复杂提示
- **元提示（Meta-Prompting）**：用 LLM 生成和优化提示词

### 5. 系统提示词设计

- **System Prompt 最佳实践**：
  - 明确角色定位与能力边界
  - 定义行为规则与约束条件
  - 设置输出格式与风格要求
  - 处理边界情况与拒绝策略
- **多轮对话管理**：上下文窗口利用、对话历史压缩

### 6. Prompt 管理工程化 ⭐

> 在生产环境中，一个 Prompt 的改动可能影响数万个请求的输出。Prompt 管理是被严重低估的工程问题。

- **五大工程化难题**：
  - **版本管理**：没有 diff 能力看不出两版 Prompt 对输出的影响、没有审计日志、没有快速回滚、dev/staging/prod 版本不一致
  - **Prompt 与上下文耦合**：Prompt 与 RAG 文档片段、工具定义、对话历史、用户信息强耦合，换上下文可能产生幻觉
  - **Prompt 膨胀**：随需求累加 System Prompt 从 200 tokens 膨胀到 8000+ tokens，成本翻 40 倍
  - **多语言/多场景管理**：中英文版本、不同客户定制化、不同业务场景变体
  - **Prompt 注入防御**：防御指令占用 Token 且不一定可靠

- **Prompt 工程化管理架构**：
  ```
  Prompt 模板仓库 (Git) → 渲染引擎 (Jinja2) → LLM 调用
       ↓
  CI/CD Pipeline: Lint & Review → 回归评测 (RAGAS) → 灰度发布 / 回滚
  ```

- **解决方案**：
  - **Prompt as Code**：Prompt 存储在 Git 仓库，用 Jinja2 模板引擎管理变量和条件逻辑
  - **回归测试绑定**：每个 Prompt 版本关联一组评测用例，CI 自动运行回归
  - **灰度发布**：新 Prompt 先对 10% 流量生效，观察核心指标无退化后全量
  - **工具选型**：PromptLayer（版本管理 + A/B 测试）、Humanloop（管理 + 评估 + 监控一体化）、自建（Git + Jinja2 + RAGAS）

### 7. Prompt 与业务逻辑的解耦设计 ⭐

> Prompt 膨胀的根因往往是把太多业务逻辑塞进了 Prompt。架构层面的解耦设计是治本之策。

- **Prompt 组合模式**：
  - 将大 Prompt 拆分为可复用的模块：
  ```
  最终 Prompt = 角色模块 + 规则模块 + 格式模块 + 场景模块 + 动态上下文
  
  角色模块：定义 Agent 的身份和能力边界（通用，跨场景复用）
  规则模块：安全规则、合规要求（通用，全局生效）
  格式模块：输出格式要求（按场景选择）
  场景模块：特定业务场景的指令（按场景动态加载）
  动态上下文：RAG 结果、用户信息、对话历史（每次请求不同）
  ```

- **条件 Prompt 加载**：
  ```python
  # 伪代码：根据场景动态组装 Prompt
  def build_prompt(user_role, scenario, context):
      modules = []
      modules.append(load_module("base_role"))        # 基础角色（必选）
      modules.append(load_module("safety_rules"))     # 安全规则（必选）
      
      if user_role == "vip":
          modules.append(load_module("vip_treatment"))  # VIP 专属指令
      
      modules.append(load_module(f"scenario_{scenario}"))  # 场景指令
      
      if context.get("has_rag_results"):
          modules.append(load_module("rag_instructions"))  # RAG 指令
      else:
          modules.append(load_module("no_context_fallback"))  # 无上下文兜底
      
      return render_prompt(modules, context)
  ```

- **Prompt 与代码的边界判断**：
  | 逻辑类型 | 放在 Prompt 里 | 放在代码里 | 判断标准 |
  |---------|---------------|-----------|----------|
  | 流程控制 | ❌ | ✅ | 确定性逻辑用代码 |
  | 输出格式 | ✅ | 🟡 配合 Schema 校验 | Prompt 定义 + 代码验证 |
  | 业务规则 | 🟡 简单规则 | ✅ 复杂规则 | 规则超过 3 条用代码 |
  | 安全约束 | 🟡 辅助 | ✅ 主要 | 安全不能只靠 Prompt |
  | 语气风格 | ✅ | ❌ | 天然适合 Prompt |
  | 知识注入 | ✅ RAG 上下文 | ✅ 工具调用 | 静态知识用 RAG，动态数据用工具 |

- **Prompt 依赖管理**：
  - Prompt A 引用了 Prompt B 的输出格式，B 改了 A 会不会崩？
  - 解决：定义 Prompt 间的接口契约（输入输出 Schema），变更时自动检查依赖
  - 类似代码的 API 契约测试（Contract Testing）

- **Prompt 模块的独立测试**：
  - 每个 Prompt 模块可以独立运行评测
  - 组合后的 Prompt 也需要集成测试
  - 监控每个模块的 Token 消耗，发现膨胀及时优化

### 8. 提示词安全

- **Prompt Injection（提示注入）**：恶意用户试图覆盖系统指令
- **Jailbreak（越狱）**：绕过模型安全限制
- **防御策略**：输入过滤、指令隔离、输出检测

## 学习路线建议

1. 掌握基础提示技巧（角色设定、格式控制、示例提供）
2. 学习 CoT 等高级推理策略
3. 实践不同的提示词设计框架
4. 学习系统提示词的设计与优化
5. 了解提示词安全与防御策略

## 推荐资源

- 📄 [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- 📄 [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/abs/2305.10601)
- 📘 [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- 📘 [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
- 📘 [PromptLayer](https://promptlayer.com/) — Prompt 版本管理与 A/B 测试
- 📘 [Humanloop](https://humanloop.com/) — Prompt 管理、评估与监控
- 🎓 [DeepLearning.AI - ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/)
