# 🧪 AI 测试与质量保障 — AI Testing & Quality Assurance

## 概述

AI 系统的非确定性特征使其测试与传统软件截然不同。LLM 的输出不可完全预测，Agent 的行为路径多样，这对质量保障提出了全新挑战。系统化的 AI 测试方法论是确保 AI 产品可靠上线的关键。

## 核心知识体系

### 1. LLM 输出测试

#### 1.1 确定性测试的挑战
- LLM 输出具有随机性（temperature > 0）
- 同一输入可能产生多种合理输出
- 输出质量难以用简单规则判断
- 模型更新后行为可能发生变化

#### 1.2 测试策略
- **基于断言的测试**：
  - 格式断言：输出是否符合预期格式（JSON、Markdown 等）
  - 包含/排除断言：输出是否包含/不包含特定内容
  - 长度断言：输出长度是否在合理范围内
  - 语义断言：输出语义是否与预期一致（用 Embedding 相似度判断）
- **LLM-as-Judge**：
  - 用另一个 LLM 评估输出质量
  - 评分维度：准确性、相关性、完整性、安全性
  - 评分标准设计（Rubric）
  - 多评委投票机制
- **参考答案对比**：
  - 与标准答案的语义相似度
  - BLEU、ROUGE、BERTScore
- **人工评估**：
  - 盲评（Blind Evaluation）
  - A/B 对比评估
  - Elo 评分系统（如 Chatbot Arena）

### 2. Prompt 回归测试

- **测试集构建**：
  - 覆盖核心场景的测试用例集
  - 边界情况（Edge Cases）收集
  - 对抗样本（Adversarial Examples）
- **回归测试流程**：
  ```
  Prompt 变更 → 运行测试集 → 对比基线结果
      → 质量评分 → 通过/回退决策
  ```
- **版本管理**：
  - Prompt 版本化（Git 管理）
  - 测试结果历史追踪
  - 性能趋势分析

### 3. Agent 端到端测试 ⭐

- **工作流测试**：
  - 单步工具调用正确性
  - 多步工作流完整性
  - 分支路径覆盖
  - 异常路径处理
- **工具调用测试**：
  - 工具选择准确性
  - 参数提取正确性
  - 工具调用顺序合理性
  - Mock 工具与真实工具切换
- **状态管理测试**：
  - 上下文传递正确性
  - 记忆读写一致性
  - 长对话状态保持
- **性能测试**：
  - 端到端延迟
  - Token 消耗量
  - 并发处理能力

### 4. 对抗测试（Red Teaming）⭐

- **Prompt Injection 测试**：
  - 直接注入：在用户输入中嵌入恶意指令
  - 间接注入：通过检索文档、工具返回值注入
  - 多轮对话注入：逐步引导模型偏离
- **Jailbreak 测试**：
  - 角色扮演攻击（DAN、AIM 等）
  - 编码绕过（Base64、Unicode）
  - 多语言绕过
  - 上下文窗口攻击
- **安全边界测试**：
  - 有害内容生成测试
  - 隐私信息泄露测试
  - 权限越界测试
- **自动化 Red Teaming**：
  - 用 LLM 自动生成攻击样本
  - Garak：LLM 安全扫描工具
  - 持续安全监控

### 5. RAG 系统测试

- **检索质量测试**：
  - Recall@K、Precision@K、MRR、NDCG
  - 检索结果相关性人工评估
- **生成质量测试**：
  - 忠实度（Faithfulness）：回答是否基于检索到的文档
  - 相关性（Relevancy）：回答是否切题
  - 幻觉检测：回答中是否包含文档中没有的信息
- **端到端测试**：
  - RAGAS 框架：自动化 RAG 评估
  - TruLens：RAG 应用追踪与评估
  - 自定义评估管道

### 6. 测试工具与框架

| 工具 | 用途 |
|------|------|
| **promptfoo** | Prompt 测试与评估框架，支持多模型对比 |
| **DeepEval** | LLM 应用测试框架，丰富的评估指标 |
| **RAGAS** | RAG 系统专用评估框架 |
| **Garak** | LLM 安全漏洞扫描 |
| **LangSmith** | LLM 应用追踪、测试与评估 |
| **Langfuse** | 开源 LLM 可观测性与评估 |
| **Pytest + 自定义断言** | 基础测试框架 + AI 特定断言 |

### 7. 评测金字塔与分层设计 ⭐

> "你无法改进你无法度量的东西"——生产阶段需要回答：整体准确率、哪类问题好/差、改动有没有退化、幻觉率趋势、用户满意度。

```
评测金字塔（从底到顶）：
  单元测试（秒级）→ 离线评测集（分钟级）→ 回归评测（CI/CD 门禁）
  → A/B 测试 → 线上评测（用户满意度、NPS、业务指标）
```

- **单元测试**：工具调用正确性、输出格式校验、Prompt 渲染测试
- **离线评测集**：覆盖 FAQ/工具调用/安全拒绝等场景的标注数据集（YAML/JSON 格式）
- **RAGAS 评测维度详解**：
  | 维度 | 含义 | 衡量方式 |
  |------|------|---------|
  | Faithfulness（忠实度） | 回答是否被检索文档支撑 | 回答中每个声明是否在上下文中有依据 |
  | Answer Relevancy（答案相关性） | 回答是否切题 | 回答是否回答了用户问题 |
  | Context Precision（上下文精准度） | 检索到的文档是否真的相关 | 检索结果中相关文档的比例 |
  | Context Recall（上下文召回率） | 相关文档是否都被检索到 | 标准答案需要的信息是否都在检索结果中 |
  | Harmfulness（有害性） | 回答是否包含有害内容 | 安全分类器 + LLM 审核 |
- **回归评测门禁**：faithfulness ≥ 0.85、answer_relevancy ≥ 0.80、退化用例 ≤ 5 个、安全测试 100% 通过

### 8. Bad Case 管理闭环 ⭐

```
发现 Bad Case（线上日志/用户反馈/评测发现）
  → 结构化记录（问题、实际回答、期望回答、根因、严重性、Trace ID、影响面）
  → 根因分析（基于 Trace 定位：检索没搜到？搜到了被过滤？LLM 没用好？）
  → 修复（补充文档 / 调整切片 / 修改 Prompt / 调整检索权重）
  → 转化为评测用例（加入回归评测集，确保以后不再出现）
  → 验证 & 上线
```

### 9. 可测试性设计（Design for Testability）⭐

> 当前覆盖了"怎么测"，但同样重要的是"怎么设计才好测"。可测试性应该是架构设计的一等公民。

- **确定性 Seam 设计**：
  - 在 Agent 流程中预留确定性注入点，方便测试时替换 LLM 输出
  - 将 LLM 调用抽象为接口，测试时可注入 Mock 实现
  ```python
  # 伪代码：可测试的 Agent 设计
  class LLMProvider(Protocol):
      async def complete(self, messages: list) -> str: ...
  
  class MockLLM(LLMProvider):
      def __init__(self, responses: dict):
          self.responses = responses  # 预设的输入→输出映射
      async def complete(self, messages):
          return self.responses.get(key(messages), "default")
  ```

- **Mock LLM 策略**：
  | 策略 | 适用场景 | 特点 |
  |------|---------|------|
  | 录制回放（Record & Replay） | 回归测试 | 录制真实 LLM 响应，回放时零成本 |
  | 规则引擎 Mock | 单元测试 | 基于规则返回预设响应，速度快 |
  | 小模型替代 | 集成测试 | 用本地小模型替代云端大模型 |
  | 固定响应 Mock | 流程测试 | 固定返回特定响应，测试下游逻辑 |

- **快照测试（Snapshot Testing）**：
  - 记录 Agent 的完整执行轨迹（每步的输入、输出、工具调用、状态变化）
  - 后续变更时对比快照，发现意外的行为变化
  - 类似前端的 Snapshot Testing，但应用于 Agent 行为

- **混沌工程（Chaos Engineering）**：
  - 主动注入故障测试系统韧性：
    | 故障类型 | 注入方式 | 验证目标 |
    |---------|---------|----------|
    | LLM 超时 | 延迟注入 | 降级策略是否生效 |
    | 工具调用失败 | 返回错误码 | 重试和补偿机制 |
    | 网络抖动 | 随机丢包 | 断点续传能力 |
    | 向量数据库不可用 | 断开连接 | 备用检索策略 |
    | Token 超限 | 注入超长上下文 | 上下文管理策略 |

- **可重放性（Reproducibility）**：
  - 记录完整的请求上下文（包括随机种子、模型版本、Prompt 版本、检索结果）
  - 使问题可复现：给定相同的输入和上下文，能重现相同的问题
  - 对于 temperature > 0 的场景，记录 seed 参数

- **测试数据管理**：
  - 评测集的版本管理（与代码/Prompt 版本关联）
  - 标注流程标准化（标注指南、质量审核、一致性检查）
  - 评测集覆盖度分析（哪些场景缺少测试用例）

### 10. AI 系统可调试性（Debuggability）⭐

> 可观测性告诉你"发生了什么"，可调试性帮你回答"为什么发生"。

- **Agent 决策归因**：
  - 不只是看 Trace，而是理解 LLM 的"思考过程"
  - 记录 LLM 的 Chain-of-Thought 输出（即使不展示给用户）
  - 对比不同决策路径：为什么选了工具 A 而不是工具 B？

- **Prompt 调试**：
  | 问题 | 调试方法 |
  |------|----------|
  | 哪句话导致了错误输出？ | 逐步删减 Prompt 内容，定位影响因子 |
  | Few-shot 示例是否有效？ | 对比有/无示例的输出差异 |
  | System Prompt 冲突？ | 检查多条指令是否存在矛盾 |
  | 上下文干扰？ | 对比不同 RAG 结果下的输出变化 |

- **检索调试**：
  - 为什么这个文档没被检索到？
    - Embedding 相似度是否足够？（计算 Query 与文档的余弦相似度）
    - 是否被元数据过滤掉了？（检查过滤条件）
    - 索引是否包含该文档？（检查索引状态）
    - 切片策略是否合理？（检查文档被切成了什么样）
  - 检索调试工具：可视化检索结果的相似度分布、展示每个过滤步骤的结果

- **交互式调试工具**：
  - 类似传统代码调试，在 Agent 执行的每一步设断点
  - 查看每步的完整状态（上下文、工具参数、LLM 输入输出）
  - 修改中间状态后重跑后续步骤
  - 工具推荐：LangSmith Playground、Langfuse Trace 回放

- **根因分析自动化**：
  ```
  Bad Case 输入 → 自动化根因分析流水线：
    1. 检索阶段：检索结果是否包含正确答案？→ 否 → 检索问题
    2. 重排阶段：正确文档是否被重排到前列？→ 否 → 重排问题
    3. 生成阶段：Prompt 中是否包含正确信息？→ 是但回答错 → LLM/Prompt 问题
    4. 工具阶段：工具是否被正确调用？→ 否 → 工具描述/路由问题
  ```

- **Playground 与生产环境一致性**：
  - Playground 调好的效果上线后不一样的常见原因：
    | 差异点 | 说明 |
    |--------|------|
    | 上下文不同 | Playground 用固定上下文，生产中上下文动态变化 |
    | 模型版本 | Playground 用最新模型，生产中可能是固定版本 |
    | 并发影响 | 高并发下 API 行为可能不同（限流、排队） |
    | 数据差异 | Playground 用测试数据，生产中是真实数据 |
  - 解决：确保 Playground 环境与生产环境配置一致，支持"生产回放"模式

### 11. CI/CD 中的 AI 测试

- **自动化测试管道**：
  ```
  代码/Prompt 提交 → 单元测试 → Prompt 回归测试
      → Agent 集成测试 → 安全测试 → 性能测试
      → 质量门禁（Quality Gate）→ 部署
  ```
- **质量门禁设计**：
  - 最低通过率阈值
  - 关键场景必须通过
  - 安全测试零容忍
- **测试成本控制**：
  - 分层测试策略（快速测试 → 深度测试）
  - 增量测试（只测试受影响的部分）
  - 缓存测试结果

## 学习路线建议

1. 理解 AI 测试与传统软件测试的差异
2. 学习 LLM-as-Judge 评估方法
3. 实践 Prompt 回归测试（推荐 promptfoo）
4. 掌握 Agent 端到端测试策略
5. 学习 Red Teaming 方法论
6. 构建 CI/CD 中的自动化 AI 测试管道

## 推荐资源

- 📘 [promptfoo](https://github.com/promptfoo/promptfoo) — Prompt 测试框架
- 📘 [DeepEval](https://github.com/confident-ai/deepeval) — LLM 测试框架
- 📘 [RAGAS](https://github.com/explodinggradients/ragas) — RAG 评估框架
- 📘 [Garak](https://github.com/leondz/garak) — LLM 安全扫描
- 📄 [Red Teaming Language Models](https://arxiv.org/abs/2209.07858)
- 📄 [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)
- 🎓 [DeepLearning.AI - Quality and Safety for LLM Applications](https://www.deeplearning.ai/short-courses/)
