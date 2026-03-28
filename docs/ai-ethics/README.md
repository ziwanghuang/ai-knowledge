# 🔐 AI 伦理与治理 — AI Ethics & Governance

## 概述

随着 AI 系统能力的快速增长，伦理与治理问题变得日益重要。AI 伦理关注技术对社会的影响，AI 治理则关注如何通过制度、法规和技术手段确保 AI 的安全、公平和可控。这不仅是技术问题，更是关乎 AI 长期发展的社会问题。

## 核心知识体系

### 1. AI 偏见与公平性

- **偏见来源**：
  - 训练数据偏见：数据采集不均衡、历史偏见固化
  - 标注偏见：标注者的主观判断引入偏差
  - 算法偏见：模型结构或优化目标导致的系统性偏差
  - 部署偏见：不同群体使用场景差异
- **偏见检测**：
  - 统计指标：人口统计均等（Demographic Parity）、机会均等（Equalized Odds）
  - 对比测试：不同群体输入的输出差异分析
  - Red Teaming：系统性的偏见探测
- **偏见缓解**：
  - 数据层面：数据增强、重采样、合成数据平衡
  - 模型层面：对抗训练、公平性约束
  - 输出层面：后处理校准、Guardrails 过滤

### 2. 可解释性（Explainability）

- **模型可解释性方法**：
  - **注意力可视化**：Attention Map 分析
  - **特征归因**：SHAP、LIME、Integrated Gradients
  - **概念激活向量（TCAV）**：用人类可理解的概念解释模型
  - **Chain-of-Thought 作为解释**：让模型展示推理过程
- **LLM 特有的可解释性挑战**：
  - 黑盒问题：大模型内部机制难以完全理解
  - 幻觉溯源：为什么模型会生成错误信息
  - 决策归因：Agent 为什么选择某个工具或策略
- **Mechanistic Interpretability**：
  - 神经元级别的功能分析
  - 电路发现（Circuit Discovery）
  - 稀疏自编码器（SAE）探测

### 3. AI 安全（AI Safety）

- **对齐问题（Alignment）**：
  - 目标对齐：确保 AI 的目标与人类意图一致
  - RLHF / DPO / Constitutional AI：当前主流对齐方法
  - Scalable Oversight：如何监督超越人类能力的 AI
  - Reward Hacking：AI 找到奖励函数的漏洞
- **灾难性风险**：
  - 失控风险：AI 系统超出人类控制
  - 滥用风险：恶意使用 AI 造成危害
  - 系统性风险：AI 在关键基础设施中的级联故障
- **安全评估**：
  - 能力评估（Capability Evaluation）
  - 危险能力检测（Dangerous Capability）
  - 安全基准测试

### 4. AI 法规与合规

| 法规 | 地区 | 核心要求 |
|------|------|---------|
| **EU AI Act** | 欧盟 | 风险分级监管，高风险 AI 需合规评估 |
| **生成式 AI 管理办法** | 中国 | 内容安全、数据合规、算法备案 |
| **AI Executive Order** | 美国 | 安全测试、透明度报告 |
| **GDPR** | 欧盟 | 数据隐私保护，用户知情权 |
| **个人信息保护法** | 中国 | 个人数据处理规范 |

- **合规实践**：
  - 算法备案与影响评估
  - 数据本地化存储
  - 用户知情同意机制
  - 审计追踪与日志留存
  - 内容安全审核机制

### 5. 负责任 AI（Responsible AI）

- **核心原则**：
  - 透明性（Transparency）：模型能力与局限性的公开
  - 问责制（Accountability）：明确 AI 决策的责任归属
  - 隐私保护（Privacy）：数据最小化、差分隐私
  - 包容性（Inclusivity）：确保不同群体都能受益
  - 可靠性（Reliability）：系统稳定性与一致性
- **实践框架**：
  - Microsoft Responsible AI Standard
  - Google AI Principles
  - Anthropic Responsible Scaling Policy
- **AI 伦理委员会**：组织内部的伦理审查机制

### 6. 版权与知识产权

- **训练数据版权**：使用受版权保护的数据训练模型的法律争议
- **AI 生成内容的版权归属**：AI 创作的内容是否受版权保护
- **开源模型许可证**：Apache 2.0、MIT、LLaMA License 等的区别与限制
- **商用合规**：模型商用时的许可证合规检查

## 学习路线建议

1. 了解 AI 伦理的基本概念与核心原则
2. 学习偏见检测与缓解的技术方法
3. 关注各国 AI 法规的最新动态
4. 理解 AI 安全与对齐的前沿研究
5. 在实际项目中落实负责任 AI 实践

## 推荐资源

- 📄 [On the Dangers of Stochastic Parrots](https://dl.acm.org/doi/10.1145/3442188.3445922) — AI 伦理经典论文
- 📘 [EU AI Act 全文](https://artificialintelligenceact.eu/)
- 📘 [Microsoft Responsible AI](https://www.microsoft.com/en-us/ai/responsible-ai)
- 📘 [Anthropic Research on AI Safety](https://www.anthropic.com/research)
- 📖 [《AI 伦理》](https://book.douban.com/subject/35559507/)
- 🎓 [Stanford HAI - AI Ethics](https://hai.stanford.edu/)
- 📘 [NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/ai-risk-management-framework)
