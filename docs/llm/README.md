# 🧠 LLM — 大语言模型

## 概述

大语言模型（Large Language Model, LLM）是基于 Transformer 架构、通过海量文本数据训练而成的深度学习模型，具备强大的自然语言理解与生成能力。LLM 是当前 AI 领域最核心的基础技术之一。

## 核心知识体系

### 1. 基础原理

- **Transformer 架构**：Self-Attention 机制、Multi-Head Attention、位置编码（Positional Encoding）
- **预训练范式**：自回归（Autoregressive）、掩码语言模型（Masked LM）、Seq2Seq
- **Scaling Law**：模型规模、数据量、计算量之间的幂律关系
- **涌现能力（Emergent Abilities）**：随模型规模增大而突然出现的能力（如 CoT 推理）
- **长上下文处理**：RoPE、ALiBi、Flash Attention、KV Cache 优化

### 2. 主流模型

| 模型系列 | 开发者 | 特点 |
|---------|--------|------|
| GPT 系列 | OpenAI | 自回归生成，闭源商用 |
| Claude 系列 | Anthropic | 强调安全对齐，长上下文 |
| LLaMA 系列 | Meta | 开源，社区生态丰富 |
| Gemini 系列 | Google | 原生多模态，超长上下文 |
| Qwen 系列 | 阿里 | 中文能力强，开源 |
| DeepSeek 系列 | DeepSeek | MoE 架构，高性价比 |

### 3. 预训练数据工程

- **数据清洗**：去除低质量、重复、有害内容
- **数据去重**：MinHash、SimHash 等近似去重方法
- **数据配比**：不同语言、领域、格式数据的混合比例策略
- **数据质量评估**：困惑度过滤、分类器过滤、人工抽检
- **合成数据**：用大模型生成高质量训练数据

### 4. 微调技术

#### 4.1 对齐方法
- **SFT（Supervised Fine-Tuning）**：指令微调
- **RLHF（Reinforcement Learning from Human Feedback）**：基于人类反馈的强化学习对齐
- **DPO（Direct Preference Optimization）**：直接偏好优化，无需训练 Reward Model
- **GRPO（Group Relative Policy Optimization）**：DeepSeek 提出，无需 Reward Model，组内相对排序优化

#### 4.2 高效微调（PEFT）
- **LoRA（Low-Rank Adaptation）**：低秩适配，最主流的高效微调方法
- **QLoRA**：4bit 量化 + LoRA，单卡微调大模型的利器
- **AdaLoRA**：自适应分配 LoRA 秩
- **IA3**：通过抑制和放大内部激活的注入适配器
- **Adapter**：在 Transformer 层间插入小型适配模块
- **Prompt-Tuning**：冻结模型参数，只训练软提示（Soft Prompt）
- **P-Tuning v1 / v2**：连续提示嵌入优化
- **Prefix-Tuning**：在每层添加可学习前缀向量
- **PEFT 方法选择策略**：根据数据量、计算资源、任务类型选择

#### 4.3 全参数微调工程
- **显存优化技术**：梯度检查点（Gradient Checkpointing）、梯度累积（Gradient Accumulation）、混合精度训练（FP16/BF16）
- **DeepSpeed 分布式训练**：ZeRO-1/2/3 优化器，不同阶段的显存优化策略
- **FSDP（Fully Sharded Data Parallel）**：PyTorch 原生分布式方案

#### 4.4 模型蒸馏
- **知识蒸馏原理**：Teacher-Student 框架
- **大模型蒸馏到小模型**：如 70B → 7B
- **蒸馏数据生成与质量控制**

#### 4.5 微调框架实战 ⭐
| 框架 | 特点 |
|------|------|
| Hugging Face Transformers + PEFT | 最通用的微调组合 |
| LLaMA-Factory | Web UI 可视化微调，支持多种方法和模型 |
| ms-Swift（魔搭 Swift） | 阿里开源快速微调框架 |
| Unsloth | 2x-5x 微调加速，显存减半 |
| Axolotl | 配置化训练，灵活度高 |
| FastChat | 对话模型训练与部署 |

#### 4.6 微调工程实践
- **训练环境搭建**：CUDA、PyTorch、NCCL
- **超参数调优**：学习率、batch size、训练轮次、warmup、LoRA rank 选择
- **训练监控**：Loss 曲线分析、WandB / TensorBoard 实验追踪
- **模型合并与打包**：LoRA 权重合并、GGUF 导出

### 5. 推理与部署

#### 5.1 模型压缩
- **量化技术**：PTQ（训练后量化）/ QAT（量化感知训练），格式：GPTQ、AWQ、GGUF
- **剪枝（Pruning）**：结构化/非结构化剪枝，减少冗余参数

#### 5.2 推理框架
| 框架 | 特点 |
|------|------|
| vLLM | PagedAttention 高吞吐推理，生产级首选 |
| TGI | Hugging Face 推理服务 |
| TensorRT-LLM | NVIDIA 优化推理引擎 |
| llama.cpp | 纯 C/C++ 实现，CPU/边缘端推理利器 |
| Ollama | 本地大模型一键运行，开发调试友好 |

#### 5.3 推理优化技术
- **KV Cache 优化**：PagedAttention、连续批处理
- **投机解码（Speculative Decoding）**：小模型草稿 + 大模型验证加速
- **分布式推理**：张量并行、流水线并行
- **推理框架对比与选型指南**

### 6. 评估与基准

- **通用基准**：MMLU、HellaSwag、ARC、TruthfulQA
- **中文基准**：C-Eval、CMMLU
- **代码能力**：HumanEval、MBPP
- **数学能力**：GSM8K、MATH
- **对话能力**：MT-Bench、Chatbot Arena
- **评估方法**：LLM-as-Judge、Red Teaming（对抗测试）

## 学习路线建议

1. 理解 Transformer 架构与注意力机制
2. 了解预训练与微调的基本流程
3. 动手实践 LoRA 微调一个开源模型（推荐 LLaMA-Factory）
4. 学习模型量化与部署优化（推荐 vLLM + Ollama）
5. 掌握全参数微调与分布式训练
6. 关注前沿论文与技术趋势

## 推荐资源

- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- 📄 [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- 📄 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- 📘 [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers)
- 📘 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- 📘 [DeepSpeed 文档](https://www.deepspeed.ai/)
- 🎓 [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
