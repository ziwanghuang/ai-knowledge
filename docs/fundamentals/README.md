# 📐 基础理论 — ML / DL / 概率统计 / 认知科学

## 概述

基础理论是所有 AI 技术的底层根基。扎实的理论基础能帮助你理解 LLM 为什么有效、Agent 为什么能做决策、强化学习如何驱动对齐训练。这些知识不会过时，是长期技术竞争力的核心。

## 核心知识体系

### 1. 机器学习基础

#### 1.1 监督学习
- **分类**：逻辑回归、SVM、决策树、随机森林、XGBoost
- **回归**：线性回归、多项式回归、岭回归、Lasso
- **损失函数**：交叉熵、MSE、Hinge Loss
- **正则化**：L1/L2 正则化、Dropout、Early Stopping
- **模型评估**：准确率、精确率、召回率、F1、AUC-ROC

#### 1.2 无监督学习
- **聚类**：K-Means、DBSCAN、层次聚类
- **降维**：PCA、t-SNE、UMAP
- **生成模型**：GMM、VAE、GAN

#### 1.3 强化学习（RL）⭐ Agent 核心
- **马尔可夫决策过程（MDP）**：状态、动作、奖励、转移概率
- **值函数方法**：Q-Learning、SARSA、DQN
- **策略梯度方法**：REINFORCE、PPO、A3C
- **模型预测控制（Model-Based RL）**
- **多臂老虎机（Bandit）问题**
- **逆强化学习（IRL）**：从专家行为推断奖励函数
- **离线强化学习（Offline RL）**：从历史数据中学习策略
- **与 LLM 对齐的关系**：RLHF、PPO 在 LLM 训练中的应用

#### 1.4 元学习（Meta-Learning）
- 学会学习，Few-shot 场景下的快速适应
- MAML、Prototypical Networks
- 与 In-Context Learning 的联系

### 2. 深度学习基础

- **神经网络基础**：前馈网络、反向传播、激活函数（ReLU/GELU/SiLU）、优化器（Adam、SGD、AdamW）
- **CNN**：卷积、池化、残差连接，图像感知相关
- **RNN / LSTM / GRU**：序列建模，理解 Transformer 之前的序列处理方式
- **Transformer 架构** ⭐ 重中之重：
  - Self-Attention 机制
  - Multi-Head Attention
  - 位置编码（Sinusoidal、RoPE、ALiBi）
  - KV Cache
  - Flash Attention
  - 长上下文处理
- **扩散模型（Diffusion）**：DDPM、Stable Diffusion，多模态生成的基础

### 3. 概率与统计基础

- **贝叶斯推理与贝叶斯网络**：先验、后验、似然
- **概率图模型（PGM）**：有向图模型、无向图模型
- **蒙特卡洛方法**：MCMC、重要性采样
- **信息论**：熵、KL 散度、互信息、交叉熵
- **最大似然估计（MLE）与最大后验估计（MAP）**

### 4. 认知科学与决策理论

- **有限理性（Bounded Rationality）**：Herbert Simon 的理论，Agent 决策的理论基础
- **认知架构**：SOAR、ACT-R，经典 AI Agent 的认知模型
- **效用理论与期望效用最大化**
- **博弈论基础**：纳什均衡、多智能体博弈，Multi-Agent 系统的理论基础

## 学习路线建议

1. 从机器学习基础入手（推荐吴恩达课程）
2. 深入深度学习，重点理解 Transformer
3. 学习强化学习基础，理解 PPO 等算法（与 RLHF 直接相关）
4. 按需补充概率统计和信息论知识
5. 了解认知科学，加深对 Agent 决策的理解

## 推荐资源

- 🎓 [吴恩达 - Machine Learning（Coursera）](https://www.coursera.org/learn/machine-learning)
- 🎓 [李宏毅 - 机器学习](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
- 📖 [《深度学习》花书](https://www.deeplearningbook.org/) — Goodfellow et al.
- 📖 [《强化学习导论》](http://incompleteideas.net/book/the-book-2nd.html) — Sutton & Barto
- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- 🎓 [David Silver - Reinforcement Learning](https://www.davidsilver.uk/teaching/)
- 📖 [《统计学习方法》](https://book.douban.com/subject/33437381/) — 李航
