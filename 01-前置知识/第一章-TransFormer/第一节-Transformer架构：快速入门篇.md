# 第一章 Transformer 架构：快速入门篇

> 适用人群：AI Infra 工程师、零基础学员
> 学习目标：吃透 Transformer 原理，为后续推理引擎、分布式训练、KV Cache 优化等 Infra 主题打好地基

---

## 📑 目录

- [1. 为什么 AI Infra 工程师必须懂 Transformer](#1-为什么-ai-infra-工程师必须懂-transformer)
  - [1.1 一个朴素的问题](#11-一个朴素的问题)
  - [1.2 Transformer 统治了哪些场景](#12-transformer-统治了哪些场景)
  - [1.3 Infra 视角下的 Transformer 关键概念](#13-infra-视角下的-transformer-关键概念)
- [2. Transformer 网络结构全貌](#2-transformer-网络结构全貌)
  - [2.1 一张图看懂整体结构](#21-一张图看懂整体结构)
  - [2.2 三大核心组件](#22-三大核心组件)
  - [2.3 输入输出的形状（Infra 工程师的必备技能）](#23-输入输出的形状infra-工程师的必备技能)
- [3. 位置编码](#3-位置编码)
  - [3.1 为什么需要位置编码](#31-为什么需要位置编码)
  - [3.2 原始 Transformer：正弦位置编码（Sinusoidal PE）](#32-原始-transformer正弦位置编码sinusoidal-pe)
  - [3.3 旋转位置编码（RoPE）—— 现代 LLM 的标配](#33-旋转位置编码rope--现代-llm-的标配)
  - [3.4 其他位置编码一览](#34-其他位置编码一览)
  - [3.5 Infra 视角](#35-infra-视角)
- [4. Self-Attention 机制](#4-self-attention-机制)
  - [4.1 为什么需要 Attention](#41-为什么需要-attention)
  - [4.2 Q、K、V 是什么](#42-qkv-是什么)
  - [4.3 Self-Attention 的完整计算过程](#43-self-attention-的完整计算过程)
  - [4.4 完整公式](#44-完整公式)
  - [4.5 Multi-Head 的意义](#45-multi-head-的意义)
  - [4.6 Infra 视角：Self-Attention 的瓶颈](#46-infra-视角self-attention-的瓶颈)
- [5. 前馈网络（FFN）](#5-前馈网络ffn)
  - [5.1 升维-激活-降维 三步详解](#51-升维-激活-降维-三步详解)
  - [5.2 激活函数](#52-激活函数)
  - [5.3 FFN 的参数占比](#53-ffn-的参数占比)
  - [5.4 Infra 视角](#54-infra-视角)
- [6. LayerNorm 与残差连接](#6-layernorm-与残差连接)
  - [6.1 残差连接（Residual Connection）](#61-残差连接residual-connection)
  - [6.2 LayerNorm](#62-layernorm)
  - [6.3 Pre-Norm vs Post-Norm](#63-pre-norm-vs-post-norm)
  - [6.4 RMSNorm](#64-rmsnorm)
  - [6.5 Infra 视角](#65-infra-视角)
- [7. 完整的 Transformer Decoder Block](#7-完整的-transformer-decoder-block)
  - [7.1 伪代码实现](#71-伪代码实现)
  - [7.2 完整 LLM 推理流程](#72-完整-llm-推理流程)
  - [7.3 形状流转一览（推理时）](#73-形状流转一览推理时)
  - [7.4 KV Cache：推理时为什么需要它](#74-kv-cache推理时为什么需要它)
- [8. 从 Transformer 到 LLM：自回归生成](#8-从-transformer-到-llm自回归生成)
  - [8.1 什么是自回归生成](#81-什么是自回归生成)
  - [8.2 推理的两种模式](#82-推理的两种模式)
  - [8.3 采样策略](#83-采样策略)
  - [8.4 停止条件](#84-停止条件)
  - [8.5 推理性能指标](#85-推理性能指标)
  - [8.6 优化全景](#86-优化全景)
- [9. 本章小结](#9-本章小结)
  - [9.1 关键概念速查表](#91-关键概念速查表)
  - [9.2 给后续章节的"钩子"](#92-给后续章节的钩子)
- [10. 思考题](#10-思考题)

---

## 1. 为什么 AI Infra 工程师必须懂 Transformer

### 1.1 一个朴素的问题

作为一个 AI Infra 工程师，你每天打交道的东西可能是：
- GPU 显存调度
- 推理服务（vLLM、TGI、SGLang）
- KV Cache 管理
- Tensor Parallel / Pipeline Parallel
- 量化、投机解码、PagedAttention

这些都是"上层建筑"。但无论你做哪一项优化，都绕不开一个事实——

> **你优化的对象，99% 都是 Transformer（或它的变体）。**

不理解 Transformer 内部到底在算什么，你就只能机械地调参、抄配置、出问题靠玄学。懂了 Transformer，你看到 vLLM 里 `block_size=16`、看到 FlashAttention 的 `causal mask`、看到 `num_kv_heads` 为什么能省显存——都会立刻明白为什么这么设计。

### 1.2 Transformer 统治了哪些场景

| 场景 | 代表模型 | 主要架构 |
|---|---|---|
| 通用对话 / 写作 | GPT-4、Claude、Qwen、DeepSeek | Decoder-only |
| 翻译 / 摘要 | 原始 Transformer | Encoder-Decoder |
| 图文理解 | LLaVA、GPT-4V | Decoder + ViT Encoder |
| 图像生成 | DiT、Stable Diffusion 3 | Diffusion + Transformer |
| 多模态语音 | Whisper | Encoder-Decoder |
| 代码 / Agent | Codex、Cursor 后端 | Decoder-only + Tool |

> **结论**：所有主流大模型的"骨架"几乎都是 Transformer。AI Infra 的所有优化，本质上都是在优化 Transformer 的某一部分。

### 1.3 Infra 视角下的 Transformer 关键概念

理解 Transformer 不仅仅是"知道 attention 公式"，更重要的是知道：

- **计算密集 vs. 访存密集**：Attention 是访存密集（memory-bound），FFN 是计算密集（compute-bound），这决定了它们适合的优化策略不同。
- **激活值（Activation）有多大**：在 `batch × seq_len × hidden_dim` 下，每一层会产生多大中间结果，决定了显存占用。
- **KV Cache 是什么**：自回归推理时为什么必须缓存 K、V，缓存多大、放在哪。
- **为什么 Decoder 比 Encoder 慢**：因为有 causal mask，每次只能看历史。
- **LayerNorm 在哪**：它和残差连接一起，决定了 pre-norm 还是 post-norm，影响数值稳定性。

这些概念，本章都会讲清楚。

---

## 2. Transformer 网络结构全貌

### 2.1 一张图看懂整体结构

```
输入序列:  ["我", "爱", "学习"]  (token IDs: [23, 1024, 5678])
                    │
                    ▼
        ┌───────────────────────┐
        │  Token Embedding      │  把每个 token 转成向量
        │  + 位置编码 (RoPE)    │  加入位置信息
        └───────────────────────┘
                    │
                    │  x₀
                    ▼
╔════════════════════════════════════════════╗
║  Transformer Decoder Block ×N   (N = 32)   ║
║                                            ║
║    xᵢ ──────────────────────┐              ║
║    │                          │  (残差)     ║
║    ▼                          │             ║
║  ┌──────────┐                 │             ║   ← Pre-Norm
║  │ RMSNorm  │                 │             ║
║  └──────────┘                 │             ║
║    │                          │             ║
║    ▼                          │             ║
║  ┌──────────────────────┐     │             ║
║  │ Masked Multi-Head    │     │             ║
║  │ Self-Attention       │     │             ║   ← Causal Mask
║  │ (causal mask)        │     │             ║
║  └──────────────────────┘     │             ║
║    │                          │             ║
║    └─────────►(+) ◄───────────┘             ║
║              │                              ║
║              ▼  xᵢ₊₁                        ║
║    xᵢ₊₁ ─────────────────────┐             ║
║    │                           │  (残差)    ║
║    ▼                           │            ║
║  ┌──────────┐                  │            ║   ← Pre-Norm
║  │ RMSNorm  │                  │            ║
║  └──────────┘                  │            ║
║    │                           │            ║
║    ▼                           │            ║
║  ┌────────────────┐            │            ║
║  │ FFN (MLP)      │            │            ║
║  │ 升维 → 激活 → 降维│           │            ║
║  └────────────────┘            │            ║
║    │                           │            ║
║    └─────────►(+) ◄────────────┘            ║
║              │                              ║
║              ▼  xᵢ₊₂  (本 Block 输出)        ║
╚════════════════════════════════════════════╝
                    │
                    ▼
        ┌───────────────────────┐
        │  Final RMSNorm        │   ← 容易被遗漏的关键一步！
        └───────────────────────┘   ← Pre-Norm 架构标配
                    │
                    ▼
        ┌───────────────────────┐
        │  LM Head              │   线性映射到词表
        │  (vocab_size 维)      │   例如 128000
        └───────────────────────┘
                    │
                    ▼
              logits [B, S, vocab_size]
                    │
                    ▼
            softmax → 下一 token 概率
```

> 💡 **备注**：
> - **位置编码**：Self-Attention 本身是"置换不变"的，看不出 token 的先后顺序，于是需要解决"区分'猫吃鱼'和'鱼吃猫'"的问题，做法是在词向量上**加**入一个代表"第几位"的位置向量（RoPE 则通过旋转 Q/K 注入），结果让模型能感知顺序、理解语法、捕捉长距离依赖。
> - **Transformer Block ×N**：就像"千层饼"——把"看别人 + 自己消化"这套动作重复 N 次（LLaMA 是 32 次），每重复一次就加深一层理解，层数越多模型越"深"。
> - **LayerNorm（Pre-Norm）**：网络越深，数据就越容易"跑偏"（数值一会儿爆炸、一会儿消失），深层根本训不出来，于是需要在每一层解决"数据跑偏"的问题，做法是在进入 Attention/FFN **之前**先做一次"校准"（LayerNorm 标准化），结果是 LLaMA、GPT 这些几十层、上百层的大模型才能稳稳地训练出来。
> - **Multi-Head Self-Attention**：单个 head（一个专家）想同时搞懂语法、指代、搭配、距离等多种关系，结果每样都吃不透，于是需要解决"一个专家看不全"的问题，做法是把"一个大专家"拆成多个"小专家"（head），每个 head 专注一种关系（一个看语法、一个看指代、一个看搭配……），并行算完再汇总，结果是模型能从多个维度全面理解语言，效果远超单 head。
> - **FFN（前馈网络）**：每个 token 听完别人的意见（Attention）后，还得"自己消化"，但 D 维的"脑子"装不下太多东西，于是需要解决"消化不动"的问题，做法是把向量先"撑大"到 4 倍 → 中间做激活函数"非线性过滤" → 再"压回"原维度，每个 token 独立处理，结果是模型有了"独立思考"环节，还顺带储存了大量知识（占总参数的 2/3）。

### 2.2 三大核心组件

Transformer Block 由三个关键部分组成：

| 组件 | 作用 | 直觉类比 |
|---|---|---|
| **Self-Attention** | 让每个 token 看其他 token，决定"我应该关注谁" | 一群人在开会，每个人都根据别人发言调整自己的观点 |
| **FFN（前馈网络）** | 对每个 token 单独做非线性变换 | 每个人独处时消化刚才听到的信息 |
| **残差 + LayerNorm** | 让深层网络能训练 | 防止"听歪了"，保留原始信号 |

每一层（Block）都做两件事：
1. **Self-Attention**：跨 token 信息混合
2. **FFN**：单 token 内部信息加工

两者交替进行，多层堆叠，让模型学到丰富的语言表示。

### 2.3 输入输出的形状（Infra 工程师的必备技能）

#### 为什么 Infra 工程师必须懂"形状"

AI Infra 工程师的日常工作——显存估算、性能分析、并行切分——都依赖一个前提：**精确知道每个组件的输入输出形状**。

不熟悉形状会寸步难行：
- 估显存 = 数形状元素个数 × 字节数
- 并行切分 = 找到能"切"的维度（一般是 batch 或 seq 维）
- 算子优化 = 看清形状才能选对 kernel

> 不熟悉张量形状，就相当于 Infra 工程师的"文盲"——这就是为什么把它列为必备技能。

#### 要掌握哪些关键形状

只需建立**两个层面**的形状直觉：

1. **Block 整体**：输入输出形状变不变？
2. **Self-Attention 内部**：Q、K、V 怎么 reshape？

#### 用一个具体例子走一遍

假设一个典型输入（LLaMA-7B）：
- `batch_size = B = 1`
- `seq_len = S = 2048`
- `hidden_dim = D = 4096`

> 💡 **备注**：B 是一次处理几句话（决定吞吐）、S 是一句话多长（决定上下文）、D 是每个 token 多"厚"（决定模型容量）——三者组成 `[B, S, D]`，就是后续所有 Infra 优化（显存、并行、量化）的共同语言。

**层面 1：Block 整体形状不变**

```
输入:  [B, S, D]   =   [1, 2048, 4096]
  │ Transformer Block（Attention + FFN + 残差）
输出:  [B, S, D]   =   [1, 2048, 4096]   ← 形状完全不变
```

> 💡 **关键洞察**：Block 不改变形状，只是把每个位置的表示"换了一种编码方式"。
> 所以 Block 能像积木一样堆 N 层——每层输入输出形状一致，无缝衔接。

> 💡 **备注**：
> - **形状不变意味着什么**：形状不变 = "换手册不换人"——2048 个员工（对应 S）不变、每人 4096 页手册厚度（对应 D）不变，唯一升级的是手册里的内容（向量更懂上下文）。
> - **这样设计有什么作用**：(1) 让 Block 能像积木一样无限堆叠 N 层；(2) 让显存能用 `B×S×D×2` 公式精确估算；(3) 让一个 GPU kernel 可以写一次跑 N 层复用——这是 Transformer 能加深、能让 Infra 优化的根本前提。

**层面 2：Self-Attention 内部 reshape**

**为什么 Block 内部要"变形"一次**：
Block 整体形状不变（[B, S, D]），但内部还要做跨 token 的信息混合——一个 4096 维的"全能向量"想同时处理语法、指代、搭配等多种关系会顾此失彼，于是需要解决"一个专家看不全"的问题，做法是把 D 维向量拆成 `num_heads` 个小向量（每个 `head_dim` 维），让多个 head 各管一头。

**怎么 reshape**：
进入 Attention 后，Q、K、V 都被切成多头：

```
Q: [B, S, D]            →  reshape  →  [B, num_heads, S, head_dim]
K: [B, S, D]            →  reshape  →  [B, num_heads, S, head_dim]
V: [B, S, D]            →  reshape  →  [B, num_heads, S, head_dim]
```

代入数字（LLaMA-7B）：

```
Q: [1, 2048, 4096]  →  [1, 32, 2048, 128]
K: [1, 2048, 4096]  →  [1, 32, 2048, 128]
V: [1, 2048, 4096]  →  [1, 32, 2048, 128]
```

其中维度关系：`D = num_heads × head_dim = 32 × 128 = 4096` ✓

**reshape 之后的副产物**：
多头并行算 Attention 时，会产生一个中间张量——注意力分数矩阵：

```
scores: [B, num_heads, S, S]   =   [1, 32, 2048, 2048]
```

这就是 Attention 显存爆炸的根源（也是后续 FlashAttention、PagedAttention 优化的重点）。

**最后再 reshape 回去**：
```
out: [B, num_heads, S, head_dim]  →  reshape  →  [B, S, D]   = [1, 2048, 4096]
```

**结果**：经过这一次"切—算—拼"的来回，Block 在保持整体形状不变的前提下，让多个 head 各展所长完成多角度理解语言的任务。

#### 掌握后你能做什么

| 能做的事 | 例子 |
|---|---|
| **一眼看懂**任何 Transformer 形状 | 看 model config 就知道每层长啥样 |
| **快速估算显存** | Block 输出：[1,2048,4096] × 2B = 16 MB / 层，32 层 ≈ 512 MB |
| **识别可并行维度** | B、S、num_heads 都可独立切分 |
| **理解 Infra 优化本质** | 所有优化都在切这些维度（TP、PP、SP） |

---

## 3. 位置编码

### 3.1 为什么需要位置编码

Self-Attention 有一个特性：**它是置换不变的（permutation invariant）**。

把"我爱你"打乱成"你爱我"，Attention 算出的输出完全一样（只是对应位置变了）。

但语言是有顺序的！没有位置信息，模型就分不清：
- "猫吃鱼" vs "鱼吃猫"
- "不……很……" vs "很……不……"

所以必须**显式注入位置信息**。

### 3.2 原始 Transformer：正弦位置编码（Sinusoidal PE）

公式：

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

特点：
- 每个维度用不同频率的 sin/cos
- 理论上可以外推到比训练更长的序列
- 实际效果一般，现代 LLM 已不用

### 3.3 旋转位置编码（RoPE）—— 现代 LLM 的标配

RoPE 是目前最主流的位置编码，GPT-NeoX、LLaMA、Qwen、DeepSeek 全部在用。

**核心思想**：把位置信息编码成向量的**旋转角度**。

对于第 $m$ 个位置的 query 向量 $q$，RoPE 把它看作二维平面上的向量，逐对做旋转：

```text
q'_i = q_i · e^(i · m · θ_i)
```

其中 $\theta_i = 10000^{-2i/d}$。

直观理解：
- 每个 token 的 Q、K 向量被旋转了不同角度
- 两个 token 的注意力分数（Q·K）只取决于它们的**相对位置差**
- 自然支持长度外推

RoPE 的优势：
1. **相对位置编码**：Q·K 自然包含相对位置信息
2. **长度外推**：可以处理比训练更长的序列
3. **高效计算**：可以写成逐元素乘法，不需要额外参数

### 3.4 其他位置编码一览

| 编码 | 代表模型 | 特点 |
|---|---|---|
| Sinusoidal | 原始 Transformer | 固定，不能学习 |
| Learned PE | BERT、GPT-2 | 可学习，但有长度限制 |
| **RoPE** | LLaMA、Qwen、DeepSeek | 当前主流 |
| ALiBi | BLOOM | 通过注意力 bias 注入位置 |
| YaRN | LLaMA 长上下文扩展 | RoPE 的改良版 |

### 3.5 Infra 视角

- RoPE 的实现看似简单，但**复数乘法效率很关键**：现代 kernel 会把它转成 sin/cos 表 + 元素乘法。
- 扩展上下文（如 8K→128K）需要 RoPE 插值（Linear Scaling、NTK-aware、YaRN 等）。
- 多模态模型中，文本和图像往往用**不同的位置编码**（比如图像用 2D RoPE）。

---

## 4. Self-Attention 机制

### 4.1 为什么需要 Attention

在 Attention 出现之前，主流序列模型是 RNN/LSTM。它们有一个致命问题：

> **信息必须一步步传过去，距离一长就衰减。**

Attention 的核心思想：**任意两个 token 之间可以直接相连，距离不再是障碍。**

### 4.2 Q、K、V 是什么

Self-Attention 中，每个 token 会生成三个向量：

- **Query (Q)**：我想找什么样的信息（"我在问什么"）
- **Key (K)**：我能提供什么信息（"我的标签"）
- **Value (V)**：我具体的内容（"我携带的实质信息"）

类比：你在图书馆找书。
- 你手里有个**问题**（Q）
- 每本书背面有**索引标签**（K）
- 你根据 Q 和 K 的相似度决定借哪几本**书的内容**（V）

### 4.3 Self-Attention 的完整计算过程

**第 1 步：生成 Q、K、V（每个 token 都有）**

```
Q = X · W_Q    [B, S, D] × [D, D]  →  [B, S, D]
K = X · W_K
V = X · W_V
```

> 💡 **备注**：
> - **X 是什么**：Self-Attention 的输入——对于**第一层**，X = token embedding + 位置编码（原始输入）；对于**后续每一层**，X = 上一层 Transformer Block 的输出（已经过加工的"半成品"）。形状始终是 `[B, S, D]`，每位置一个 D 维向量。
> - **W_Q 等的作用**：W_Q、W_K、W_V 是三个"翻译官"，分别把 X 翻译成 Q（"我的问题是什么"）、K（"我能匹配什么"）、V（"我的具体内容是什么"）。**不用它们会怎样**？Q=K=X，每个 token 的"问题"和"标签"完全一样，Attention 就退化成"自相似度计算"，什么关系都学不到。**W_Q 的目的 = 三件事**：①让 Q、K、V 各说各的"话"（角色分工）；②让模型有可学习的旋钮（训练时自动调整"如何提问"）；③每个 head 还有自己的 W_Q，能问不同的问题（语法 / 指代 / 搭配）。

**第 2 步：拆成多头**

**为什么需要拆**：
经过第 1 步得到的 Q、K、V，每个都是 `[B, S, D]` 的 D 维向量——但这个 D 维向量是个"大杂烩"，里面混着语法、指代、搭配、距离等多种关系，挤在一个空间里施展不开，于是需要解决"一个空间装不下多种关系"的问题，做法是把 D 维拆成 `num_heads` 个小向量，让每个 head 各管一种关系。

**怎么拆**：

```
把 D 维拆成 num_heads 个 head_dim：
Q → [B, num_heads, S, head_dim]
K → [B, num_heads, S, head_dim]
V → [B, num_heads, S, head_dim]
```

代入 LLaMA-7B 数字（D=4096, num_heads=32, head_dim=128）：

```
Q: [1, 2048, 4096]  →  [1, 32, 2048, 128]
K: [1, 2048, 4096]  →  [1, 32, 2048, 128]
V: [1, 2048, 4096]  →  [1, 32, 2048, 128]
```

其中维度关系：`D = num_heads × head_dim = 32 × 128 = 4096` ✓

**拆完之后做什么**：
接下来 32 个 head 会**并行地、独立地**算 Attention——每个 head 拿着自己的 128 维子空间，去专注处理"我这种关系"（语法/指代/搭配等）。

**最后怎么合回去**：
第 7 步会把 32 个 head 的输出拼回 `[B, S, D]`，恢复原状。

**结果**：经过"切—并行算—拼"的来回，多种关系在各自的子空间里被独立处理，最后汇总成"多角度理解"的完整表示。

**第 3 步：算注意力分数**

**为什么需要算分数**：
经过前两步，Q、K、V 都准备好了，每个 token 都有自己的 Q（"我的问题是什么"）和 K（"我能匹配什么"）。但 Q 和 K 之间还没有"匹配度"——我们需要算出"每个 token 应该把注意力多分给哪些 token、少分给哪些 token"（比如对"银行"这个 token，"北京"应该比"小明"得到更多关注，因为"北京的银行"是有效搭配），于是需要解决"Q 和 K 没对上"的问题，做法是用点积把 Q 和 K 配对打分。

**怎么做**：
用 `Q · K^T`（点积）算出 Q 和 K 的"匹配度"：

```text
scores = Q · K^T / sqrt(head_dim)
形状:   [B, num_heads, S, S]
```

具体来说：`scores[i, j]` 表示"第 i 个 token 对第 j 个 token 的关注度"。

**为什么除以 sqrt(head_dim)**：
点积的数值范围会随 head_dim 增大而变大，会导致后续 softmax 进入"饱和区"（梯度极小、训不动）。除以 sqrt(head_dim) 是为了让分数保持合理范围，方便训练。

**结果**：
得到 scores `[B, num_heads, S, S]` 的注意力分数矩阵——形状是 **S × S 的方阵**。

> 注意：`S × S` 这个注意力矩阵是 Attention 显存爆炸的根源，也是 FlashAttention、PagedAttention 优化的重点对象。

**第 4 步：Mask（Decoder 必须做）**

**为什么需要 Mask**：
LLM 是 Decoder-only 的自回归模型——生成第 i 个 token 时**不能偷看第 i+1、i+2...位置**（否则就是"作弊"），但第 3 步算出的分数让每个 token 都能看到全部位置，于是需要解决"不能看未来"的问题，做法是用一个"下三角矩阵"把未来位置屏蔽掉。

**怎么做 Mask（causal mask）**：

```
   t1  t2  t3  t4
t1  ✓   ✗   ✗   ✗    ← t1 只能看自己
t2  ✓   ✓   ✗   ✗    ← t2 只能看 t1 + 自己
t3  ✓   ✓   ✓   ✗    ← t3 能看 t1、t2 + 自己
t4  ✓   ✓   ✓   ✓    ← t4 能看全部
```

整体形状 `[S, S]`，**下三角有效**（含对角线），上三角全部屏蔽。

**具体怎么实现**：
把 mask 矩阵中"✗"位置填成 `-∞`（负无穷大），加到 scores 上：

```
scores_masked = scores + mask
   ✓ 位置：mask = 0    （保留原分数）
   ✗ 位置：mask = -∞   （彻底屏蔽）
```

下一步做 softmax 时，`e^(-∞) = 0`，这些位置的关注权重自动变成 0。

**结果**：经过 Mask，每个 token 只能关注自己和之前的位置，符合自回归生成的物理意义。这种下三角 mask 称为 **causal mask（因果掩码）**，是 LLaMA、GPT、Qwen 等 Decoder-only 模型的标配。

**第 5 步：Softmax**

**为什么需要 Softmax**：
第 3 步算出的 scores 是任意范围的实数（0.5、3.2、-1.8 ...），第 4 步加了 mask 后未来位置是 -∞。这些"原始分数"还不能直接当"权重"用——它们不能解释为"我关注谁多少"，于是需要解决"分数没法直接当权重"的问题，做法是用 softmax 把分数转成"加权和为 1"的概率分布。

**怎么做**：
softmax 的本质就两步：

```
weights = softmax(scores + mask)
   = exp(scores + mask) / sum(exp(scores + mask))
                ↑                  ↑
              变正数              归一化
```

1. **变正数**：对每个分数做 `e^x`，所有数都变成正数（`e^x > 0`）
2. **归一化**：除以总和，让所有权重加在一起 = 1

**对 mask 位置的特殊处理**：
第 4 步把未来位置填成了 `-∞`：

- `e^(-∞) = 0` → 未来位置的权重自动变成 0（彻底不关注）
- `e^(正常分数) > 0` → 当前及之前位置的权重正常计算

**结果**：得到一个"概率分布"形式的关注权重 `[B, num_heads, S, S]`——每个 token 对所有位置（除未来）的关注度**加权和恰好为 1**，可以直接交给第 6 步做加权求和。

**第 6 步：加权求和**

**为什么需要这一步**：
第 5 步得到了 `weights`（关注权重，加权和 = 1），但 weights 只是"关注度比例"——它告诉我们"每个 token 应该关注其他 token 多少"，但没告诉我们"该关注的内容是什么"。内容在 V 里（第 1 步生成），于是需要把 weights 和 V 结合，做法是用 weights 对 V 做加权求和——关注度高的位置拿走更多 V，关注度低的位置少拿。

**怎么做**：
矩阵乘法：

```
output = weights · V
```

具体到每个 token：

```
output[i] = Σ weights[i, j] × V[j]    （j 遍历所有位置）
           ↑                ↑
        关注度比例        具体内容
```

含义：**关注度作为"取走比例"**，从 V[j] 里按比例取走内容。

**结果**：得到每个 token 的 output `[B, num_heads, S, head_dim]`——每个 token 的 output = 它"应该关注的内容"的加权汇总。其中 mask 位置的 weights = 0 → 完全不取这些位置的内容。形状保持，等第 7 步合并多头。

**第 7 步：合并多头**

**为什么需要合并**：
第 6 步得到 output，形状是 `[B, num_heads, S, head_dim]`——32 个 head 各自产出了一个 128 维的"小结论"，但 Block 期待的是 `[B, S, D]` 的统一表示（要和入口形状一致才能继续往下传）。于是需要解决"分散的小结论没法直接交给下一层"的问题，做法是把它们拼回原形状。

**怎么做**：
两步：reshape + 线性映射

```
[B, num_heads, S, head_dim]  →  reshape  →  [B, S, D]  →  W_O  →  [B, S, D]
       （拼接）              （形状恢复）   （融合信息）
```

**结果**：最终输出 `[B, S, D]`，形状恢复成 Block 入口的样子，Self-Attention 完成一次完整闭环。每个 token 的 D 维向量 = 32 个 head 的"小结论"汇总融合后的结果。

### 4.4 完整公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\right) V
$$

> **为什么除以 $\sqrt{d_k}$？**
>
> 如果不除，当 `d_k` 大时，点积的方差会变大，softmax 会进入梯度极小的饱和区。除以 $\sqrt{d_k}$ 是为了让点积方差维持在 1 左右。

### 4.5 Multi-Head 的意义

如果只有 1 个 head，所有信息必须塞到 `head_dim` 维里。
多个 head 让模型在**不同子空间**并行学习不同的关系：

- head 1：可能学"语法主谓关系"
- head 2：可能学"指代消解"
- head 3：可能学"距离相近的相关性"
- ...

### 4.6 Infra 视角：Self-Attention 的瓶颈

| 维度 | 复杂度 | 瓶颈类型 |
|---|---|---|
| 计算 | $O(S^2 \cdot D)$ | 长序列下计算量爆炸 |
| 显存（激活） | $O(S^2)$（注意力矩阵） | 中等 |
| 显存（KV Cache） | $O(B \cdot S \cdot D)$ | 推理时关键瓶颈 |
| 访存 | 读写 K、V 多次 | **memory-bound** |

> **记忆点**：Attention 是 memory-bound（访存密集），FFN 是 compute-bound（计算密集）。这决定了 GPU kernel 优化策略完全不同。

---

## 5. 前馈网络（FFN）

**为什么需要 FFN**：
Self-Attention 让每个 token 看到了其他 token 的信息（"跨 token 通信"），但光"看"还不够——每个 token 还需要把看到的信息"消化吸收"，变成自己的更深层理解。Attention 解决"听别人说什么"，FFN 解决"自己消化理解"，于是需要一个对每个 token **独立加工**的环节，做法就是用一个两层 MLP 把每个 token 的向量"放大→思考→压回"。

**FFN 做了什么**：
简单来说就是一个两层全连接网络，带一个非线性激活：

```text
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

形状变化（以 LLaMA 为例，`hidden_dim=4096`）：

```
[B, S, 4096]
   │ W_1 (4096 → 11008)   ← 扩展约 2.7 倍
   ▼
[B, S, 11008]
   │ activation (SiLU / GELU / ReLU)
   ▼
[B, S, 11008]
   │ W_2 (11008 → 4096)   ← 压回原维度
   ▼
[B, S, 4096]
```

**结果**：每个 token 经过"扩—激活—压"的来回，输出一个升级版的 D 维向量——信息被加工得更丰富。大量研究表明，**FFN 里储存了模型的事实性知识**（"法国的首都是巴黎"这种"死知识"基本都在 FFN 里）。

### 5.1 升维-激活-降维 三步详解

**第 1 步：升维（用 W_1 把 D 维变高维）**

W_1 是 D × D_ff 的矩阵（LLaMA-7B: 4096 × 11008）：

```text
x [4096] · W_1 [4096 × 11008] → h [11008]
```

**为什么升维？** 低维空间能表达的"非线性变换"有限，升维给模型更多"思考空间"——相当于把一张小桌子换成大桌子，能摆下更多东西。

**第 2 步：激活（用非线性函数过滤）**

激活函数给升维后的向量做"非线性过滤"，决定哪些特征通过、哪些被抑制：

```text
activation: h [11008] → a [11008]
```

不同激活函数特性不同（详见 5.2 节），核心都是引入**非线性**——没有激活函数，再多层的 MLP 也只是线性变换，没意义。

**第 3 步：降维（用 W_2 把高维压回 D）**

W_2 是 D_ff × D 的矩阵（LLaMA-7B: 11008 × 4096）：

```text
a [11008] · W_2 [11008 × 4096] → out [4096]
```

把"加工后的精华"浓缩回 D 维，准备交给下一层或下游使用。

### 5.2 激活函数

| 模型 | 激活函数 | 特点 |
|---|---|---|
| 原始 Transformer | ReLU | 简单、稀疏 |
| GPT 系列 | GELU | 平滑、表现好 |
| LLaMA | **SwiGLU** | 三个权重矩阵，效果最好 |

**SwiGLU**（LLaMA 用的）：

```
FFN_SwiGLU(x) = (SiLU(W_1 · x) ⊙ W_3 · x) · W_2
```

注意它有 **3** 个权重矩阵（`W_1, W_2, W_3`），而不是 2 个。这也是为什么 LLaMA 的 FFN 参数计算要乘 3。

### 5.3 FFN 的参数占比

以 `D=4096, D_ff=11008` 为例，单层 FFN 参数：

```
W_1: 4096 × 11008 = 45M
W_2: 11008 × 4096 = 45M
W_3 (SwiGLU):    = 45M  (LLaMA)
─────────────────────────
总计：约 135M / 层
```

而 Self-Attention 单层参数（`W_Q, W_K, W_V, W_O`）：

```
4 × (4096 × 4096) = 67M
```

> **结论**：FFN 是参数的大头（约 2/3），也是 FLOPs 的大头。这就是为什么大模型量化、剪枝的主要战场都在 FFN 上。

### 5.4 Infra 视角

- **计算密集型**：FFN 是大矩阵乘法，完美匹配 GPU Tensor Core。
- **容易并行**：每个 token 的 FFN 独立计算，可以天然 batch。
- **量化友好**：权重分布相对集中，INT8/INT4 量化损失小。
- **MoE 的基础**：把 FFN 拆成多个专家，每次只激活部分（Mixtral、DeepSeek-V3）。

---

## 6. LayerNorm 与残差连接

**为什么需要这两个机制**：
深层网络训练有两个致命问题——一是**数值失控**：数据经过一层层变换后，数值要么爆炸要么消失（"滚雪球效应"），到几十层时已经完全没法训；二是**梯度消失**：反向传播时梯度需要一层层连乘回去，越乘越小，到浅层时几乎为 0，权重无法更新。于是需要同时解决"数值跑偏"和"梯度消失"两个问题，做法是用两个机制分工合作：**LayerNorm 解决前者**（数值校准），**残差连接解决后者**（梯度短路）。

**它们各自做了什么**：

| 机制 | 公式 | 解决什么问题 | 一句话类比 |
|---|---|---|---|
| **残差连接** | `y = x + Sublayer(x)` | 梯度消失 | 给梯度留一条"高速公路" |
| **LayerNorm** | 把向量归一化到合理范围 | 数值失控 | 进加工车间前先"校准仪表盘" |

两者结合的 Pre-Norm 写法：`y = x + Sublayer(LayerNorm(x))`，是 LLaMA、GPT、Qwen 等现代 LLM 的标配。

**结果**：有了这两个机制，深层网络（32 层、96 层、甚至上千层）才能稳定训练——这是大模型能变"深"的根本前提。

### 6.1 残差连接（Residual Connection）

公式：

```
y = x + Sublayer(x)
```

其中 `Sublayer` 是 Attention 或 FFN。

**为什么有效？**

1. **梯度直通**：反传时，梯度可以走"短路"+ 输入，避免连乘消失。
2. **易于优化**：网络至少可以学到"什么都不做"（把 Sublayer 输出学成 0）。
3. **支持极深网络**：没有残差，100+ 层根本训不出来。

### 6.2 LayerNorm

公式（对一个 token 的 D 维向量做归一化）：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu, \sigma^2$ 是这个 token 在 D 维上的均值和方差，$\gamma, \beta$ 是可学习参数。

**和 BatchNorm 的区别**：

| | BatchNorm | LayerNorm |
|---|---|---|
| 归一化维度 | 跨 batch 同一特征 | 单个 token 跨所有特征 |
| 和 batch_size 关系 | 强相关（batch 太小会出问题） | 无关 |
| 推理时 | 需要维护 running mean/var | 直接算 |
| 序列模型适用性 | ❌ 差 | ✅ 好 |

> **为什么 LLM 都用 LayerNorm？**
>
> 因为 LLM 是逐 token 生成的，batch=1 也要能跑。BatchNorm 在这种场景会失效。

### 6.3 Pre-Norm vs Post-Norm

这是 Transformer Block 的两种写法：

**Post-Norm**（原始 Transformer）：
```
y = LayerNorm(x + Sublayer(x))
```

**Pre-Norm**（现代 LLM）：
```
y = x + Sublayer(LayerNorm(x))
```

**为什么现代 LLM 几乎全部用 Pre-Norm？**

| | Post-Norm | Pre-Norm |
|---|---|---|
| 训练稳定性 | 差，容易发散 | 好 |
| 深层训练 | 难（100+ 层很吃力） | 易（千层也没问题） |
| 最终性能 | 略好（论文说法） | 实际更好 |
| 工程友好 | 差 | 好 |

> **记忆点**：所有现代 LLM（LLaMA、GPT、Qwen、DeepSeek）都用 Pre-Norm + RMSNorm（LayerNorm 的简化版，只用 $\gamma$ 不算均值）。

### 6.4 RMSNorm

LLaMA 用的归一化：

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

去掉了均值中心化，计算更快，效果几乎一样。

### 6.5 Infra 视角

- Pre-Norm 让激活值（activation）在各层尺度稳定，**对量化很友好**。
- 残差连接产生的中间结果（`x + Sublayer(x)`）会**占用大量显存**，是 activation checkpointing 的主要优化对象。
- RMSNorm 比 LayerNorm 少一次均值计算，**减少约 10% 的开销**。

---

## 7. 完整的 Transformer Decoder Block

### 7.1 伪代码实现

这是 LLaMA 风格的 Decoder Block（伪代码）：

```python
def decoder_block(x):
    # Self-Attention 子层（Pre-Norm）
    h = x + self_attention(rms_norm(x))

    # FFN 子层（Pre-Norm）
    h = h + ffn(rms_norm(h))

    return h
```

更详细的代码：

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn   = FeedForward(d_model, d_ff)        # SwiGLU
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, mask, kv_cache=None):
        # x: [B, S, D]
        # mask: causal mask [S, S]

        # Self-Attention with residual
        attn_out = self.attn(self.norm1(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_out

        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        return x
```

### 7.2 完整 LLM 推理流程

```python
class LLM:
    def __init__(self):
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(...) for _ in range(num_layers)]
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # 1. Embedding
        x = self.embed(input_ids)   # [B, S, D]

        # 2. N 层 Decoder Block
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # 3. 最终归一化
        x = self.norm(x)

        # 4. 映射到词表
        logits = self.lm_head(x)    # [B, S, vocab_size]

        return logits
```

### 7.3 形状流转一览（推理时）

假设 `B=1, S=2048, D=4096, num_layers=32, vocab_size=128000`：

| 步骤 | 张量 | 形状 | 显存 |
|---|---|---|---|
| Input IDs | input_ids | [1, 2048] | ~16KB |
| Embedding | x | [1, 2048, 4096] | 32MB |
| Block 0 输入 | x | [1, 2048, 4096] | 32MB |
| Block 0 输出 | x | [1, 2048, 4096] | 32MB |
| ... | ... | ... | ... |
| Block 31 输出 | x | [1, 2048, 4096] | 32MB |
| Logits | logits | [1, 2048, 128000] | 1GB |

> **注意**：最后 logits 的 vocab 维度很大，推理时通常只算最后一个 token 的 logits，避免全量计算。

### 7.4 KV Cache：推理时为什么需要它

**没有 KV Cache 时**：

生成 1000 个 token 需要 1000 次 forward，每次重算所有历史 token 的 K、V。

**有 KV Cache 时**：

每次只算新 token 的 Q、K、V，历史 K、V 缓存在显存里直接用。

KV Cache 的形状：

```
K: [num_layers, B, num_heads, S_cached, head_dim]
V: [num_layers, B, num_heads, S_cached, head_dim]
```

以 LLaMA-7B、`S=4096` 为例：

```
32 层 × 2 (K和V) × 1 batch × 32 heads × 4096 seq × 128 head_dim × 2 bytes (FP16)
= 32 × 2 × 1 × 32 × 4096 × 128 × 2
= 2 GB
```

> **记忆点**：KV Cache 是 LLM 推理的最大显存消耗，比模型权重本身还大。这就是 vLLM、PagedAttention 出现的根本原因。

---

## 8. 从 Transformer 到 LLM：自回归生成

### 8.1 什么是自回归生成

LLM 的生成方式是**自回归（autoregressive）**：

```
输入: "今天天气"
↓
模型 → "很"
↓
"今天天气很" → "好"
↓
"今天天气很好" → "，"
↓
"今天天气很好，" → "适合"
↓
...（一直生成，直到结束符或达到最大长度）
```

每一步：
1. 把当前序列喂进模型
2. 取最后一个 token 的 logits
3. argmax（或采样）得到下一个 token
4. 把新 token 拼回序列
5. 重复

### 8.2 推理的两种模式

#### Prefill 阶段

- 把整个 prompt 一次性喂给模型
- 计算所有 token 的 K、V，缓存到 KV Cache
- 输出第一个 token

特点：**compute-bound**，可以充分利用 GPU 并行。

#### Decode 阶段

- 每次只输入一个新 token
- 复用历史的 KV Cache
- 输出下一个 token

特点：**memory-bound**，每生成一个 token 都要读完整 KV Cache。

> **这就是为什么推理服务要做 "continuous batching"、"chunked prefill"**：把 prefill 和 decode 混合调度，提升 GPU 利用率。

### 8.3 采样策略

模型输出的 logits 是 vocab_size 维的向量，要变成下一个 token 的选择：

| 策略 | 公式 | 效果 |
|---|---|---|
| **Greedy** | `argmax(logits)` | 确定，但容易循环 |
| **Temperature** | `logits / T`，再 softmax | T 大 → 更随机；T 小 → 更确定 |
| **Top-K** | 只保留概率最高的 K 个 | 过滤掉低概率噪声 |
| **Top-P** | 保留累积概率达到 P 的最小集合 | 自适应截断 |
| **Beam Search** | 维护 K 条最优路径 | 适合翻译等确定任务 |

### 8.4 停止条件

生成什么时候停止？

1. 生成 **EOS token**（end-of-sequence）
2. 达到 **max_new_tokens** 上限
3. 触发停止字符串（stop strings）
4. 用户中断

### 8.5 推理性能指标

AI Infra 工程师最关心的几个指标：

| 指标 | 含义 | 影响因素 |
|---|---|---|
| **TTFT** (Time To First Token) | 第一个 token 的延迟 | Prefill 时间，受 prompt 长度影响 |
| **TPOT** (Time Per Output Token) | 每个输出 token 的延迟 | Decode 时间，受 KV Cache 影响 |
| **Throughput** | 每秒生成的 token 总数 | 受 batch size、并行度影响 |
| **Latency** | 端到端总延迟 | TTFT + TPOT × 输出长度 |

### 8.6 优化全景

下图展示了 Transformer 推理中各部分的优化空间：

```
            ┌──────────────────────────────────────┐
            │   Transformer 推理优化全景            │
            └──────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼─────┐       ┌────▼─────┐       ┌────▼─────┐
   │ Attention│       │   FFN    │       │ 整体调度  │
   └────┬─────┘       └────┬─────┘       └────┬─────┘
        │                  │                   │
   ┌────┴────────┐    ┌────┴────────┐     ┌────┴────────┐
FlashAttention  │    │ 量化 (INT8/ │     │ Continuous  │
PagedAttention  │    │  INT4/FP8)  │     │ Batching    │
Multi-Query    │    │ 投机解码     │     │ Chunked     │
Grouped-Query  │    │ (Speculative│     │ Prefill     │
KV Cache 量化  │    │ Decoding)   │     │ Prefix Cache│
KV Cache 压缩  │    │ MoE         │     │ 动态批处理  │
└───────────────┘    └─────────────┘     └─────────────┘
```

每一项优化，都对应 Transformer 某个具体的内部组件。

---

## 9. 本章小结

### 9.1 关键概念速查表

| 概念 | 一句话总结 |
|---|---|
| Self-Attention | 让每个 token 关注其他 token，复杂度 $O(S^2 \cdot D)$ |
| Multi-Head | 把注意力拆成多个子空间，并行学不同关系 |
| Causal Mask | Decoder 必须有，遮住未来位置 |
| FFN | 两层 MLP，参数和计算的大头，存知识 |
| RoPE | 旋转位置编码，当前 LLM 标配 |
| Pre-Norm + RMSNorm | 现代 LLM 的标配归一化方式 |
| 残差连接 | 让深层网络能训练 |
| KV Cache | 推理时缓存历史 K、V，自回归必需 |
| Prefill vs Decode | 推理的两个阶段，前者计算密集，后者访存密集 |
| 自回归生成 | 一次生成一个 token，直到停止 |

### 9.2 给后续章节的"钩子"

本章只是打地基，后续 AI Infra 课程会深入：

- **KV Cache 优化**：PagedAttention、KV Cache 压缩、量化
- **Attention 优化**：FlashAttention、FlashDecoding、Multi-Query Attention
- **并行策略**：Tensor Parallel、Pipeline Parallel、Sequence Parallel
- **推理引擎**：vLLM、TGI、SGLang、TensorRT-LLM 的内部架构
- **量化**：INT8、INT4、FP8、AWQ、GPTQ
- **投机解码**：小模型辅助大模型加速

每一个主题，都会回到本章的某个具体组件。所以请务必把基础打牢。

---

## 10. 思考题

1. 为什么 Decoder 必须用 Causal Mask，而 Encoder 不用？
2. Pre-Norm 和 Post-Norm 哪个更适合超深层网络？为什么？
3. LLaMA-7B 有 32 层，`d_model=4096`，`num_heads=32`。请估算一次 forward（`seq_len=2048`）的 KV Cache 显存占用（FP16）。
4. 推理时 Decode 阶段是 memory-bound，这意味着什么优化思路？
5. 如果你设计一个超长上下文（1M token）模型，Transformer 的哪些部分会最先成为瓶颈？

---

> **下一章预告**：[第二章 Attention 进阶：从 MHA 到 FlashAttention](../02-Attention进阶/02-第二章-Attention进阶.md)
