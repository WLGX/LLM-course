# L1 思考题三组：平滑与 ∞-gram（P122）、PyTorch 实现 NLL（P109）、LAS 三问（P295）

> 作业依据：L1 第 122 张（KN 为什么看续接多样性、JM/Katz/KN 谁更工程友好、∞-gram 能否解决长依赖）、第 109 张（实际训练中如何用 PyTorch 实现 NLL，并审查幻灯片上 Qwen 示例代码是否正确）、第 295 张（LAS 换子词、流式解码、Transformer 化的三个问题）。
> 所有结论均来自真实运行（P109 的验证脚本与输出见 GitHub 仓库 [hw1/nll_check.py](https://github.com/WLGX/LLM-course/tree/main/hw1) 与本帖附录）；P122 的分析与我自己 n-gram 作业的实测相互印证。

## 一、P122：平滑与长依赖的三个问题

### 1.1 为什么 KN 要看"续接多样性"而不是词频？

**关键在于"回退发生时，低阶分布被问到的是什么问题"。** 回退发生的场合是：当前上下文的某个高阶 n-gram 在训练语料里没被观察到。此时低阶分布的任务不是回答"哪个词在语料里最常见"，而是回答"**哪个词适合出现在任意一个新上下文之后**"——这两个问题的答案可以完全不同。

- 词频回答不好第二个问题：高频词往往是"绑定词"。经典例子（讲义同款）：`Francisco` 出现次数不少，但几乎只被 `San` 接续；`Kong` 几乎只跟 `Hong`。若回退到词频，`Francisco` 会霸占概率，把大量"真正通用"的词挤掉。
- 续接多样性 P_cont(w) = **w 被多少种不同的左邻上下文接续过**：`glasses` 可以被 `wear`、`broken`、`sung`、`reading`……接，多样性高；即使它全局频次不算顶尖，作为"通用续接者"也值得信任。

一句话总结：**高阶分布回答"这个特定上下文后接什么"，低阶分布的任务是"谁适合作为新续接"——两个问题需要两个统计量**。这正是 KN 相对 JM/Katz 的"致命缺陷修复"：JM 把"词本身常见"和"词容易接在任意词后"混在一起；Katz 回退后的低阶分布仍是 MLE 词频，同样没建模续接倾向。

我的 hw1 实跑提供了一组旁证：自写 Stupid Backoff（回退到词频 unigram）与 KenLM 默认的 KN 在受限候选集排序上几乎重合（Top-1 34–35%），因为候选集小、高阶命中率高、回退走得浅；一旦回退成为主要路径，词频兜底与续接兜底的差距就会显现（Goodman 2001 的 PPL 对比正是这个结论）。

### 1.2 JM、Katz、KN 谁更"工程友好"？

从**实现成本、参数估计、效果**三个维度比较：

| 方法 | 原理（讲义口径） | 参数估计成本 | 实现复杂度 | 效果 |
|---|---|---|---|---|
| JM（Jelinek–Mercer 插值） | "谁都说点"：各阶线性混合 | λ 权重要靠 EM 迭代或网格搜索，阶数×组合爆炸 | 低，但调参贵 | 基准水平；λ 敏感 |
| Katz（回退+折扣） | "见过的先说，没见过再回退" | 每阶要解全局归一化方程，算出折扣系数 β 与回退权重 α(w) | **高**，数值细节多、易错 | 好，但不如 KN |
| KN / IKN | "回退也要聪明回退"：低阶看续接多样性 | 只需一个折扣参数（IKN 用分段折扣，仍是一次性统计），**无需迭代** | 低 | 大样本比较中最稳健（IKN 尤佳） |

结论：**KN 最工程友好**——单参数、可离线一次算完、效果还最好，工具链也最成熟（KenLM 的 `lmplz` 默认即出 KN 模型）。我的实测：`lmplz` 一次运行 427.9s 同时产出 1–10 阶全部 KN 模型，全程没有任何需要手调的平滑参数；换成自写 Katz 的话，光每阶的折扣系数归一化就够写一天。JM 的唯一亮点是**跨域混合直观**（线性插值权重可以按域手工调整，IR 领域有系统比较）；Katz 是三者中实现最繁琐的，学术工具包（SRILM）里常见但"工程友好"最差。

两个提醒：① 讲义区分 KN 与 IKN（插值版 Modified KN），插值版通常更好；② 工程友好≠永远选 KN——超大数据下 Stupid Backoff 用常数回退换取 MapReduce 可并行，是另一种"工程友好"。

### 1.3 ∞-gram 能解决长依赖问题吗？

先回答"依赖问题的根本原因和目的"：语言中的约束不是发生在相邻词层面，而是结构层面——代词指代、主谓一致、跨句主题连贯——本质是**符号间按意义关系的组合**，n-gram 只能捕捉窗口内表面词串的共现。建模长依赖的目的，是让"远处的信息"约束当前的预测。

**∞-gram 不能解决，有三个结构性理由**：

1. **稀疏性（最致命）**：历史越长，完全相同的上下文越不可能重现。n→∞ 时，几乎每个"无限历史"都是训练中从未逐字出现过的组合，高阶计数全部失效、只能一路回退——∞-gram 的实际行为退化成低阶模型。而长依赖恰恰发生在"没见过的组合"上，计数表对此**原则上无能为力**。
2. **指数爆炸**：参数与存储随 |V|^n 增长。我 hw1 的实测：5 阶 distinct n-gram 见顶 4,895,726 后回落、10 阶二值化模型已达 832 MiB——n 每加一阶，成本翻数倍，"∞"在工程上无意义。
3. **没有语义泛化**：计数表不存在"相似"概念。"胜利召开了"与"成功举办了"字面完全不同，n-gram 无法共享它们后面的分布；而真正约束续写的，是"会议报道"这个抽象语域。**窗口加长 ≠ 理解加深。**

诚实补充：讲义 NOTES 链接的 infinigram（Liu et al. 2024）把 n 放大到万亿 token 级的检索式模型，PPL 确实显著下降——但它的收益来自"**检索到更长的字面匹配**"，且是"检索+回退"架构；稀疏性与字面匹配问题依然存在。它证明的是"更长的匹配有用"，不是"n-gram 结构能建模长依赖"。这也正是自注意力占优的核心动因（内容寻址、O(1) 路径、表示共享），与 P123"结构性局限无法解决"一致。我 hw1 的续写实验是同一结论的另一面：5 阶模型输出"局部通顺、全局空洞"。

## 二、P109：实际训练中如何用 PyTorch 实现 NLL？

幻灯片给了 Qwen 生成的示例代码并问"这个代码正确吗？"。我的结论：**能跑，但有三处真坑，且"nll 等价于 cross_entropy"这个论断按原代码不成立**。以下每条都有真实运行输出（torch 2.9.0+cpu，完整脚本见仓库 `hw1/nll_check.py`）。

### 2.1 逐条审查幻灯片代码

```python
probs = F.softmax(logits, dim=-1)                    # 坑 1：算了但从未使用
log_probs = F.log_softmax(logits, dim=-1)            # 实际用的是这个
predicted_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
nll = -predicted_log_probs.sum()                     # 坑 2：sum
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))   # 默认 mean
```

1. **死代码与注释脱节**：注释说"Step 1 计算 softmax 概率、Step 2 提取概率"，实际 Step 2 用的是 `log_softmax`——`probs` 从未被使用。这正是"看起来对、实现上是坑"的典型：读代码的人会被注释带偏。
2. **"等价"论断不成立（sum vs mean）**：`nll` 用的是 `.sum()`（全序列总量），而 `F.cross_entropy` 默认 `reduction='mean'`（每位置平均）。实测：`nll = 94.1239`，`ce(mean) = 15.6873`，比值恰好是 `seq_len = 6`。只有 `reduction='sum'` 才与 nll 逐位相等（`allclose: True`）。训练时用 sum 还是 mean 直接改变梯度尺度，学习率要跟着调。
3. **没有 padding 掩码**：真实训练中 batch 内句子长度不同，pad 位置会被当成真实标签计入 loss。实测：同一批数据不掩码 loss = 15.4133、`ignore_index=0` 掩码后 = 15.8621——两个值明显不同，**pad 位置的预测（本不该计入）参与了 loss 计算**，方向随随机初始化可高可低，这正是"污染"的表现。
4. **语言模型特有的对齐**：自回归 LM 是位置 t 的 logits 预测位置 t+1 的词（teacher forcing），targets 是 input 右移一位，示例随意编 targets 没体现这一点。
5. **数值细节**：极端 logits 下 `log(softmax(x))` 把 -1000 和 -2000 都压成 `-inf`（信息丢失），`log_softmax` 正确给出 -1000 / -2000。另一个经典坑是**把 probs 传进 `F.cross_entropy`**——它内部自带 softmax，等于做了两次 softmax。

### 2.2 正确写法与等价性验证（实测输出）

```python
# 语言建模：masked mean NLL，等价于 cross_entropy(ignore_index)
logits_lm = model(input_ids[:, :-1])            # [B, T-1, V]
lg = F.log_softmax(logits_lm, dim=-1)
g = lg.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)   # 右移对齐
mask = ~(input_ids[:, 1:] == pad_id)
nll = -(g * mask).sum() / mask.sum()            # 每 token 平均，pad 不参与

ce = F.cross_entropy(logits_lm.reshape(-1, V),
                     input_ids[:, 1:].reshape(-1), ignore_index=pad_id)
# 实测：manual = 15.594362, ce = 15.594361, allclose: True
```

工程建议：直接用 `F.cross_entropy(..., ignore_index=pad_id)`（等价、简洁、内部就是 log_softmax 路径）；需要逐 token 概率（如 RL 里做 PPO 的 log_prob）时才手工 `log_softmax + gather`。label_smoothing 等需求交给 `cross_entropy` 的参数，不要自己造轮子。

## 三、P295：LAS 的三个变体问题

### 3.1 字符改成子词（wordpiece/subword）会怎样？

**优点**：① 序列长度缩短约 3–4 倍——解码步数少、速度快、算力省；② attention 对齐更容易（字符级要几百步，注意力容易漂移），每步的单元自带更多上下文；③ 输出单元本身是词素，可直接接词典/语言模型；④ 缓解字符模型的乱拼串问题（不会拼出词表外的"新词"）。

**缺点**：① OOV 回来了——词表外词需要 `<unk>` 或回退到字符拼写；② 词表成为超参（BPE 训练语料、词表大小），跨域切分不一致；③ 丢失字符级规律（数字、专名、拼写、口音变体）；④ 错误粒度变大（一个子词错=多个字符错）；⑤ 每个单元的训练样本变少。

诚实的原文对照：LAS 原文（Chan et al. 2015）在 Google Voice Search 上**字符模型与词模型效果相当**，字符还免去了词表与 OOV 的烦恼——所以答案是"子词是词与字符的折中"，在后续实践（ESPnet、Whisper 等）中 BPE 子词成为主流，但代价是词表设计与 OOV 处理这两个字符模型没有的问题。

### 3.2 如何在 LAS 中改进 online/streaming 解码？

LAS 的结构性问题是：双向 LSTM encoder + 全序列 attention **必须等整句话说完才能开始解码**。流式化要动两层：

- **结构层**：encoder 换单向或分块（chunked）RNN；attention 限制在局部窗口。
- **注意力变体（核心）**——让输入输出位置**单调对齐**，才能"边听边出"：
  - **Monotonic attention**（Raffel et al. 2017）：每步用一个伯努利决策（对能量的 sigmoid）选择"注意力位置前进一步"或"等待（不输出）"，保证位置只进不退，天然流式。
  - **MoChA**（Monotonic Chunkwise Attention，Chiu & Raffel 2018）：修复 monotonic attention 只聚焦单帧、质量略降的问题——在单调边界处一次 attend 一个固定大小 chunk，chunk 内做 softmax，质量接近全序列 attention。
  - 其他路线：hard monotonic attention（强制每个输出至少对应一个输入，可提前知道何时终止）、triggered attention、固定窗口 attention。
  - **训练技巧**：这些硬决策不可微，常用 EMMA（期望单调注意力近似）、Gumbel-softmax 松弛或 REINFORCE 梯度；工程上也可以先训全 attention 模型再微调/蒸馏到流式变体。

### 3.3 LAS 换成 Transformer encoder + decoder（非 RNN）会带来什么变化？

这其实就是 vanilla Transformer（Vaswani et al. 2017）对标 RNN seq2seq 的问题，把 LAS 的 Listener 换成 self-attention encoder、Speller 换成自回归 decoder：

**优点**：① encoder 并行处理全序列，训练大幅加速；② 长依赖建模更好（任意位置 O(1) 路径，无 RNN 的梯度消失/遗忘）；③ 大语料下质量全面占优（Whisper 等语音大模型全部采用）。

**缺点**：① 全序列 attention 依然**非流式**——流式要 mask/分块，反而比 RNN 更难做（3.2 的单调注意力变体正是为 Transformer 补这块）；② attention 复杂度 O(T²)，长语音要分块/稀疏注意力；③ 顺序信息靠 positional encoding，而 RNN 天然有序；④ 更吃数据与算力，小语料上 RNN 未必差；⑤ 解码仍自回归逐 token（可用 non-autoregressive 加速，但那是另一笔账）。

一句话：**训练与质量的账全在 Transformer 这边，流式是它唯一的"还债"点**——这也解释了为什么后来的流式语音识别研究（RNN-T、单调注意力、分块 Transformer）几乎全在围绕"给 Transformer 补上单调性"展开。

## 附录：P109 验证运行输出（节选）

```
torch 2.9.0+cpu
1) 复现幻灯片代码
   nll (sum)            = 94.1239
   cross_entropy 默认mean = 15.6873     <- 差 seq_len=6 倍，'等价'不成立
   nll == ce(sum): True
2) padding 掩码
   不掩码 loss = 15.4133   ignore_index=0 loss = 15.8621
4) 数值稳定性
   log(softmax)  = [0.0, -inf, -inf]    <- 信息丢失
   log_softmax   = [0.0, -1000.0, -2000.0]
5) 等价性验证
   manual masked mean NLL = 15.594362
   F.cross_entropy(ignore_index=0) = 15.594361
   allclose: True
```

## 参考资料

- L1 讲义第 109、116–124、295 张；Goodman 2001（A Bit of Progress in Language Modeling）; Chen & Goodman 1999。
- Chan et al. 2015, Listen, Attend and Spell; Raffel et al. 2017, Online and Linear-Time Attention by Enforcing Monotonic Alignments; Chiu & Raffel 2018, Monotonic Chunkwise Attention; Vaswani et al. 2017; Liu et al. 2024, Infini-gram。
- 运行材料：仓库 [hw1/nll_check.py](https://github.com/WLGX/LLM-course/tree/main/hw1) 与 `hw1/results/nll_check_output.txt`。
