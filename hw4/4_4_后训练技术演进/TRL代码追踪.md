# GRPO 训练管线代码追踪（TRL v1.9.2）

> P226 大作业"算法与代码对应"部分。追踪对象：HuggingFace **TRL v1.9.2**（commit `33f9e46`，2026-07-28），
> 本地克隆 `F:/LLM-course/refs/trl/`（未入库）。对照论文：GRPO（arXiv:2402.03300）、
> DeepSeek-R1（arXiv:2501.12948）、DAPO（arXiv:2503.14476）、Dr. GRPO（arXiv:2503.20783）、OPSM（arXiv:2504.23336）。

## 0. 为什么选 TRL

三选一（TRL / verl / OpenRLHF）中，TRL 代码组织最紧凑、与 HuggingFace Trainer 生态兼容，且 v1.9.2 的 GRPO 实现已经吸收了 2025 年一整轮论文演进（默认损失即 DAPO），最适合演示"论文方法 → 训练程序"的映射。verl 更适合大规模多机 RL（Ray 异步架构），OpenRLHF 是 ray+HF 的轻量替代，本报告不展开。

## 1. 完整管线：数据 → 参数更新的六步

TRL GRPO 的管线以 `GRPOTrainer.training_step`（`grpo_trainer.py:1573`，每步计时）为轴，六个环节及关键函数如下：

```
① 数据准备   GRPODataset + repeat_generator（HF dataloader 侧，prompt 复制 G 份）
② Rollout    _generate_and_score_completions  (grpo_trainer.py:2260)
③ 奖励       _calculate_rewards                (grpo_trainer.py:1612)
④ 优势估计   compute_loss 内嵌（组内 z-score）(grpo_trainer.py:2685-2699)
⑤ 损失更新   _compute_loss → dapo_loss 等      (grpo_trainer.py:2991, 3083-3170)
⑥ 评测       evaluation_loop / log_completions（wandb completions 表）
```

入口与缓冲机制：`_prepare_inputs`（`grpo_trainer.py:1555`）区分 train/eval——训练时每 `steps_per_generation × num_iterations` 步才真正生成一次，生成结果按 `steps_per_generation` 切成小批缓冲复用（省计算）；eval 时逐批生成不缓冲。这对应论文"一次采样、多次梯度更新（num_iterations=μ）"的设置。

## 2. 各环节的公式 ↔ 代码对照

### ② Rollout：`_generate_and_score_completions`（2260 行起）

- 按 `args.batch_size` 分批对 prompt 做 `model.generate`（temperature、top_p 由 config 控制），`num_generations`（默认 8）个完成并行生成；
- 记录 `prompt_ids`（右移一格供 logp 对齐）、`completion_ids`、`logprobs`（本次采样时的 old_logp，供重要性比使用）；
- `temperature <= 0` 时走贪心（本实验自写实现也采用同一约定）。

### ③ 奖励：`_calculate_rewards`（1612 行起）

- 从 `inputs[0]` 取出除 prompt/completion/completion_ids 之外的所有列（`keys` 循环），逐列广播给 `reward_funcs`——**数据集列与奖励函数签名对应**，如 `{"prompt": ..., "answer": ...}` 列喂给 `my_reward(completions, answer, **kwargs)`；
- 奖励函数可用 `log_extra`/`log_metric` 上报自定义列与指标；
- 结果 `rewards_per_func` 按权重加权求和（`reward_weights`），再送去做 `scale_rewards`。

对应 R1 的**规则奖励**：accuracy + format 两个函数、各有权重（本实验自写版即模仿 `0.9*acc + 0.1*fmt`）。

### ④ 优势估计（2685-2699 行）

```python
mean_grouped_rewards = torch.nanmean(rewards.view(-1, num_generations), dim=1)
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
...
std_rewards = nanstd(rewards.view(-1, num_generations), dim=1)  # scale_rewards="group"
...
advantages = (rewards - mean_grouped_rewards) / (std_rewards + 1e-4)
```

这正是 GRPO 论文式 (2) 的 $\hat A_i = \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})}$，用 `nanmean/nanstd` 处理含 NaN 的组（如格式分缺失）。`scale_rewards` 可选 `group` / `batch` / `none`。注意：**TRL 没有 value model**——优势只来自组内统计，这是 GRPO 区别于 PPO 的核心。

### ⑤ 损失：`_compute_loss`（2991 行起）→ 按 `loss_type` 分派

五种 loss 分支：`grpo` / `bnpo`（3083）、`dr_grpo`（同前）、`luspo`、`cispo`/`dapo`/`vespo`（3139）。以默认 `dapo` 为例，与论文公式逐项对应：

| 论文记号 | 代码（`dapo_loss`，grpo_trainer.py 内） |
|---|---|
| 重要性比 $r_i=\pi_\theta/\pi_{\rm old}$ | `log_ratio = per_token_logps - old_per_token_logps`；`coef_1 = torch.exp(log_ratio)` |
| 下界 clip $1-\varepsilon_{\rm low}$ | `coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)` |
| 两段式 loss $\min(r\hat A,\ \mathrm{clip}(r)\hat A)$ | `per_token_loss = -torch.min(per_token_loss1, per_token_loss2)` |
| KL 惩罚 $(\mathrm{clip\_higher})$ | `per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1`（k3 估计子） |
| 长度归一 | `loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1)`（dapo 分支再除全局 token 数，3167 行） |
| 熵正则 | 配置项 `entropy_coef`，与 `loss_type` 无关 |

DAPO 相对原版 GRPO 的改进全部落在配置项上：`loss_type="dapo"`（默认！）、`epsilon_low=0.28`（原 0.2，见 `GRPOConfig.epsilon`，默认值 0.2，DAPO 推荐 0.28）、`dapo_kl_coef` 高/低 KL 双向钳制、`importance_sampling_level="token"`。**"默认即 DAPO"本身就是一个证据：2024 年 GRPO 论文方法 → 2025 年 DAPO 改进 → 2026 年成为训练框架默认，完整走了一遍"论文→代码"的技术演进。**

Dr. GRPO（arXiv:2503.20783，R1-Zero 的批判视角）进入代码的方式：`loss_type="dr_grpo"` 时 `coef_2 = (coef_2 / log_ratio.sum()) * log_ratio`（消除每 token 被 clip 不同次数造成的偏差，代码约 3160-3185 行区域）。OPSM（outcome-level per-sample mask，arXiv:2504.23336）则是 `importance_sampling_level="sequence"` 的选项（903 行可见 level 校验），实现"每个样本一次整体比例"的估计。

### 更新与调度

`create_optimizer_and_scheduler`（继承 HF Trainer）：AdamW + warmup/衰减由 `GRPOConfig`（grpo_config.py）控制；`training_step` 里 `num_iterations`（μ）决定每批梯度步数；`_step % steps_per_generation == 0` 触发下一次生成（1555 行处 `generate_every`）。

### ⑥ 评测

`evaluation_loop` + `log_completions`：评测集上贪心生成，`completion_length`（均值）与 reward 均值写入 metrics；`log_completions_table` 时生成 wandb 表格供人工检查。对应大作业要求的"记录输出长度"——TRL 把"平均完成长度"作为默认指标，正是 RL 训练"答案变长/变啰嗦"现象的监控点。

## 3. 关键配置默认值（GRPOConfig，grpo_config.py）

`beta=0.0`（TRL 中显式 KL 惩罚由 `loss_type` 内置，k3 估计子替代了 DeepSeekMath 原始公式的 `beta * KL` 项）；`num_generations=8`；`loss_type="dapo"`；`scale_rewards="group"`；`epsilon=0.2`；`epsilon_high=None`；`delta=None`（INTELLECT-2 双面截断开关）；`sapo_temperature_neg/pos=1.05/1.0`（SAPO，arXiv:2511.20347——v1.9.2 已收录 2025 年底的论文，可见代码库演进速度）。

## 4. 对照：DPO 训练器的一段式实现

DPO 是"无 rollout 的偏好拟合"，管线短得多：`DPOTrainer.compute_loss`（dpo_trainer.py:1725）→ `_compute_loss`（1329 行）——取 chosen/rejected 两列、计算 `beta * (logp_chosen - logp_rejected) - beta * (ref_chosen - ref_rejected)` 后接 `F.logsigmoid` 取负。**没有**生成、没有优势估计、没有组内标准化。GRPO 与 DPO 的代码结构差异，直观印证了思考题"在线/离线"的分界：DPO 的数据在 `Dataset` 里，GRPO 的数据在 `generate` 里。

（本实验自写实现刻意复刻了同一套管线：`sample_pairs`≈Rollout+奖励，`dpo_loss`≈DPOTrainer 公式，`grpo_train`≈④⑤两步——见 `exp/run_experiment.py` 内注释与第 2 节行号对照。）
