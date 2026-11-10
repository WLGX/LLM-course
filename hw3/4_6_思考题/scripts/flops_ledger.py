# -*- coding: utf-8 -*-
"""P140 手算账本：F_train ≈ 6*N_block*D + 12*L*D*s*d 对 "jev" 类 System One 决策模型需要做哪些修改。

纯算术核对（无训练）：先复现课件算例验证公式，再按公开的 System One 类模型特征
（Jev/TypeSafe、StartLux-Decision 公开口径：MoE 35B-A3B、flash-linear-attention、
双向注意力、单次前向多问题并行、RLCD）逐项改写，输出 results/flops_ledger.txt。
所有假设显式标注：未公开论文的架构细节一律按"示意配置"处理并注明。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(OUT, exist_ok=True)

L = []


def line(s=""):
    L.append(s)


def p(s=""):
    line(s)
    print(s)


# ---------- 0. 课件公式与算例复核 ----------
p("=" * 72)
p("0. 课件公式复核（P113/P140 算例，应与幻灯片数字一致）")
p("=" * 72)
Lc, dc, D = 32, 4096, 1_000_000_000_000  # L=32, d=4096, D=1e12 tokens
N_block = 12 * Lc * dc ** 2
F_param = 6 * N_block * D
for s in (4096, 32768):
    F_attn = 12 * Lc * D * s * dc
    p(f"  s={s:6d}: 6*N_block*D = {F_param:.3e}  12*L*D*s*d = {F_attn:.3e}  "
      f"比值 = {F_attn / F_param:.4f}（课件：1/6=0.1667，4/3=1.3333）")
p("  比值解析式 = s/(6d)：s=4096 -> 1/6；s=32768 -> 4/3。 ✓ 与幻灯片一致")

# ---------- 1. 逐项修改 ----------
p()
p("=" * 72)
p("1. 公式逐项修改：System One 决策模型 vs 课件假设")
p("=" * 72)
p("课件公式的三条隐含假设：")
p("  (a) N_block=12Ld^2：稠密 FFN，每个 token 触碰全部参数；")
p("  (b) 12LDsd：MHA、完整方阵配对（4LBs^2d 前向 × 3）；")
p("  (c) 只覆盖预训练前反向主项（6ND），不含 RL 采样与词表头。")
p("System One 类模型（Jev/StartLux 公开口径）逐条打破：")

p()
p("1.1 输出不再是「下一个 token」：无自回归生成，词表头消失")
p("  经典：词表头参数 V*d、每 token 输出 2*V*d FLOPs（V~10 万，d=4096 -> 参数 4 亿、")
p("        每 token 8.2e8 FLOPs）。")
p("  决策模型：choice/score/noul 三类有界头，输出维度 = 候选数 c 或 1，")
p("        每 token 输出 2*d*(c+1) 量级（c 常为个位数 -> 比词表头小 4 个数量级），")
p("        且 K 个问题共享一次编码（成本摊薄 1/K）。")
p("  => 6ND 中的 N 去掉 V*d 项；新增的输出头项可忽略，但需在账本中显式列明。")

p()
p("1.2 MoE：每个 token 只触碰激活专家 -> 6*N_block*D 改用激活参数")
p("  课件 N_block=12Ld^2 是稠密口径；MoE 下：")
p("    N_total = L*(12d^2 + 2*E*d*d_ff + 路由)      （全部专家，决定显存与下载）")
p("    N_active= L*(12d^2 + 2*k*d*d_ff)              （k 个激活专家，决定每 token 计算）")
p("  StartLux 公开口径 35B-A3B：激活比 = 3/35 ≈ 0.086，参数主项直接降为")
p("    6*N_active*D = 6*N_total*D*(3/35)，约为同参数稠密模型的 1/11.7。")

p()
p("1.3 GQA：K/V 投影与 KV 缓存收缩，但 s^2 配对项不变（课件 GQA 页明文）")
p("  课件 GQA 页：「KV 头数减少到 1/4，不等于整个注意力计算减少到 1/4」；")
p("  「query 的总宽度决定主要配对乘加，KV 头数决定存储宽度及 K、V 投影规模」；")
p("  即使多个 query 头共用同一 KV，各查询不同、分数与权重也不同，仍需要各算一套。")
p("  因此 12LDsd 的配对项原样保留；GQA 改的是另两本账：")
p("    (a) K/V 投影：每层每 token 的 K/V 投影 FLOPs 从 4d^2 变为 4d^2*(n_kv/h)，")
p("        N_block 从 12d^2 变为 12d^2 - 2d^2*(1 - n_kv/h)；n_kv/h=1/8 时约 10.25d^2。")
p("    (b) KV 缓存（存储账，不进 FLOPs 公式）：2*L*B*s*n_kv*hd*b，")
p("        n_kv/h=1/8 时为 MHA 的 1/8（课件 KV 缓存页的算例口径）。")

p()
p("1.4 线性注意力（flash-linear-attention）：s^2 项消失，配对项不再随 s 增长")
p("  线性注意力不物化 s^2 分数矩阵，每 token 每层做两次 d*m 量级的小乘加")
p("  （状态更新 k*v^T 与输出 q*S，m 为状态扩展维）：前向约 4*d*m，")
p("  按课件的「前向×3」口径，前反向合计约 12*L*D*d*m —— 与序列长度 s 无关：")
p("    F_attn(线性) = 12*L*D*d*m      （对照：F_attn(方阵) = 12*L*D*s*d）")
p("  等价地看：方阵项是 12LDsd，线性注意力把它变成 12LDdm，m=d 时等于")
p("  「方阵项固定在 s=d 的值」；s>m 时线性注意力才更省——注意这是方向性的：")
p("  短序列（s<m）时线性注意力反而更贵，不能无条件说线性=便宜。")
p("  同时「分数矩阵」显存从 L*B*h*s^2*b 变为状态量 L*B*h*d*m*b（s=8192、h=24、")
p("  b=2 字节时，单层单份分数矩阵从 3.2GB（不可接受）变为状态量（m=64 时约 4.7MB）。")

p()
p("1.5 双向注意力：去掉因果减半，但「多问题一次前向」重新摊薄")
p("  因果掩码可让配对约减半；StartLux 的 Instant 用双向注意力（完整方阵口径，不减半），")
p("  换取的是同一 state 上 K 个问题并行作答：编码只做一次，K 个问题各自从表示分支读出。")
p("  => 每问题的注意力成本 = 全量项 / K。这是「公式之外」的摊销，必须写进账本。")

p()
p("1.6 训练目标：RLCD 让 RL 采样成本进入公式，但单次前向使 rollouts 极便宜")
p("  课件公式只算预训练前反向。校准决策强化学习（RLCD）需要在线采样：")
p("    每步更新：G 次前向采样 + 1 次前反向训练 = (2G + 6) * N_active * s")
p("  关键差别：决策模型的「采样输出」是几个标量/概率，不是长 CoT 文本 ——")
p("  s 里没有数千 token 的思维链，G 路采样就是 G 次单前向。")
p("  口径说明：(2G+6)*N_active*s 只含参数主项；单次前向的注意力项（方阵 4LBs^2d，")
p("  线性 4LBdm）在 s 远小于 d 时是 2% 量级的小项，本算例未计入（与课件忽略注意力的")
p("  主项口径一致）。")
p("  对推理模型（System Two），同一公式里的 s 要换成 s_in + s_think（数千 token），")
p("  这是 System One 能把 RL 训练做成主训练手段的成本根源。")

# ---------- 2. 数值算例 ----------
p()
p("=" * 72)
p("2. 数值算例（示意配置，数字自洽；非未公开论文的真实逐层参数）")
p("=" * 72)
p("算例 S1（System One，方阵注意力 + GQA + MoE，示意）：")
p("  L=16, d=3072, h=24, n_kv=3(1/8), E=16 专家, k=2 激活, d_ff=3072")
L1, d1, nkv_ratio, E1, k1 = 16, 3072, 1 / 8, 16, 2
N_attn = (12 - 2 * (1 - nkv_ratio)) * L1 * d1 ** 2   # GQA 缩小 K/V 投影后
N_exp = 2 * E1 * d1 * d1 * L1
N_act = L1 * ((12 - 2 * (1 - nkv_ratio)) * d1 ** 2 + 2 * k1 * d1 * d1)
D1 = 500_000_000_000  # 0.5T tokens
p(f"  N_total = {N_attn/1e9:.2f}B + {N_exp/1e9:.2f}B ≈ {(N_attn+N_exp)/1e9:.1f}B；"
  f"N_active = {N_act/1e9:.2f}B（激活比 {N_act/(N_attn+N_exp):.2f}）")
F_param1 = 6 * N_act * D1
p(f"  F_param = 6*N_active*D = {F_param1:.2e}")
for s in (2048, 8192, 32768):
    F_attn_quad = 12 * L1 * D1 * s * d1   # 配对项与 GQA 无关（课件口径）
    F_attn_lin = 12 * L1 * D1 * d1 ** 2   # m=d 口径
    p(f"  s={s:6d}: 方阵配对项 {F_attn_quad:.2e}（占参数项 {F_attn_quad/F_param1:.3f}）；"
      f"线性注意力 {F_attn_lin:.2e}（{F_attn_lin/F_param1:.3f}，与 s 无关）")
p("  方向性结论：s=2048（<m=3072）时线性注意力反而更贵（0.140 vs 0.094），")
p("  长状态（s>>m）才显出结构性优势——256K 分块长状态场景正是后者。")

p()
p("算例 S2（训练目标拆分：预训练 vs RLCD 在线采样）：")
G, U, Q, s_rl = 8, 4, 1_000_000, 512
F_rl = Q * U * (2 * G + 6) * N_act * s_rl
p(f"  G={G} 路采样、每状态更新 U={U} 次、Q={Q} 状态、典型状态 s={s_rl}：")
p(f"    F_RLCD = Q*U*(2G+6)*N_active*s = {F_rl:.2e}")
p(f"    占 S1 预训练主项的 {F_rl/F_param1*100:.1f}% —— 单次前向让在线 RL 可负担。")
p("  对照（同一个公式，s 换成 s + t_think）：System One 的 t_think≈0（输出就是读出几个标量）；")
p("  带 2000 token 思维链的 System Two：")
F_rl2 = Q * U * (2 * G + 6) * N_act * (s_rl + 2000)
p(f"    F_RL_CoT = Q*U*(2G+6)*N_active*(s+2000) = {F_rl2:.2e}（{(s_rl+2000)/s_rl:.1f} 倍）")
p("    倍数 = (s+t_think)/s：状态越短、思维链越长越极端（s=128、t=8000 -> 63.5 倍）。")
p("    这就是「决策模型才敢把 RL 当主训练手段」的成本根源。")
p("  附：AutoThink 式置信度引导的自适应推理 = 每问题额外前向次数变成随机变量，")
p("    公式再乘 (1 + E[额外前向])，属于可控的「思考预算」项，同样需要单独记账。")

p()
p("算例 S3（部署侧，训练预算与部署选择回答不同问题）：")
p("  LLM 推理：每 query ≈ 2*N*(s_in + s_out)，取 N=7B、s_in=2048、s_out=500：")
F_llm = 2 * 7_000_000_000 * (2048 + 500)
p(f"    F_LLM = {F_llm:.2e}")
p("  System One：每问题 ≈ 2*N_active*s_in/K（K 个问题摊薄一次编码），K=10、s=4096：")
F_s1 = 2 * N_act * 4096 / 10
p(f"    F_S1  = {F_s1:.2e}（约为 LLM 的 1/{F_llm/F_s1:.0f}）——与 Jev 公开的"
  "~$0.042/百万输入 token、输出免费的价格结构同向")

p()
p("=" * 72)
p("3. 结论：修改后的公式（System One 决策模型训练账本）")
p("=" * 72)
p("F_train ≈ 6*N_active*D")
p("        + 12*L*D*d*m            （线性注意力，m 为状态扩展维；方阵注意力则为 12*L*D*s*d，")
p("                                GQA 不改变配对项，只把 K/V 投影缩进 N_active）")
p("        + 输出头小项（2*d*(c+1)*K，可忽略但需列明）")
p("        + Q*U*(2G+6)*N_active*s   （RLCD 在线采样，若训练含 RL）")
p("其中 N_active = L*((12 - 2*(1 - n_kv/h))*d^2 + 2*k*d*d_ff)（MoE 激活口径，")
p("K/V 投影已按 n_kv/h 缩小）；双向注意力不减半；KV 缓存 2*L*B*s*n_kv*hd*b 是存储账。")
p("K 个问题共享编码的成本摊薄单独记账。6ND 的「ND 主项结构」不变，变的是每一项的系数与变量。")

with open(os.path.join(OUT, "flops_ledger.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("\nsaved ->", os.path.join(OUT, "flops_ledger.txt"))
