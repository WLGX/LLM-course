# hw4：第三讲作业（2026 秋《大语言模型与应用实践》）

本文件夹包含两项作业，均为**真实运行**的实验/调研：

1. `4_1_诗词英译提示实验/` — 唐诗宋词英译的提示实验（L3 part1 第 89–90 张）。
2. `4_2_提示词泄露与攻击/` — 系统提示词泄露与提示词攻击调研（L3 part1 第 48–49 张延伸的自选题）。
3. `piazza/` — 两份 Piazza 提交帖草稿。

## 依赖

- Python 3，`requests`（唯一第三方依赖）。
- DeepSeek API（OpenAI 兼容接口，默认 `https://api.deepseek.com/chat/completions`，模型 `deepseek-chat`）。
- API key 放在 `F:/LLM-course/.env`（内容 `DEEPSEEK_API_KEY=sk-xxx`）或设为环境变量；该文件**不提交 git**（见仓库 `.gitignore`）。

## 4_1 诗词英译提示实验

三个提示版本（A 直接翻译 / B 加入任务标准 / C 只加入边界示例）× 2 次重复，dev 集 4 首、test 集 12 首，全部原始请求/响应存档。评价协议（含盲评规则与格式检查）见 `protocol_评分协议.md`，**先于实验运行定稿**。

```bash
cd /f/LLM-course/hw4/4_1_诗词英译提示实验
HW4_SPLIT=dev python run_generate.py    # 生成（原始存档 results/raw/raw_dev.jsonl）
HW4_SPLIT=dev python run_evaluate.py    # 盲评（results/scores/scores_dev.jsonl）
HW4_SPLIT=dev python analyze.py         # 汇总（results/汇总_dev.md）
HW4_SPLIT=dev python category_analysis.py  # 按类别汇总（results/类别汇总_dev.md）
# 确认 dev 流程无误后，把 dev 换成 test 重跑一遍
```

产出物：`实验报告.md`、`results/`（原始生成、盲评打分、汇总表格）。

## 4_2 提示词泄露与攻击

固定虚构系统提示词（青禾翻译助手，含保密规则与虚构密钥），15 条攻击/对照用例（注入 ×6、提取 ×6、越狱 ×2、对照 ×1），关键词初判 + 人工逐条证据定级。

```bash
cd /f/LLM-course/hw4/4_2_提示词泄露与攻击
python run_attacks.py        # 实测（results/raw_attacks.jsonl）
python analyze_attacks.py    # 关键词初判（results/classify_attacks.md）
```

产出物：`调研报告.md`（文献综述 + 实测证据，二者明确区分）。

## 提交

- Piazza 帖草稿见 `piazza/`；GitHub 仓库：https://github.com/WLGX/LLM-course
- 提交前把本文件夹镜像到仓库克隆（`C:/Users/15658/Documents/Qoder/2026-10-31/b785fe78/LLM-course/hw4/`）并 commit/push。
