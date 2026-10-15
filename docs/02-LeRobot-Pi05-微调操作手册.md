# LeRobot π₀.₅（Pi05）微调操作手册

> **项目记录说明**：本手册包含当时的操作步骤、推荐配置与排错分析，并不代表本项目所有步骤都按文中建议执行。本次实际训练参数以[公开模型的 `train_config.json`](https://huggingface.co/Shanshan777/lerobot_pi05_01/blob/main/train_config.json)为准；真机失败结果与后续验证计划见[项目 README](../README.md)及[失败复盘](./失败复盘与改进计划.md)。

> **Featurize RTX 4090 24G 全流程**：环境配置 → 数据准备 → 训练 → 推理（远程 / 本地双方案）
>
> **适用数据集**：本机 `D:\Desktop\datass`，40 条示教、17,961 帧、2 路相机、6 维 state/action、任务文本 `Grab Box`、`codebase_version: v3.0`。
>
> **本机环境**：Windows + RTX 3050 Laptop（**4 GB 显存**，驱动 546.30）—— 这个显存数字决定了第 6 章的全部结论。
>
> 本文是**自包含全流程手册**，不需要先读其他文档。所有结论都对照了本机 LeRobot 源码（标注文件与行号）与官方 `docs/source/pi05.mdx`、`rename_map.mdx`、`async.mdx`。

---

## 0. 三十秒结论

| 问题            | 答案                                                                                                 |
| ------------- | -------------------------------------------------------------------------------------------------- |
| 本机能训练 π₀.₅ 吗？ | **不能**。3.3B 参数，本机 4 GB 显存。必须上云。                                                                    |
| 本机能推理 π₀.₅ 吗？ | **不能**。bf16 权重就要 ~7 GB。**走第 5 章远程推理**。                                                             |
| 云上选什么卡？       | **RTX 4090 24G**（¥1.87/小时）。够用，但**只能冻结 VLM 微调**。                                                    |
| 数据要改吗？        | **不用改**。同一份 `datass` 直接微调。                                                                         |
| 要重算统计量吗？      | **不用**。`meta/stats.json` 里已有 `q01`/`q99`，正好满足 π₀.₅ 的 `QUANTILES` 归一化。                              |
| 相机键名要处理吗？     | **训练时要**（`--rename_map`）。**远程推理时不需要** —— 直接在相机配置里写策略键名。                                            |
| 要补空相机吗？       | **不需要**。第 3 路自动补齐（`modeling_pi05.py:1243`）。                                                        |
| 算力花多少钱？       | 训练 ¥5~~15；每做一次真机实验再 ¥2~~5（实例保持运行的机时费）。                                                             |
| 权重怎么长期保存？     | **托管到 Hugging Face**（§4.1）。实例磁盘随时会没，`/home/featurize/work` 只有 30 GB 免费额度。                          |
| 最大的坑是什么？      | ① `--policy.path` 与 `--policy.pretrained_path` 语义不同；② 远程推理的图像预处理与训练**不一致**，必须打补丁（§5.7）；③ 上行带宽可能不够。 |

---

## 1. 环境配置（Featurize 云端，约 40 分钟）

### 1.1 为什么必须上云

|              | 本机 RTX 3050 Laptop | 云端 RTX 4090          |
| ------------ | ------------------ | -------------------- |
| 显存           | **4 GB**           | 24 GB                |
| π₀.₅ bf16 权重 | 需要 ~7 GB ❌         | ✅                    |
| π₀.₅ 训练      | 需要 >24 GB ❌        | 冻结 VLM 后可训 ✅         |
| 配套内存/磁盘      | 笔记本                | 60 GB 内存 / 750 GB 磁盘 |

> π₀.₅ 是 **3.3 B 参数**的 VLA 模型（详见《03-Pi05-实现细节与原理详解》§3 的逐层手算）。BF16 权重体积 ≈ 3.3e9 × 2 B ≈ **6.6 GB**，光是把权重塞进显存就超过本机 4 GB。

### 1.2 注册与开机

1. 注册 <https://featurize.cn> → 实名认证 → 充值（首充满 5 元送 1 小时体验券）。
2. **学生认证**（<https://docs.featurize.cn/docs/activity/stu-authorization>）：通过后发放 **RTX 4090 单卡 9.9 元日租券**，有效期 24 小时。你是学生的话这一步能省不少钱。
3. 新建实例：

| 项目   | 选择                               | 说明                                                                 |
| ---- | -------------------------------- | ------------------------------------------------------------------ |
| 显卡   | **RTX 4090 / 24 GB × 1**         | ¥1.87/小时，16 核 EPYC 9354 + 60 GB 内存                                 |
| 镜像   | PyTorch 2.x + CUDA 12.x          | **镜像 = 预装环境（OS + 驱动 + 框架一体）**，GPU 实例都是 Ubuntu 系 Linux，**不需要单独选系统** |
| 磁盘   | 默认（750 GB 免费）                    |                                                                    |
| 持久化盘 | `/home/featurize/work`（30 GB 免费） | **只有这个目录在关机后保留**                                                   |

> ⚠️ **30 GB 够不够？** 算一笔账：π₀.₅ 基础权重 ~7 GB + 你的数据集 0.37 GB + 一个训练 checkpoint ~7 GB × N 个。`save_freq=1000` 跑 5000 步会存 5 个 checkpoint = 35 GB，**直接爆盘**。对策见 §3.7。

1. 开机后打开「工作区」，拿到 **SSH 登录信息**（`workspace.featurize.cn`、用户名 `featurize`、**端口号**、密码）。这三样第 5 章要用。

### 1.3 上传数据集

本机 `D:\Desktop\datass` 约 367 MB（Windows 显示 351 MiB，是 MiB/MB 单位差异，不是缺文件）。

**打包：**

```cmd
cd /d D:\Desktop
tar -czf datass.tar.gz datass
```

> Windows 10 1803+ 自带 `tar`。没有的话装 Git for Windows 后用 Git Bash。
>
> 打包前确认没有 `.DS_Store`、`Thumbs.db` 之类的垃圾文件混进去。

**上传到持久化盘**（三种方式任选）：

| 方式                | 操作                                            | 适合            |
| ----------------- | --------------------------------------------- | ------------- |
| **JupyterLab 拖拽** | 工作区 → 进入 `/home/featurize/work` → 拖文件进去       | 最简单，367 MB 够用 |
| VSCode 远程         | 连上实例后把文件拖进左侧文件树                               | 大文件更稳         |
| Featurize CLI     | 见 <https://docs.featurize.cn/docs/manual/cli> | 有脚本化需求时       |

> ⚠️ **数据集是单向的**：能本地上传 → 实例下载，**不能反向**。所以训练产物（权重）要走第 4 章的办法取回，或者干脆不取（走远程推理）。

### 1.4 在实例里解压数据

SSH 或 JupyterLab 终端里：

```bash
cd /home/featurize/work
mkdir -p datasets && tar -xzf datass.tar.gz -C datasets
ls datasets/datass/meta        # 应看到 info.json / stats.json / tasks.parquet / episodes/
```

**验证数据完整性**（可选，但建议跑一次）：

```bash
python - <<'EOF'
import json
d = json.load(open('/home/featurize/work/datasets/datass/meta/info.json'))
print('episodes       =', d['total_episodes'])
print('frames         =', d['total_frames'])
print('fps            =', d['fps'])
print('codebase_ver   =', d['codebase_version'])
print('features       =', [k for k in d['features'] if 'images' in k or 'state' in k or k == 'action'])
EOF
```

期望输出：`episodes = 40`、`frames = 17961`、`fps = 30`、`codebase_version = v3.0`。

### 1.5 装 LeRobot 环境

**第 0 步：先看镜像自带的 torch 能不能用**（决定了后面走哪条路）：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda :', torch.version.cuda); print('avail:', torch.cuda.is_available())"
```

#### 路线 A（推荐）：直接复用镜像自带环境

Featurize 的 PyTorch 镜像里 **torch 已经与实例驱动匹配好**。复用它可以省掉一个很大的坑（见下面的警告）。

```bash
# ① 拿代码
cd /home/featurize/work
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# ② 只补装 lerobot 及其额外依赖；torch 已满足版本要求时 pip 不会动它
pip install -e ".[pi,training,async]"

# ③ 视频解码依赖（AV1 解码必需）
conda install -y ffmpeg -c conda-forge      # 或：apt-get install -y ffmpeg
```

装完回到第 0 步的命令再确认一次 `avail: True`。

#### 路线 B：建独立环境（想隔离时用）

```bash
conda create -n pi05 python=3.12 -y
conda activate pi05
cd /home/featurize/work
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[pi,training,async]"
conda install -y ffmpeg -c conda-forge
```

> ⚠️ **路线 B 的坑（重要）**：lerobot 的 `pyproject.toml:373-376` 在 Linux 上把 torch 指向 **cu128** 索引，而注释里写了 *"cu128 wheels keep broad hardware reach; **the driver floor is 570.86**"*。也就是说新建环境时 pip 可能会拉一个**要求驱动 ≥570.86** 的 CUDA 12.8 wheel。
>
> 如果你的 Featurize 实例驱动低于这个版本，新环境里的 torch 会长这样：
>
> ```
> torch: 2.x.x+cu128
> avail: False          ← 典型的"装了用不了"
> ```
>
> 这时看驱动版本再对号入座重装：
>
> ```bash
> nvidia-smi | head -3          # 右上角 Driver Version
> # 例：驱动 550.x → 用 cu124
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
> # 例：驱动 546.xx → 用 cu121
> pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```
>
> **不确定就老老实实走路线 A** —— 镜像自带的 torch 一定是能用的。

**三个 extra 都是干什么的 —— 别只装 `[pi]`：**

| extra      | 提供                                              | 不装的后果                                                                                                                                         |
| ---------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `pi`       | `transformers` / `scipy` / `sentencepiece`      | π₀.₅ 模型代码 import 就失败                                                                                                                          |
| `training` | `datasets` / `pyarrow` / `accelerate` / `wandb` | 训练启动时缺包                                                                                                                                       |
| `async`    | `grpcio` + `matplotlib`                         | 远程推理起不来（`require_package("grpcio", extra="async")`，见 `async_inference/__init__.py:28`）；没有 matplotlib 则 `--debug_visualize_queue_size` 画不出队列曲线 |

> **坑**：`[pi]` 不含 `training` 的内容，`[training]` 也不含 `pi` 的内容。**必须一起写**。
>
> 记法：**训练用 `.[pi,training]`；训练 + 远程推理用 `.[pi,training,async]`。**

### 1.6 ⚠️ 门控分词器：先接受许可协议

π₀.₅ 写死了使用**受门控（gated）**&#x7684; `google/paligemma-3b-pt-224` 分词器（源码 `processor_pi05.py:154`，官方 `pi05.mdx` 也专门提示过）：

1. 打开 <https://huggingface.co/google/paligemma-3b-pt-224>，登录 → **接受许可协议**（用 `google/gemma-*` 系列的同一个许可）。
2. 回到实例：

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # 粘贴你的 HF token（需要 read 权限）
```



> **提示**：走 `hf-mirror.com` 镜像时，**门控仓库可能拉不下来** —— 这一步建议**直连官方站点**。报 401 / 403 / `GatedRepoError` 基本都是这里没做全。
>
> 如果拉 tokenizer 时发现是直连慢，可以这样组合：**tokenizer 直连下载到本地目录，其余走镜像**。
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com   # 其余下载走镜像
> ```

### 1.7 预下载基础权重（强烈建议先做）

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/featurize/work/hf          # ★ 让缓存落在持久化盘，下次开实例不用重下

python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "lerobot/pi05_base",
    allow_patterns=["*.json", "*.safetensors", "*.md", "*.txt"],
)
print("downloaded to:", p)
EOF

du -sh /home/featurize/work/hf
```

> **为什么必须设 `HF_HOME`**：默认缓存在 `~/.cache/huggingface`，**关机就没了**。π₀.₅ 基础权重 7 GB 级，每次开实例重下要多花 10~20 分钟。放到 `/home/featurize/work/hf` 后，第二次开机直接命中缓存。
>
> 先看清实际大小再决定要不要下沉到持久盘（30 GB 限额）：
>
> ```bash
> du -sh /home/featurize/work/hf/hub/models--lerobot--pi05_base
> ```

### 1.8 环境自检

```bash
python - <<'EOF'
import torch, transformers, grpc
print("torch        :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("device       :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("vram         :", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), "GB")
print("transformers :", transformers.__version__)
print("grpc         :", grpc.__version__)
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
from lerobot.policies.pi05.configuration_pi05 import PI05Config
print("pi05 import  : OK")
EOF
```

期望看到 `cuda: True`、`vram: 24.0 GB`（4090）、`pi05 import: OK`。

> 如果 `cuda: False` —— Featurize 的 PyTorch 镜像一般不会出这问题。真遇到就是镜像选错了（选到 CPU-only 镜像），重建实例。

### 1.9 wandb（可选，但推荐）

```bash
pip install wandb
wandb login          # 粘贴 API key
```

不登录也能训练，加 `--wandb.enable=false` 即可（或直接不传该参数，默认关闭）。

---

## 2. 数据侧准备

### 2.1 先查清 `pi05_base` 到底要哪些相机键名（**不要靠猜**）

`lerobot/pi05_base` 的 `config.json` 实测原文（关键部分）：

```json
{
  "type": "pi05",
  "input_features": {
    "observation.images.base_0_rgb":       { "type": "VISUAL", "shape": [3, 224, 224] },
    "observation.images.left_wrist_0_rgb": { "type": "VISUAL", "shape": [3, 224, 224] },
    "observation.images.right_wrist_0_rgb":{ "type": "VISUAL", "shape": [3, 224, 224] },
    "observation.state":                   { "type": "STATE",  "shape": [32] }
  },
  "output_features": { "action": { "type": "ACTION", "shape": [32] } },
  "paligemma_variant": "gemma_2b",
  "action_expert_variant": "gemma_300m",
  "dtype": "float32",
  "chunk_size": 50,
  "n_action_steps": 50,
  "max_action_dim": 32,
  "max_state_dim": 32,
  "num_inference_steps": 10,
  "image_resolution": [224, 224],
  "normalization_mapping": {
    "VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"
  },
  "tokenizer_max_length": 200,
  "optimizer_lr": 2.5e-05,
  "optimizer_betas": [0.9, 0.95],
  "optimizer_weight_decay": 0.01,
  "optimizer_grad_clip_norm": 1.0,
  "scheduler_warmup_steps": 1000,
  "scheduler_decay_steps": 30000,
  "scheduler_decay_lr": 2.5e-06
}
```


**结论：它要 3 路相机，你只有 2 路（`front` / `wrist`）。**

想自己在实例上核对（输出会变就以它为准）：

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
import json
c = json.load(open(hf_hub_download('lerobot/pi05_base', 'config.json')))
print(json.dumps(c.get('input_features'), indent=2, ensure_ascii=False))
EOF
```

### 2.2 `--rename_map`：把数据集键名映射到策略期望的键名

官方约定（`docs/source/rename_map.mdx`）：

```
--rename_map='{"数据集提供的键": "策略期望的键", ...}'
```

**左边写你的数据集，右边写策略要的。** 只重命名 observation 键（图像 + state），**action 键不受影响**。

你的映射：

```json
{
  "observation.images.front": "observation.images.base_0_rgb",
  "observation.images.wrist": "observation.images.left_wrist_0_rgb"
}
```

> **为什么必须 rename，而不是"随便叫什么都行"？**
>
> 两个原因，第二个才是本质：
>
> 1. 策略的 `input_features` 是一组**固定键名**，缺键就是缺输入。
> 2. 更关键的是**槽位顺序**。图像令牌是按 `config.image_features` 的字典顺序**依次拼接**进序列的。π₀.₅ 预训练时学到的语义是「**第 0 路 = 主视角，第 1 路 = 腕部**」。你把前视映射成 `base_0_rgb`、腕部映射成 `left_wrist_0_rgb`，就**同时把顺序钉死在预训练约定上** —— 如果反过来写，两路相机的语义就互换了，效果会明显变差。

### 2.3 ⚠️ `empty_cameras` 到底要不要加？

π₀.₅ 有**两个不同**的"补空相机"机制，很容易混淆：

| 机制                               | 触发条件          | 效果                                                                      |
| -------------------------------- | ------------- | ----------------------------------------------------------------------- |
| **① `--policy.empty_cameras=N`** | 你在配置里**主动声明** | 在 `input_features` 里**新增** N 个占位键：`observation.images.empty_camera_0` … |
| **② 声明了但 batch 里没有**             | **自动发生**      | `_preprocess_images` 对每个缺失的已声明相机，**自动**填一张 `-1` 的图 + **零注意力掩码**         |

```python
# src/lerobot/policies/pi05/modeling_pi05.py:1238-1243
# Create image features not present in the batch as fully 0 padded images
for _num_empty_cameras in range(len(missing_img_keys)):
    img = torch.ones_like(img) * -1   # Padded with -1 for SigLIP
    mask = torch.zeros_like(mask)     # Mask is zero for empty cameras
    images.append(img)
    images_mask.append(mask)
```

**你的情况怎么算：**

- 用 `--policy.path=lerobot/pi05_base` 时，策略声明 3 路。
- rename 之后你提供 2 路 → **`right_wrist_0_rgb` 被机制 ② 自动补空**。
- **总共正好 3 路，与 π₀.₅ 预训练布局完全一致。** ✅

**所以：`--policy.empty_cameras` 保持默认 `0` 即可，不要传。**

> 如果硬设成 `1`，会变成 **4 路**（3 路预训练相机 + 1 个 `empty_camera_0`），**反而偏离预训练布局**，多送一个无信息的空槽位进去。
>
> 那官方 `pi05.mdx` 里的 `--policy.empty_cameras=1` 是为谁准备的？是给 `pi05_libero_base` 的：那个 checkpoint 只声明 **2 路**（`observation.images.image` / `image2`），所以要用 `empty_cameras=1` 补到 3 路。**它的目标同样是"凑到 3 路"**，和我们对齐预训练布局的目的一致 —— 只是起点不同、手段不同。

### 2.4 统计量：不用重算

π₀.₅ 对 `STATE` / `ACTION` 用 **`QUANTILES`** 归一化，因此 `meta/stats.json` 里必须**同时有 `q01` 和 `q99`**。缺了会在第一个 batch 就报错：

```
ValueError: QUANTILES normalization mode requires q01 and q99 stats
```

**实测确认你的数据集已有：**

```
action              -> ['count','max','mean','min','q01','q10','q50','q90','q99','std']
observation.state   -> ['count','max','mean','min','q01','q10','q50','q90','q99','std']
```

**所以什么都不用做，直接开训。** ✅

> 旧数据集（v2.1 转过来的）常常缺 q01/q99，那时才需要：
>
> ```bash
> lerobot-edit-dataset \
>   --repo_id local/datass \
>   --new_repo_id local/datass \
>   --operation.type recompute_stats \
>   --operation.overwrite true
> ```
>
> **你不属于这种情况。**
>
> 另外注意：用 `lerobot-record` 录制时，聚合得到的是**保守包络**（`q<=50` 取 `min`、`q>50` 取 `max`）而非全体分位数；要估计真正的分位数可以跑 `src/lerobot/scripts/augment_dataset_quantile_stats.py`（加 `--skip-images` 可跳过视频解码）。

### 2.5 维度与图像会自动处理

- 6 维 state / action 会被 padding 到 `max_state_dim` / `max_action_dim`（都是 **32**），**loss 只算前 6 维**（源码 `modeling_pi05.py:1387-1389`）。
- 480×640 的图像会被**等比缩放 + 居中黑边填充（letterbox）**&#x5230; 224×224，由 `resize_with_pad_torch` 完成（`vla_utils.py:141`，openpi 的精确拷贝）。
- 归一化到 `[0,1]` 后再映到 `[-1,1]` 喂给 SigLIP（`modeling_pi05.py:1222`）。

> ⚠️ **记住这个 letterbox**。第 5 章的远程推理**默认不做 letterbox**，需要打补丁 —— 这是本文最容易被忽视的一个坑。

---

## 3. 训练

### 3.1 ⚠️ 先搞清楚 `--policy.path` 和 `--policy.pretrained_path` 的区别

这是 π₀.₅ 微调**最容易踩的坑**，两者语义完全不同，源码依据：

```python
# src/lerobot/configs/train.py:197-215  _resolve_pretrained_from_cli()
policy_path = parser.get_path_arg("policy")
...
elif policy_path:
    overrides = parser.get_yaml_overrides("policy") + (parser.get_cli_overrides("policy") or [])
    self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=overrides)
    self.policy.pretrained_path = Path(policy_path)
```

| 写法                                                              | 配置从哪来                                       | 后果                                                                              |
| --------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------- |
| **`--policy.path=lerobot/pi05_base`**                           | **从 checkpoint 的 `config.json` 加载**，再用命令行覆盖 | `input_features` = 预训练的 **3 路相机** ✅ 完全复现预训练布局                                   |
| `--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base` | `PI05Config` 的**默认值**（权重只加载，不带配置）           | `input_features` 为空 → 由**你的数据集**推导 → 只有 **2 路相机**；需要自己补 `empty_cameras` 才凑回 3 路 |

> 依据：`PreTrainedPolicy.from_pretrained` 里 `if config is None: config = PreTrainedConfig.from_pretrained(...)`，而训练脚本总是把 CLI 解析出的 cfg 传进去 —— 所以**只有当走 `--policy.path` 这条路时，checkpoint 的配置才会被读进来**。官方 `pi05.mdx` 也明确写了：*`--policy.pretrained_path` loads weights only*。
>
> ⚠️ **二者互斥**：不能同时传 `--policy.path` 和 `--policy.type`，否则直接报错  
> `Cannot specify both --policy.path and --policy.type`（`configs/parser.py:311-315`）。

**因此本手册推荐第一种写法**（也是 `rename_map.mdx` 里 PI05/XVLA 的官方写法）：`--policy.path=lerobot/pi05_base`。

> **更正（源码复核后）**：`--rename_map` **两种加载方式都支持**，不存在"只能配 `--policy.path`"这回事。
>
> - 官方 `pi05.mdx` 对照表写得很明确：`--rename_map` 在 `pretrained_path` 这一列的说明是 *"supported for canonicalizing dataset keys"*（支持用于规范化数据集键名）。
> - 官方 RoboMME 微调命令就是 `--policy.pretrained_path` + `--rename_map` 一起用的。
> - 训练脚本会把 `rename_map` **一致地应用于三处**，所以不会出现"把键改到策略不认识"的情况：
>   1. 数据集特征名 → 策略特征名（`policies/factory.py:330-331`）
>   2. 归一化统计量的键名（`scripts/lerobot_train.py:500` → `rename_stats`）
>   3. 每个 batch 的键名（`scripts/lerobot_train.py:731` → `rename_batch_keys`）
>   4. 处理器管线里的 `rename_observations_processor` 覆盖项（`scripts/lerobot_train.py:512`）
>
> **两者的真正差别只在相机槽位数量**：`--policy.path` 从 checkpoint 拿到 **3 个**相机槽位（第 3 个缺数据时自动补 `-1` + 零掩码，`modeling_pi05.py:1239-1243`）；`--policy.pretrained_path` 只有你数据集里的 **2 个**。想要凑回 3 槽就加 `--policy.empty_cameras=1` —— 官方 LIBERO 示例（2 路相机）正是这么干的。

### 3.2 4090 24G 能跑多"开"？—— 官方给了明确答案

官方培训文档（ROBOTIS LeRobot Model Training Guide）原话：

> **"A full fine-tune does not fit on a 24 GB card such as an RTX 4090. Set `--policy.train_expert_only=true` there to freeze the VLM."**

翻译过来：**全参微调在 4090 24G 上放不下，必须开 `train_expert_only`。**

| 配置                                                      | 可训练参数                           | 4090 24G |
| ------------------------------------------------------- | ------------------------------- | -------- |
| 全参微调                                                    | 100%（≈3.3 B）                    | ❌ 放不下    |
| `freeze_vision_encoder=true`                            | ~88%（≈2.9 B）                    | ❌ 仍偏紧    |
| **`train_expert_only=true`**                            | **~13%（≈430 M）**                | ✅ **推荐** |
| `train_expert_only=true` + `freeze_vision_encoder=true` | ~13%（同左，视觉塔本来就被 expert_only 冻住） | ✅ 最省     |

> `train_expert_only` 冻结整个 VLM（PaliGemma 主干 + SigLIP），**只训 action expert 与投影层**。这不只是省显存 —— 它同时是一种**强正则**（不动 VLM 的世界知识），对 40 条这种小数据量反而有好处。
>
> 想要真正的全参微调效果，得换 **PRO 6000 96 GB**（¥6.19/小时），成本翻 3 倍以上。
>
> **为什么「全参」放不下？把账算给你看**：`--policy.dtype=bfloat16` 时，参数/梯度/`exp_avg`/`exp_avg_sq` **四份各占一份显存**（`optim/optimizers.py:226-231` 用的是裸 `torch.optim.AdamW`，动量 dtype 跟随参数，既没有 fp32 master weight 也没有 8-bit），≈7.5 GB × 4 ≈ **30 GB**。完整推导 + π0 同款结论见 **§3.11**。

### 3.3 第一步：冒烟测试（**必做**，先花 10 分钟排雷）

```bash
conda activate pi05                 # ← 走路线 A 复用镜像环境的话，这行删掉
cd /home/featurize/work/lerobot
export HF_HOME=/home/featurize/work/hf
export HF_ENDPOINT=https://hf-mirror.com

lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.path=lerobot/pi05_base \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_smoke \
  --job_name=pi05_smoke \
  --batch_size=4 \
  --steps=100 \
  --save_freq=100 \
  --log_freq=10 \
  --num_workers=8
```

**这 100 步要确认的四件事：**

1. 第一个 batch 不报 `Feature mismatch` / `QUANTILES normalization mode requires q01 and q99`；
2. 显存峰值 **明显低于 24 GB**（`nvidia-smi` 另开一个终端看，或看日志里的 `Mem`）；
3. 拿到 **step/s 实测值** —— 用它外推正式训练时间（见 §3.6）；
4. `outputs/train/pi05_smoke/checkpoints/000100/pretrained_model/` 里 7 个文件齐全。

**显存不够就按这个顺序往下砍：**

```
--policy.train_expert_only=true     ← 已经在用
--policy.gradient_checkpointing=true ← 已经在用
--policy.dtype=bfloat16             ← 已经在用
--batch_size=4 → 2 → 1              ← 继续砍这里
--num_workers 降到 4                ← 数据加载进程也吃内存
```

### 3.4 正式训练命令（4090 24G 推荐配置）

```bash
conda activate pi05                 # ← 走路线 A 复用镜像环境的话，这行删掉
cd /home/featurize/work/lerobot
export HF_HOME=/home/featurize/work/hf
export HF_ENDPOINT=https://hf-mirror.com

lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.path=lerobot/pi05_base \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_datass \
  --job_name=pi05_datass \
  --batch_size=8 \
  --steps=5000 \
  --save_freq=2500 \
  --log_freq=50 \
  --num_workers=10
```

**把它做成脚本 + nohup，防掉线：**

```bash
cat > /home/featurize/work/run_pi05.sh <<'EOF'
#!/bin/bash
set -e
conda activate pi05                 # ← 走路线 A 复用镜像环境的话，这行删掉
cd /home/featurize/work/lerobot
export HF_HOME=/home/featurize/work/hf
export HF_ENDPOINT=https://hf-mirror.com
lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.path=lerobot/pi05_base \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_datass \
  --job_name=pi05_datass \
  --batch_size=8 \
  --steps=5000 \
  --save_freq=2500 \
  --log_freq=50 \
  --num_workers=10
EOF
chmod +x /home/featurize/work/run_pi05.sh

# 后台跑 + 落日志
cd /home/featurize/work
nohup bash run_pi05.sh > /home/featurize/work/pi05_train.log 2>&1 &

# 看进度
tail -f /home/featurize/work/pi05_train.log

# 训练完发个微信通知（Featurize CLI 提供，需关注公众号）
featurize notify -t "π0.5 训练结束"
```


> ⚠️ **务必 nohup**：SSH 一断，前台进程就被 SIGHUP 杀掉了。写到 `work` 目录里的日志在断线后依然可读。

### 3.5 参数逐条解释

| 参数                                | 含义                                             | 建议值                                 |
| --------------------------------- | ---------------------------------------------- | ----------------------------------- |
| `--policy.path`                   | 预训练权重 + **其配置**（从 Hub 或本地目录）                   | `lerobot/pi05_base`                 |
| `--rename_map`                    | 观察键重命名（数据集键 → 策略键）                             | 见 §2.2                              |
| `--policy.train_expert_only`      | 冻结整个 VLM，**只训 action expert 与投影层**             | **`true`（4090 必需）**                 |
| `--policy.gradient_checkpointing` | 梯度检查点，显著降显存、略降速度                               | **`true`**                          |
| `--policy.freeze_vision_encoder`  | 单独冻结 SigLIP 视觉塔                                | 被 `train_expert_only` 覆盖，可省         |
| `--policy.dtype`                  | `bfloat16` / `float32`（只允许这两个）                 | **`bfloat16`**                      |
| `--policy.compile_model`          | `torch.compile` 加速（`modeling_pi05.py:563-567`） | **`false`（24 G 卡别开）**，见下方警告         |
| `--batch_size`                    | 批大小                                            | 4090 从 `8` 起探，OOM 就降到 4 / 2         |
| `--steps`                         | 训练步数                                           | **5000** 起（40 条数据，见 §3.7）           |
| `--save_freq`                     | 每多少步存一个 checkpoint                             | **2500**（**×7 GB，别设太小**，见 §3.7）     |
| `--num_workers`                   | 数据加载进程数                                        | 4090 实例 16 核 → **10**，OOM 内存时降到 4~8 |
| `--policy.push_to_hub`            | 自动推 Hub（预训练配置里默认是 `true`）                      | 显式设 **`false`**                     |

> ⚠️ **为什么 `--policy.compile_model` 建议设 `false`**：它会对 `forward` 和 `sample_actions` 同时做 `torch.compile(..., mode="max-autotune")`（`modeling_pi05.py:563-567`）。`max-autotune` 会在一个 3.3 B 模型上长时间搜内核，**编译期本身就可能吃满 24 GB 显存触发 OOM**；与 `gradient_checkpointing` 叠加时收益又很有限。你这次是几千步的短训练，编译预热占比过高，不开更划算。

**不用传的参数（传了反而错）：**

| 不用传                              | 原因                                         |
| -------------------------------- | ------------------------------------------ |
| `--policy.empty_cameras`         | 第 3 路相机会被自动补空，已经正好 3 路（§2.3）               |
| `--policy.normalization_mapping` | 默认就是 `QUANTILES`，且你的数据有 q01/q99（§2.4）      |
| `--policy.n_action_steps`        | `pi05_base` 配置里就是 50（因为走了 `--policy.path`） |
| `--policy.chunk_size`            | 同上，50                                      |

**学习率与调度器不用动**（`pi05_base` 配置里已是最优）：

| 项                          | 值           | 说明                                                                                       |
| -------------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| `optimizer_lr`             | **2.5e-5**  | 峰值学习率                                                                                    |
| `optimizer_betas`          | (0.9, 0.95) | β₂ 比常见的 0.999 小，是 VLM 微调的惯例                                                              |
| `optimizer_weight_decay`   | 0.01        |                                                                                          |
| `optimizer_grad_clip_norm` | **1.0**     | 比 ACT 的 10.0 严得多，说明 VLA 微调对梯度更敏感                                                         |
| `scheduler`                | 余弦衰减 + 预热   | warmup 1000 步，衰减到 2.5e-6，周期 30000 步                                                      |
| 自动缩放                       | ✅           | `--steps < scheduler_decay_steps` 时**自动等比缩放**（`steps=5000` → warmup 缩到 ~167、衰减周期缩到 5000） |

### 3.6 时间与成本预估

**先说结论：以 §3.3 冒烟测试的实测 step/s 为准，下面的数字只是量级参考。**

估算方法：

```
总时长 ≈ steps ÷ (实测 step/s)
成本   ≈ 总时长(小时) × ¥1.87
```

#### 实测锚点（你自己的日志，最可信）

你上一轮 π0.5 的日志里，**17:31 → 17:58 共 2625 步 ≈ 1.61 step/s**（5000 步 / batch 8 / 只训 expert / bf16 / 梯度检查点 / 4090 24G）。这个数字直接可以外推：

| 步数         | 时长（1.61 step/s） | 成本      |
| ---------- | --------------- | ------- |
| 3,000      | 31 分钟           | ¥1.0    |
| 5,000      | 52 分钟           | ¥1.6    |
| 8,000      | 83 分钟           | ¥2.6    |
| 15,000     | 2.6 小时          | ¥4.8    |
| **50,000** | **8.6 小时**      | **¥16** |

> **π0 会比 π0.5 再快一点点**：π0 的 `tokenizer_max_length=48`，而 π0.5 是 200（`configuration_pi0.py:103` / `configuration_pi05.py:87`）—— prefix 短 152 个 token。图像 token 占大头（3 路 × 256 = 768），所以总加速大约 10~15%，即 **π0 ≈ 1.75~1.85 step/s**。按 1.75 算，50000 步 ≈ **7.9 小时**。

**怎么拿到你自己的精确数字**：训练日志每条 `step:... smpl:... ep:... loss:...` 行前面都有时间戳，取两条相隔 200 步（默认 `log_freq`）的，**Δt 除以 200 就是 s/step**。或者直接看进度条右侧的 `it/s`。

#### 每步 = 多少 epoch？

这一步比时间更重要，因为**它决定了 `--steps` 该填多少**：

```
1 epoch = 数据集总帧数 ÷ batch_size = 17,961 ÷ 8 ≈ 2,245 步
```

> 交叉验证：你 `--steps=5000` 那次，日志里 `ep:2.23` —— 5000 ÷ 2245 = **2.23** ✅ 完全吻合。

| 步数     | epoch    | 说明                            |
| ------ | -------- | ----------------------------- |
| 3,000  | 1.34     | 官方 `pi0.mdx` 示例值，偏少但可用        |
| 5,000  | 2.23     | **你上一轮的量级**，loss 已降到 0.031    |
| 8,000  | **3.6**  | ★ **推荐**，小数据微调的性价比甜点          |
| 15,000 | 6.7      | 上限，再多就要靠真机成功率而不是 loss 来判断     |
| 50,000 | **22.3** | ⚠️ 40 条数据跑 22 个 epoch，过拟合几乎必然 |

> **外部参考（不同来源口径差异较大，仅作量级对照）**：
>
> - Pi0.5 完整微调（30,000 步）在 80 GB 卡上约 3~6 小时；
> - RTC 论文（arXiv 2506.07339 §A.6）的实机实验：**8×H100 上单次微调约 24 小时** —— 但他们用的是更大规模数据。
>
> **40 条数据 + 5000~8000 步，一小时量级是合理的。**

#### 顺带的固定开销

| 环节                        | 预估                          |
| ------------------------- | --------------------------- |
| 权重下载（首次，走镜像）              | 5~15 分钟                     |
| 环境安装                      | 10~20 分钟                    |
| `compile_model=true` 首次编译 | 2~5 分钟（`max-autotune`，只付一次） |

### 3.7 ⚠️ 40 条数据的过拟合风险 & 30 GB 磁盘风险

**风险一：过拟合。**

| 对比      | ACT                       | π₀.₅                |
| ------- | ------------------------- | ------------------- |
| 你的数据量   | 40 条 / 17,961 帧 / 约 10 分钟 | 同左                  |
| 模型参数量   | 51.67 M                   | ≈ 3.3 B（**大 64 倍**） |
| 官方微调数据量 | —                         | 通常数千~数万条 episode    |

**约 10 分钟示教去微调一个 33 亿参数的模型，过拟合风险是真实存在的。** 对策：

1. **步数别贪多。** 你的数据集 17,961 帧 ÷ batch 8 = **2,245 步就是 1 个 epoch**。所以：
   | `--steps` | epoch    | 判断                          |
   | --------- | -------- | --------------------------- |
   | 3,000     | 1.3      | 官方示例值，勉强够                   |
   | **8,000** | **3.6**  | ★ 推荐                        |
   | 15,000    | 6.7      | 上限                          |
   | 50,000    | **22.3** | ⚠️ 反复背同一批 40 条示教 22 遍，必然过拟合 |
   **`--steps=50000` 是从 ACT 那边（51.67 M 参数）沿袭过来的习惯，对 π0/π0.5 这种 3.3 B 模型完全没有必要** —— 你上一轮 5000 步时 loss 已经降到 **0.031**，说明模型早就把 40 条示教背下来了。
2. **用冻结策略。** `train_expert_only=true` 本质是一种强正则 —— 只学动作头，不动 VLM 的世界知识。这在数据少时反而比全参微调更稳（而且 §3.2 已经证明 24 GB 卡也只能这么跑）。
3. **按 checkpoint 实测。** `--save_freq=2000`，在 2K / 4K / 6K / 8K 几个 checkpoint 上各跑 10~20 次真机，**看成功率而不是看 loss**。loss 在 2000 步之后基本就不动了，之后的步数买不到性能。
4. **长期看数据量。** 数据越补越多，π0/π0.5 相对 ACT 的优势才越明显；40 条数据它的优势可能还体现不出来。

**风险二：磁盘爆掉。**

`save_freq` 默认是 20000，但 checkpoint **只会越存越多，不会自动清理**（源码里没有 retention 机制，`configs/train.py:151-154` 只有 `save_checkpoint` 和 `save_freq`）。

算账：

```
checkpoint 大小 = 3.3 B × 2 B(bf16) ≈ 7 GB / 个
5000  步 + save_freq=1000  → 5 个 checkpoint = 35 GB  ❌ 爆 30 GB 免费额度
5000  步 + save_freq=2500  → 3 个 checkpoint = 21 GB  ⚠️ 勉强
50000 步 + save_freq=20000（默认）→ 3 个（20000/40000/50000）= 20 GB  ⚠️ 再叠加已有的 pi05_A(7 GB) + HF 缓存(7 GB) 必爆
50000 步 + save_freq=10000 → 6 个 = 42 GB  ❌
8000  步 + save_freq=2000  → 4 个 = 28 GB  ⚠️ 建议只留 2 个（见下）
```

> **`--steps=50000` 有个反直觉的副作用**：`save_freq` 默认 20000，看起来「省盘」，实际会在 20000 / 40000 / 50000 各存一个 ≈7 GB 的 checkpoint（最后一步**总会**存），**20 GB 起步**。你上一轮 5000 步只落了 1 个，正是因为 5000 < 20000 且最后一步强制保存。

**控制办法：**

```bash
# 训练中/训练后随时看占用
du -sh /home/featurize/work/hf /home/featurize/work/lerobot/outputs /home/featurize/work/datasets
df -h /home/featurize/work

# 只留最后一个 checkpoint，删掉中间的（★ 删前确认最后一个完整）
ls /home/featurize/work/lerobot/outputs/train/pi05_datass/checkpoints/
rm -rf /home/featurize/work/lerobot/outputs/train/pi05_datass/checkpoints/002500
```

> 也可以训练时直接 `--save_freq=5000`（等于只在最后存一次），但那样中途挂了就没有可用 checkpoint。**推荐 `2500` + 及时清理中间产物**。

### 3.8 训练日志怎么看

关键字段：

| 字段                  | 含义                          | 健康信号                                                      |
| ------------------- | --------------------------- | --------------------------------------------------------- |
| `loss`              | 流匹配 MSE 损失（预测的速度场 vs 真实速度场） | **持续下降**，前几百步下降最快                                         |
| `grad_norm`         | 梯度范数                        | 前期较大、后期收敛变小。被 `grad_clip_norm=1.0` 截断是正常的                 |
| `lr`                | 当前学习率                       | 从 warmup 爬到 2.5e-5 后余弦衰减                                  |
| `data_s` / `updt_s` | 单步数据加载 / 更新耗时               | 与 ACT 不同，VLA 的 `updt_s` 会**远大于** `data_s`（模型大 64 倍），这是正常的 |
| `Mem`               | 显存占用                        | 应稳定在 20 GB 以下并留有余量                                        |

**读日志的一个重要提醒**：π₀.₅ 的 `loss` 数值和 ACT **不可直接比较**。ACT 的 loss 是 `L1 + 10×KL`，π₀.₅ 是流匹配的 MSE —— 量纲和量级都不同，只能各自看趋势。

### 3.9 训练结束后

```bash
# 确认产物
ls -lh /home/featurize/work/lerobot/outputs/train/pi05_datass/checkpoints/005000/pretrained_model/

# 期望看到 7 个文件（缺一不可，尤其 policy_preprocessor*.safetensors 里的归一化统计量）
```

**必须整个 `pretrained_model` 目录一起用**，不能只拷 `model.safetensors` + `config.json`。缺归一化统计量会导致**动作完全错误、机械臂乱飞**。

> 接下来走第 4 章的岔路口：**要做远程推理的话，什么都不要动**（权重留在原地就行）；**要做本地推理的话，才需要下载**。

### 3.10 把「别人的训练命令」改写成自己的（逐行对照）

网上抄来的 π₀.₅ 训练命令几乎都会踩同一批坑。下面用一份真实的社区命令做样例，逐行改写。

**① 别人的命令（藏了 6 个坑）**

```bash
lerobot-train \
--dataset.repo_id=TommyZihao/lerobot_zihao_dataset_shake_hands \
--dataset.root=~/lerobot_zihao_dataset_shake_hands\
--dataset.revision=v0.1.0 \
--policy.type=pi05 \
--output_dir=~/output_lerobot_train/shake/pi05_A \
--job_name=shake_pi05_A \
--policy.pretrained_path=lerobot/pi05_base \
--policy.compile_model=true \
--policy.gradient_checkpointing=true \
--policy.dtype=bfloat16 \
--policy.freeze_vision_encoder=false \
--policy.train_expert_only=false \
--steps=50000 \
--policy.device=cuda \
--policy.push_to_hub=false \
--wandb.enable=true \
--wandb.project=Lerobot_Zihao_Project \
--batch_size=8
```

**② 改写成你能用的命令**

```bash
export HF_HOME=/home/featurize/work/hf
export HF_ENDPOINT=https://hf-mirror.com

lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.path=lerobot/pi05_base \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_datass \
  --job_name=pi05_datass \
  --batch_size=8 \
  --steps=3000 \
  --save_freq=1500 \
  --log_freq=50 \
  --num_workers=8 \
  --wandb.enable=true \
  --wandb.project=Lerobot_Zihao_Project
```


**③ 逐行对照 + 改动理由**

| 别人的写法                                         | 改成                                                    | 为什么                                                                                                                                                                                                   |
| --------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset.repo_id=TommyZihao/...`            | `--dataset.repo_id=local/datass`                      | 那是别人的 Hub 数据集 id。本地数据无所谓，统一用 `local/<名字>` 这个惯例即可（真正决定读哪份数据的是 `root`）                                                                                                                                  |
| `--dataset.root=~/lerobot_...\`               | `--dataset.root=/home/featurize/work/datasets/datass` | ⚠️ **两个坑叠在一起**：① 反斜杠在**行尾**是 bash 的续行符，会把下一行的 `--dataset.revision=...` 并进 root 的值里；② `~` 在 `--opt=~/x` 这种**参数**里**不会被 bash 展开**（实测：Python 收到的是字面量 `--dataset.root=~/xyz`），会读到字面名叫 `~` 的目录。**必须写绝对路径** |
| `--dataset.revision=v0.1.0`                   | **删掉**                                                | `revision` 是去 Hub 上取指定版本用的；本地数据集没有这个概念                                                                                                                                                                |
| `--policy.type=pi05`                          | **删掉**                                                | 与下一行的 `--policy.path` **互斥**（`configs/parser.py:311-315` 会直接报 `Cannot specify both --policy.path and --policy.type`）。走 `--policy.path` 时类型由 checkpoint 的 `config.json` 自带                             |
| `--policy.pretrained_path=lerobot/pi05_base`  | `--policy.path=lerobot/pi05_base`                     | 语义不同：`path` 加载**权重 + 配置**（相机键名、`n_action_steps` 都继承预训练值），`pretrained_path` **只加载权重**。想让 3 路相机槽位完整复现，用 `path`（若坚持用 `pretrained_path`，见本页第一段的变体说明）                                                      |
| （没有 rename_map）                               | **新增** `--rename_map='{...}'`                         | 你的数据集键是 `observation.images.front` / `wrist`，pi05_base 期望 `base_0_rgb` / `left_wrist_0_rgb`。顺序正好对得上（第一路是正视图、第二路是腕部）                                                                                   |
| `--policy.compile_model=true`                 | **删掉**（默认 `false`）                                    | `max-autotune` 在 3.3 B 模型上编译期就可能吃满 24 G 显存；几千步的短训练收益不成正比                                                                                                                                              |
| `--policy.freeze_vision_encoder=false`        | **删掉**                                                | `train_expert_only=true` 已经把整个 VLM（含 SigLIP）冻住了，这一行是冗余的                                                                                                                                               |
| `--policy.train_expert_only=false`            | **`=true`**                                           | ⚠️ **4090 24 G 上的硬约束**。官方原文：*"A full fine-tune does not fit on a 24 GB card such as an RTX 4090."* 可训练参数从 100% 降到 ~13%（≈430 M）                                                                        |
| `--steps=50000`                               | `--steps=3000`                                        | ⚠️ 40 条 / 17,961 帧、batch 8 → **1 个 epoch ≈ 2,245 步**。50000 步 ≈ **22 个 epoch**，对 3.3 B 模型必然过拟合。3000 步 ≈ 1.3 个 epoch                                                                                    |
| `--policy.dtype=bfloat16`                     | 保留                                                    | 4090 支持 bf16                                                                                                                                                                                          |
| `--policy.device=cuda`                        | 保留                                                    | 云端实例就是 CUDA                                                                                                                                                                                           |
| `--policy.push_to_hub=false`                  | 保留                                                    | 显式关掉，**别传到公开 Hub**                                                                                                                                                                                    |
| `--wandb.enable=true` + `--wandb.project=...` | 保留                                                    | 前提是先在实例里 `wandb login`（见 §1.9）                                                                                                                                                                        |
| `--batch_size=8`                              | 保留（可升到 16 试）                                          | 只训 expert + 梯度检查点 + bf16，8 在 24 G 上很宽松                                                                                                                                                                |
| `--output_dir=~/output_.../pi05_A`            | `--output_dir=outputs/train/pi05_datass`              | 同上，`~` 不展开；用相对路径最稳（相对当前工作目录）                                                                                                                                                                          |
| （没有）                                          | **新增** `--save_freq=1500`                             | 默认 `save_freq=20000`，而 `--steps=3000` → **中途一个 checkpoint 都不会存**，一旦掉线全白跑                                                                                                                              |
| （没有）                                          | **新增** `--log_freq=50`                                | 默认 200，日志太稀，看不清 loss 趋势                                                                                                                                                                               |
| （没有）                                          | **新增** `--num_workers=8`                              | 默认 4。4090 实例核多，可提速数据加载；内存紧张时降到 4                                                                                                                                                                      |

> **`--policy.pretrained_path` 变体**：如果你想保留原命令的写法（`--policy.type=pi05` + `--policy.pretrained_path=...`），把上面 `--policy.path=lerobot/pi05_base` 这一行换成原来两行、并在 `--rename_map` 后面补一行 `--policy.empty_cameras=1` 即可（补回第 3 个相机槽位）。两种写法都能跑，区别只有 §3.1 里说的「相机槽位 3 vs 2」。

### 3.11 π0 与 π0.5 的差别，以及官方 `pi0.mdx` 命令为什么不能照抄

如果你把策略从 `pi05` 换成 `pi0`（`--policy.type=pi0` + `--policy.pretrained_path=lerobot/pi0_base`），**上面 §3.4 的结论一条都不用改** —— 两者在本项目关心的一切维度上几乎同构：

| 维度                          | π0（`configuration_pi0.py`）  | π0.5（`configuration_pi05.py`） | 影响                               |
| --------------------------- | --------------------------- | ----------------------------- | -------------------------------- |
| 主干                          | `gemma_2b` + `gemma_300m`   | 同左                            | **参数总量都 ≈3.3 B**，显存/速度同量级        |
| `chunk_size` / `max_*_dim`  | 50 / 32                     | 50 / 32                       | 同                                |
| `image_resolution`          | 224×224                     | 224×224                       | 同                                |
| **`tokenizer_max_length`**  | **48**（`:103`）              | **200**（`pi05:87`）            | π0 的 prompt 短 152 token，**每步略快** |
| **`normalization_mapping`** | STATE/ACTION = **MEAN_STD** | STATE/ACTION = **QUANTILES**  | 归一化口径不同，靠 checkpoint 自带的统计量自动对齐  |
| `train_expert_only` 可训练比例   | ~13%                        | ~13%                          | 同                                |
| `empty_cameras` 默认          | **0**（`:68`）                | 0                             | **两路相机数据集都要手动补第 3 路**            |

**⚠️ `docs/source/pi0.mdx:58-73` 那段示例命令不能直接照抄。** 它是：

```bash
--policy.compile_model=true  --policy.gradient_checkpointing=true  --policy.dtype=bfloat16
--policy.freeze_vision_encoder=false  --policy.train_expert_only=false
--steps=3000  --batch_size=32
```

文档自己在下面第 80 行写了 *"`--batch_size=32`: Batch size for training, **adapt this based on your GPU memory**"* —— 也就是说**那套超参不是为 24 GB 卡写的**。`train_expert_only=false` + `batch_size=32` 是 80 GB 级卡上的配方。

#### 为什么 `train_expert_only=false` 在 4090 24G 上一定 OOM（把账算给你看）

π0/π0.5 总参数 ≈ **3.3 B**。`--policy.dtype=bfloat16` 的实际行为是（`modeling_pi0.py:247-268`）：先整体 `.to(bfloat16)`，再把 `vision_tower` / `multi_modal_projector` / 所有 `layernorm` / `model.norm` **转回 float32**。也就是 ≈0.47 B 走 fp32、≈2.83 B 走 bf16。

`torch.optim.AdamW` 的动量 dtype **跟随参数**（`optim/optimizers.py:226-231`，没有 fp32 master weight、没有 8-bit），所以全参微调的常驻显存是：

```
参数       2.83B×2B + 0.47B×4B  ≈ 7.5 GB
梯度       与参数同 dtype 同形状  ≈ 7.5 GB
exp_avg    与参数同 dtype 同形状  ≈ 7.5 GB   ← AdamW 的第一个动量
exp_avg_sq 与参数同 dtype 同形状  ≈ 7.5 GB   ← AdamW 的第二个动量
─────────────────────────────────────────
合计                            ≈ 30 GB
```

**30 GB > 24 GB**，还没算激活、CUDA context、以及 `compile_model=true` 时 `max-autotune` 的工作区。所以不是「慢一点」的问题，是**第一步就爆**。

| 配置                           | 可训练参数       | 常驻显存估算    | 4090 24G |
| ---------------------------- | ----------- | --------- | -------- |
| `train_expert_only=false`    | 3.3 B（100%） | ≈30 GB    | ❌ OOM    |
| `freeze_vision_encoder=true` | ≈2.9 B      | ≈26 GB    | ❌ OOM    |
| **`train_expert_only=true`** | **≈0.43 B** | **≈4 GB** | ✅ **推荐** |

（这张表是 §3.2 那句中译英的量化版本；`train_expert_only` 的具体行为见 `modeling_pi0.py:270-279` —— `self.paligemma.eval()` + 全部 `requires_grad=False`。）

#### 改好之后能用的 π0 命令

```bash
conda activate pi05                        # 或删掉这行走镜像自带环境
cd /home/featurize/work/lerobot
export HF_HOME=/home/featurize/work/hf

lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.path=lerobot/pi0_base \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.empty_cameras=1 \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.compile_model=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi0_datass \
  --job_name=pi0_datass \
  --steps=8000 \
  --save_freq=2000 \
  --log_freq=50 \
  --batch_size=8 \
  --num_workers=8 \
  --wandb.enable=true \
  --wandb.project=pi0
```

相对你原来的命令，改了 6 处：

| 原命令                                                  | 改成                                                                   | 为什么                                                                                              |
| ---------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `--policy.type=pi0` + `--policy.pretrained_path=...` | `--policy.path=lerobot/pi0_base`                                     | §3.1：`path` 连 config 一起加载，相机槽位以 checkpoint 为准（否则要手动补 `empty_cameras`）                            |
| `--policy.train_expert_only=false`                   | **`true`**                                                           | 上面那笔账，**不改就跑不起来**                                                                                |
| `--policy.freeze_vision_encoder=false`               | 删掉（`train_expert_only=true` 已覆盖）                                     | 冗余；留着反而容易看糊涂                                                                                     |
| （无）                                                  | **`--rename_map` + `--policy.empty_cameras=1`**                      | 你的数据集是 `front`/`wrist` 两路，π0_base 期望 3 路 `*_rgb`。**不加必然报特征缺失**                                   |
| `--steps=50000`                                      | **`8000`**                                                           | 见 §3.7：50000 步 = 22 个 epoch，40 条数据必过拟合                                                           |
| `--output_dir=~/output_.../pi0_A`                    | `--output_dir=outputs/train/pi0_datass`                              | `~` 在参数中间不展开（§3.1 实测）；别再造出名叫 `~` 的目录                                                             |
| （无）                                                  | **`--save_freq=2000`** / **`--log_freq=50`** / **`--num_workers=8`** | 默认 20000/200/4：50000 步会闷头存 3 个 6.6 GB 的 checkpoint，日志也稀得看不见趋势                                    |
| `--dataset.root=/home/featurize/datasets/datass`     | **先确认这个路径真的存在**                                                      | 这个路径**不在**持久盘 `/home/featurize/work` 下。Featurize 数据集的标准挂载点是 `/featurize/datasets/`，先 `ls` 一眼（见下） |

**先跑这两行确认数据在不在：**

```bash
ls -d /home/featurize/work/datasets/datass /featurize/datasets/datass /home/featurize/datasets/datass 2>/dev/null
```

哪个存在就用哪个。三个都不存在的话，说明你之前是放在别处的 —— 用 `find /home/featurize -maxdepth 4 -name "datass" -type d 2>/dev/null` 找出来。

---

## 4. 岔路口：权重取回还是不取回？

训练完成后，根据第 5 / 6 章选定的推理方案决定：

> **不管选哪个方案，都建议先看 §4.1「把权重托管到 Hugging Face」** —— 实例释放后权重就没了，托管一次，以后任何机器一条 `--policy.path=<repo_id>` 就能加载。

| 你的推理方案                | 要不要下载权重                      |
| --------------------- | ---------------------------- |
| **方案 A：远程推理**（推荐）     | **不用下**。权重留在实例上，第 5 章直接原地起服务 |
| **方案 B1：本地跑 ACT**     | 不用（ACT 权重早就在本机了）             |
| **方案 B2：π0.5 CPU 推理** | **要下**，约 7 GB                |

**取回权重的办法（方案 B2 才需要）：**

| 方式              | 操作                                                                                                                                                                                   | 说明                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| VSCode 远程下载     | 连上实例，右键文件 → Download                                                                                                                                                                 | 7 GB 比较慢，但最省事                           |
| JupyterLab 右键下载 | 文件 → Download                                                                                                                                                                        | JupyterLab **只对文件提供 Download**，不支持整个文件夹 |
| `scp`（推荐大文件）    | 本机 PowerShell：`scp -P <SSH端口> -r featurize@workspace.featurize.cn:/home/featurize/work/lerobot/outputs/train/pi05_datass/checkpoints/005000/pretrained_model D:\Desktop\pi05_datass` | 支持断点续传（`-C` 压缩、`rsync` 更好）              |
| 打包 tar          | `tar -czf pi05.tar.gz pretrained_model/` 再下载                                                                                                                                         | ⚠️ 权重是稠密浮点矩阵，**压缩率极低**，纯浪费时间            |

> 30 GB 同步盘塞一个 7 GB 权重是可行的，但注意别和数据集、其他 checkpoint 挤在一起。
>
> **顺带说一句**：`--policy.push_to_hub` 在源码里的**默认值是 `True`**（`src/lerobot/configs/policies.py:70`）。你第 3 章的命令里写了 `--policy.push_to_hub=false`，所以训练结束时没有自动上传 —— 也就是说**这一步必须手动补做**。


### 4.1 把权重托管到 Hugging Face（**推荐，不管选 A 还是 B 都建议做**）

**为什么值得做**：Featurize 实例上只有 `/home/featurize/work` 会持久化，而它只有 30 GB 免费额度（超了 ¥0.001/GB/小时）。一个 π₀.₅ 微调权重就 6.6 GB，再攒几个 checkpoint 额度就爆了。托管到 Hub 之后：① 实例可以随时释放；② 任何机器上一条 `--policy.path=<你的用户名>/<仓库名>` 就能加载；③ 想分享给别人只需给一个链接。

#### ⚠️ 先说三个坑，不然一定失败

| 坑                                                       | 真相                                                                                                       |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 以为能用 `hf-mirror.com` 加速上传                             | **镜像站是只读的**。它只做公开仓库的下载同步，**不支持上传、不支持创建仓库、不支持私有仓库**。上传必须**直连 `huggingface.co`**。                        |
| 复用 §1.6 那个 token                                      | 那次拉门控分词器用的是 **read** 权限 token，**上传不够**。必须新建一个 **write** 权限的 token。                                         |
| 传完能用就行，不管目录结构                                         | `pretrained_model/` 里的文件**必须平铺在仓库根目录**。一旦被套进 `pretrained_model/` 子目录，`--policy.path` 就找不到 `config.json`，报 `[Errno 2]`——就是你在上一轮踩过的那个错。 |

#### 第 0 步：连通性自测（30 秒，决定走哪条路）

```bash
unset HF_ENDPOINT    # ★ 关键：§1.7 你 export 过这个变量，不清掉的话上传会 POST 到只读镜像站
curl -sS -o /dev/null -w 'huggingface.co -> %{http_code}  %{time_total}s\n' --max-time 10 https://huggingface.co
curl -sS -o /dev/null -w 'hf-mirror.com  -> %{http_code}  %{time_total}s\n' --max-time 10 https://hf-mirror.com
```

| 输出                                   | 含义           | 怎么办                        |
| ------------------------------------ | ------------ | -------------------------- |
| `huggingface.co -> 200  x.xs`        | 直连可用，速度看耗时   | 继续第 1 步                     |
| `huggingface.co -> 000`（或卡住超时）       | 直连不通         | 跳到本文最后的「备选路线」              |
| 只有 `hf-mirror.com -> 200`            | 只能下载，不能上传    | 同上                         |

> `%{time_total}` 是握手往返耗时。`< 3s` 基本能用；`> 8s` 说明通了但很慢，6.6 GB 会传到天荒地老，也建议走备选路线。

#### 第 1 步：确认 CLI 可用

LeRobot 依赖 `huggingface-hub>=1.6.0,<2.0.0`（`pyproject.toml:71`），所以环境里已经有了，一般不用重装：

```bash
pip install -U "huggingface_hub[cli]"
hf version
```

> **命名变更提醒**：v1.x 起原来的 `huggingface-cli` 已更名为 **`hf`**。如果你的环境里敲 `hf` 提示 command not found，说明版本旧，把下文所有 `hf` 换成 `huggingface-cli` 即可（参数完全一样）。

#### 第 2 步：建一个 write token 并登录

1. 浏览器打开 <https://huggingface.co/settings/tokens> → **New token** → **Type 选 `Write`** → 复制那串 `hf_...`
2. 回到实例：

```bash
hf auth login          # 粘贴 write token
hf auth whoami         # 验证：应打印你的用户名
```

> 上一次（§1.6）那个 read token 留着别删，拉门控分词器还要用。两个 token 互不影响。

#### 第 3 步：把 checkpoint 挪到干净路径（**必做**）

你现在的路径是：

```
/home/featurize/work/lerobot/~/output_lerobot_train/shake/pi05_A/checkpoints/005000/pretrained_model
```

中间那个 `~` **是字面量的目录名，不是家目录**（shell 只在词首展开 `~`）。后面每条命令都得额外套引号，极容易抄错。先搬走：

```bash
SRC='/home/featurize/work/lerobot/~/output_lerobot_train/shake/pi05_A'
DST='/home/featurize/work/ckpt_pi05_A'
mv "$SRC" "$DST"

CKPT="$DST/checkpoints/005000/pretrained_model"
ls -lh "$CKPT"
```

> **`last` 软链会不会断？不会。** 它是**相对**软链（`train_utils.py:128-130`：`checkpoint_dir.relative_to(checkpoint_dir.parent)`），整个 run 目录一起挪，链接依然有效。

`ls -lh "$CKPT"` 应该正好看到 **7 个文件**：

| 文件                                            | 大小         | 作用                                                    | 漏了会怎样                                        |
| --------------------------------------------- | ---------- | ----------------------------------------------------- | -------------------------------------------- |
| `config.json`                                 | 几 KB       | 策略配置（`PI05Config`）：相机键名、`chunk_size`、归一化模式、`compile_model` | `--policy.path` 直接报「找不到 config.json」         |
| `model.safetensors`                           | **≈6.6 GB** | 权重本体。**单文件不分片**（`pretrained.py:54` `_SINGLE_FILE_SHARD_SIZE = "1TB"`） | 空壳，加载出来是随机权重                                  |
| `train_config.json`                           | 几十 KB      | 训练配置。`--config_path=<它> --resume=true` 续训用             | 不能续训（推理不受影响）                                 |
| `policy_preprocessor.json`                    | 几 KB       | 预处理流水线定义                                              | 输入不做 resize / 归一化                            |
| `policy_preprocessor_step_*.safetensors`      | 几 KB ~ MB  | ★ **归一化统计量**（mean / std）落地于此                          | **动作尺度全错，机械臂乱动/不动** —— 最容易被忽略的一个 |
| `policy_postprocessor.json`                   | 几 KB       | 后处理流水线定义                                              | 动作不做反归一化                                     |
| `policy_postprocessor_step_*.safetensors`     | 几 KB ~ MB  | 后处理的内部状态                                              | 同上                                           |

> ⭐ 一句话记住：**这 7 个文件是一个整体，一个都不能漏**。要分享给别人的也是这一整个目录，不是那个 6.6 GB 的大文件。
>
> 这套清单来自官方源码注释 `src/lerobot/common/train_utils.py:207-218`（`save_checkpoint` 的 docstring 画了完整目录树）。

#### 第 4 步：上传（就一条命令）

```bash
HF_USER=TommyZihao              # ← 换成你自己的 HF 用户名（就是 hf auth whoami 打印的那个）
REPO=$HF_USER/pi05_shake_A      # ← 仓库名随便取，但建议带策略名，见下面「命名建议」

hf upload "$REPO" "$CKPT" --repo-type model --private
```

逐段解释：

| 片段                   | 含义                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `hf upload`          | v1.x CLI 的上传子命令，内部等价于 `HfApi().upload_folder()`                                            |
| `"$REPO"`            | 目标仓库 `<用户名>/<仓库名>`。**不存在会自动创建**，不需要先去网页建仓库                                                 |
| `"$CKPT"`            | 本地要传的目录。**故意不写第三个参数** —— 这样目录里的文件会**平铺到仓库根目录**，正是 LeRobot 期望的结构                        |
| `--repo-type model`  | 建的是模型仓库。这本来就是默认值，写出来是为了跟数据集上传区分开                                                           |
| `--private`          | 仓库不存在时建成**私有**；**仓库已存在则此参数被忽略**（想去网页 Settings → Change visibility 改）                          |

> ⚠️ **千万不要**写成 `hf upload "$REPO" "$CKPT" pretrained_model`。那会在仓库里多套一层 `pretrained_model/` 子目录，`--policy.path=$REPO` 就加载不到 `config.json` —— 又会报你之前那个 `[Errno 2]`。官方给 LeRobot 的上传示例也是不带第三个参数的（`docs/source/il_robots.mdx:606`）：
>
> ```bash
> hf upload ${HF_USER}/act_so101_test \
>   outputs/train/act_so101_test/checkpoints/last/pretrained_model
> ```
>
> **想要公开仓库**：去掉 `--private` 就是公开（公开有好处——别人能直接拿来当 baseline，而且 `hf-mirror.com` 能加速下载你的模型）。

**命名建议**：`<策略>_<任务>_<数据集版本>`，例如 `pi05_shake_A`、`pi05_shake_40eps`。同一个 run 里多个 checkpoint 想都传上去，用后缀区分：

```bash
CKPT5k='/home/featurize/work/ckpt_pi05_A/checkpoints/005000/pretrained_model'
hf upload "$HF_USER/pi05_shake_A/005000" "$CKPT5k" --repo-type model --private   # 仓库名里带步数，各占一个仓库
```

**上传耗时**：6.6 GB。

| 直连速度     | 耗时     |
| -------- | ------ |
| 5 MB/s   | ≈ 22 分钟 |
| 20 MB/s  | ≈ 6 分钟  |
| 1 MB/s   | ≈ 110 分钟 |

- 进度条看到 `Uploading ░░░ 1/7 files 6.6GB · x MB/s` 就对了，`Found 7 files to upload` 也和上面那张表对得上。
- **中断了直接重跑同一条命令**。v1.x 走 Xet 流式管道，已提交的文件、已上传的分块会自动跳过（官方原话：*"if the upload is interrupted for any reason, simply re-run the same call"*），不需要 `--resume` 之类的参数。
- 反复失败时换这个，它专为大目录/弱网设计，支持多线程和断点续传：

```bash
hf upload-large-folder "$REPO" "$CKPT" --repo-type model --private --num-workers 8
```

#### 第 5 步：验证（**别跳过**）

最直观的：浏览器打开 `https://huggingface.co/<REPO>/tree/main`，确认 7 个文件**平铺在根目录**、没有多套一层文件夹。

命令行验证：

```bash
hf download "$REPO" config.json --local-dir /tmp/hf_verify
head -c 400 /tmp/hf_verify/config.json; echo
```

- 看到 `"type": "pi05"` → 结构正确 ✅
- 报 404 / 找不到 → 说明被套进了子目录。重新传（`--delete="*"` 会先清空远端再提交）：

```bash
hf upload "$REPO" "$CKPT" . --repo-type model --delete="*"
```

#### 第 6 步：上传之后怎么用

| 场景                   | 参数写法                                                                              |
| -------------------- | --------------------------------------------------------------------------------- |
| 本机 `lerobot-rollout` | `--policy.path=$HF_USER/pi05_shake_A`                                              |
| 云端远程推理服务端            | 把 `--pretrained_name_or_path=<本地路径>` 换成 `--pretrained_name_or_path=$HF_USER/pi05_shake_A` |
| 钉住某个版本               | 追加 `--policy.pretrained_revision=<commit sha>`（仓库里传了多个 checkpoint 时用）              |
| 续训                   | `lerobot-train --config_path=$HF_USER/pi05_shake_A/train_config.json --resume=true` |

**为什么 repo id 能直接当本地路径用**：服务端 `policy_server.py:152` 是 `policy_class.from_pretrained(policy_specs.pretrained_name_or_path)`，而 `from_pretrained` 本身就走 HF Hub 解析逻辑 —— 本地目录和 repo id 是**同一套代码路径**（`utils/hub.py` 的 `from_pretrained` 先试本地、再走 Hub）。所以填 repo id 不需要任何额外参数，它会自己下载并缓存。

> **首次用 repo id 加载会先下载 6.6 GB**。想省这一步，加 `export HF_HOME=/home/featurize/work/hf`（同 §1.7 的理由），或者干脆把 Hub 缓存当成本地副本用。

#### 备选路线（第 0 步测出直连不通时）

| 路线                             | 命令                                                                                                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. 换 ModelScope 魔搭**（国内直连，无需代理） | `pip install modelscope` 然后 `modelscope upload <你的用户名>/pi05_shake_A "$CKPT" --token <魔搭 token>`（token 在 <https://modelscope.cn/my/myaccesstoken> 领）。⚠️ 魔搭默认建**公开**仓库，要私有得上传后去网页改 |
| **B. 先 scp 回本机，本机挂代理再传**        | 见 §4 上面那张表里的 `scp` 命令；本机 `set HF_ENDPOINT=` 清空后重跑第 4 步                                                                                                         |
| **C. 走 Featurize 数据集**（不推荐）     | 官方文档明确说 dataset 功能「仅能在 Featurize 中的实例中添加使用」，**不提供对外下载**，拿不到跨平台链接                                                                                                |

> 路线 A 的代价：以后 `--policy.path` 得配 ModelScope 的加载方式，不像 HF repo id 那样开箱即用。如果只是**备份**权重、不打算到处加载，A 完全够用且最省心。

---

## 5. 推理方案 A：远程推理（云端 GPU + 本机机械臂）★ 推荐

### 5.1 为什么这是你的主方案

|                 | 数值                        | 结论      |
| --------------- | ------------------------- | ------- |
| 本机显存            | **4 GB**（RTX 3050 Laptop） |         |
| π₀.₅ bf16 权重    | ≈ 6.6 GB                  | **装不下** |
| π₀.₅ fp32 权重    | ≈ 13.4 GB                 | 更不可能    |
| 官方对 PI0 推理显存的说明 | 约 14 GB                   | 印证      |

**但π₀.₅ 的推理速度其实很快** —— 慢的是显存门槛，不是算力：

| 测量口径                                                             | RTX 4090 上的耗时                        |
| ---------------------------------------------------------------- | ------------------------------------ |
| Vanilla π₀.₅ 单次推理（bf16，5 步去噪，RTC 论文 Table 1）                     | **76 ms**                            |
| RTC 引导版（同上）                                                      | 97 ms                                |
| 组件分解（4090 + `torch.compile`）：VLM 前向 37 ms + 每个 DiT 去噪步 3 ms × 10 | **≈ 67 ms**                          |
| 端到端含网络（RTC 论文 Table 2，非移动场景）                                     | 109 ms（其中网络 6.9 ms、图像 resize 1.4 ms） |

**关键推论**：chunk 50 步 @ 30 fps = **1.67 秒**的缓冲，而推理只要 **0.08~0.15 秒**。所以：

- **GPU 不是瓶颈，网络带宽才是瓶颈**（见 §5.9）；
- 网络延迟（家宽到云端 20~50 ms）相对 1.67 秒的 chunk 完全可忽略；
- 远程推理在**时序上是安全的**，不会因为网络抖动把机械臂卡住。

### 5.2 架构

```
┌─────────────────────────────┐         ┌──────────────────────────────────┐
│  Featurize RTX 4090 实例     │         │  本机 Windows                     │
│  （Ubuntu）                  │         │                                  │
│                             │  gRPC   │                                  │
│  PolicyServer               │◄───────►│  RobotClient                     │
│   · 3.3B π₀.₅ 推理            │ 隧道/公网│   · 读 2 路 USB 相机               │
│   · cuda, bf16              │         │   · 发串口指令到 SO-101            │
│   · 加载微调后的 checkpoint    │         │   · 维护动作队列、RTC 聚合          │
│   · 监听 :8080               │         │   · 不跑模型，不需要 GPU            │
└─────────────────────────────┘         └──────────────────────────────────┘
```

**两端各自要装什么：**

| 端          | 需要                                                     | 不需要            |
| ---------- | ------------------------------------------------------ | -------------- |
| 云端实例       | `pip install -e ".[pi,async]"`（已在 §1.5 装好）+ checkpoint | 相机、机械臂         |
| 本机 Windows | lerobot + `[async]`（grpcio）、USB 相机、机械臂、标定文件            | **CUDA / GPU** |

> **本机不需要 GPU！** client 端只做「读相机 → 序列化 → 发 gRPC → 收动作 → 发串口」。你之前装的那套 CUDA 版 torch 可以留着，但**纯粹为了跑 client 的话，CPU 版 torch 就够了**。

> ⚠️ **重要限制**：LeRobot 的 `lerobot-rollout` 命令**只支持 `--inference.type=sync|rtc`**（`rollout/inference/factory.py:17,60,66`），**不支持 remote/async**。所以远程推理**必须**用本章的 `async_inference` 独立 server + client，**不是** rollout 加个参数。

### 5.3 云端：启动 Policy Server

在 Featurize 实例上（SSH / JupyterLab 终端）：

```bash
conda activate pi05                 # ← 走路线 A 复用镜像环境的话，这行删掉
cd /home/featurize/work/lerobot

python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=2
```

**建议用 tmux/screen 跑，别用前台：**

```bash
tmux new -s server
# 在 tmux 里执行上面的命令，然后 Ctrl-B D 退出（进程继续跑）
# 回到会话： tmux attach -t server
```

看到这行就说明起来了：

```
PolicyServer started on 127.0.0.1:8080
```

> **注意：server 启动时并不知道要跑哪个模型。** 策略类型、权重路径、设备都是**第一次握手时由 client 传过来**的（`policy_server.py:116-165`）。所以**权重路径要写在 client 命令里**（§5.6）。
>
> **绑定 `127.0.0.1` 还是 `0.0.0.0`？**
>
> - 走 **SSH 隧道**（§5.4 方式①）→ 绑 `127.0.0.1` 就够，**更安全**；
> - 走 **公网暴露**（§5.4 方式②）→ 必须绑 `0.0.0.0`，但这意味着**任何人都能连上并控制你的机械臂**。

**PolicyServer 关键参数：**

| 参数                    | 默认                   | 含义                                      |
| --------------------- | -------------------- | --------------------------------------- |
| `--host` / `--port`   | `localhost` / `8080` | 监听地址。gRPC `add_insecure_port`，**明文无鉴权** |
| `--fps`               | 30                   | 服务端按这个频率产出动作（内部有 FPS 跟踪）                |
| `--inference_latency` | 1/30 ≈ 0.033         | 目标推理延迟（秒）。算得比它快就 sleep 补齐，用于**限速**      |
| `--obs_queue_timeout` | 2                    | 观测队列等待超时（秒）                             |

### 5.4 连通：两种方式，**强烈推荐方式①**

#### 方式① SSH 本地端口转发（推荐）

Featurize 的所有实例**没有独立公网 IP**，但**支持 SSH 登录**（端口、用户名 `featurize`、密码都在工作区页面）。

在**本机 Windows** 开一个 PowerShell/CMD 窗口：

```cmd
ssh -p <你的SSH端口> -L 8080:127.0.0.1:8080 featurize@workspace.featurize.cn
```

- `<你的SSH端口>`：工作区页面显示的 SSH 端口（形如 `58727`）
- `-L 8080:127.0.0.1:8080`：把**本机** `127.0.0.1:8080` 转发到**实例上的** `127.0.0.1:8080`
- 输入实例密码后，**这个窗口要保持开着**（它就是隧道）

**优点**：全程加密、实例侧不需要暴露任何公网端口、server 可以绑 `127.0.0.1`。  
**代价**：SSH 加解密有少量 CPU 开销（对 gRPC 这点流量可忽略）。

**验证隧道通了**（本机另开一个窗口）：

```cmd
python -c "import socket; s=socket.create_connection(('127.0.0.1',8080),3); print('tunnel OK'); s.close()"
```

#### 方式② `featurize port export` 暴露公网（**不推荐**）

```bash
# 实例上
featurize port export 8080
# 输出：Local port 8080 has been exported to 60623
#       You can visit http://workspace.featurize.cn:60623
```

然后 client 用 `--server_address=workspace.featurize.cn:60623`。

> ⚠️ **为什么强烈不推荐**：
>
> 1. PolicyServer 是 `grpc.server(...)` + `add_insecure_port`，**没有凭证、没有拦截器、没有 TLS**（`policy_server.py:426-428`）。客户端也是 `grpc.insecure_channel`。
> 2. 这个端口一暴露，**互联网上任何扫到它的人都能给你的机械臂下达动作指令**。这是物理安全问题，不是数据安全问题。
> 3. Featurize 官方也明确提示：*"暴露的端口所有人都能够访问，可能会存在安全隐患"*、*"请不要将敏感信息暴露出去"*。
> 4. 每台实例最多暴露 10 个端口，且**必须是 HTTP 服务**才能直接在浏览器访问；gRPC 走 HTTP/2 长连接，代理兼容性也不保证。
>
> **除非你在完全隔离的内网里，否则请用方式①。**

### 5.5 ⚠️ 必做：修复服务端的图像预处理（**不改会掉点**）

这是本手册**最重要的一处代码修改**，原因如下。

**训练/`lerobot-rollout` 路径**的图像处理链：

```
480×640 原图 → policy 内部 _preprocess_images
            → 尺寸 ≠ 224 → resize_with_pad_torch()  ← 等比缩放 + 居中黑边（letterbox）
            → ×2−1 → SigLIP
```

**`async_inference` 路径**的图像处理链：

```
480×640 原图 → helpers.resize_robot_observation_image()
            → F.interpolate(size=(224,224), mode="bilinear")  ← ★ 直接拉伸，宽高比失真！
            → 交给 policy 的 _preprocess_images
            → 尺寸 == 224 → if spatial_shape != image_resolution 为假 → ★ 跳过 letterbox
            → ×2−1 → SigLIP
```

**两个问题叠加**：图像先在服务端被**拉伸变形**成 224×224，然后 policy 看到尺寸已经对了，就**跳过了本该做的 letterbox**（`modeling_pi05.py:1211` 的 `if spatial_shape != self.config.image_resolution:`）。

**结果**：推理时的输入分布与训练时**不一致**（训练是 4:3 保比例 + 黑边，推理是 4:3 被压成 1:1），**成功率会下降**。

**修复补丁**（只改**云端实例**的 `helpers.py`，因为 resize 在服务端执行）：

```bash
cd /home/featurize/work/lerobot
cp src/lerobot/async_inference/helpers.py src/lerobot/async_inference/helpers.py.bak
```

然后编辑 `src/lerobot/async_inference/helpers.py`：

**① 在文件顶部 import 区域加上：**

```python
from lerobot.policies.common.vla_utils import resize_with_pad_torch
```

**② 把原有的 `resize_robot_observation_image` 整个函数替换为：**

```python
def resize_robot_observation_image(
    image: torch.Tensor, resize_dims: tuple[int, int, int]
) -> torch.Tensor:
    """(H, W, C) -> (C, H, W)，用 letterbox（等比缩放 + 居中黑边）而非直接拉伸。

    与训练时 policy 内部 _preprocess_images 的处理保持一致：
    这样下游 _preprocess_images 看到 224x224 而跳过自己的 letterbox 才是正确的。
    """
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    image = image.permute(2, 0, 1)              # (H, W, C) -> (C, H, W)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0           # -> [0, 1] float32
    # resize_with_pad_torch 要求 float32 且取值在 [0, 1]（内部会 clamp 并 pad 0.0）
    resized = resize_with_pad_torch(image, resize_dims[1], resize_dims[2])
    return resized.squeeze(0)                   # (1, C, H, W) -> (C, H, W)
```

改完**不需要重装**（`pip install -e .` 是 editable 安装），但**要重启 PolicyServer**（建议重启整个实例的 tmux 会话）。

**怎么验证补丁生效**：改前/改后各跑一次真机，同样的起始位置、同样的任务，比较成功率。也可以先纯 CPU 单测：

```bash
python - <<'EOF'
import torch
from lerobot.async_inference.helpers import resize_robot_observation_image

img = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8)   # (H, W, C)
out = resize_robot_observation_image(img, (3, 224, 224))
print("shape:", tuple(out.shape), "| dtype:", out.dtype,
      "| range:", round(float(out.min()), 3), round(float(out.max()), 3))

# 480x640 是 4:3 横向 -> 等比缩放到 168x224 -> 上下各补 28 行黑边（左右不补）
print("前 5 行是否全黑 (应为 True):", bool((out[:, :5, :] == 0).all()))
print("后 5 行是否全黑 (应为 True):", bool((out[:, -5:, :] == 0).all()))
print("中间行是否非黑 (应为 True):", bool((out[:, 112, :] > 0).any()))
EOF
```


**判读**：`形状 (3,224,224)` + 前后各 5 行全黑 → **letterbox 生效** ✅  
（拉伸版本不会有黑边；如果你看到三个 `True` 就说明补丁打对了。）

> 补充一个容易搞混的点：480×640 是"**高 480、宽 640**"的横向图，letterbox 到 224×224 正方形后，**黑边出现在上下**（各 28 行），左右是满的。这和不少人的直觉相反。

### 5.6 本机：启动 Robot Client

先在**本机**确认相机编号没变（USB 重新插拔后编号可能变）：

```cmd
lerobot-find-cameras
```

记下前置相机和腕部相机的 `index`。

**完整命令（Windows cmd，续行符 `^`）：**

```cmd
python -m lerobot.async_inference.robot_client ^
  --server_address=127.0.0.1:8080 ^
  --robot.type=so101_follower ^
  --robot.port=COM7 ^
  --robot.id=zihao_follower_arm ^
  --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" ^
  --task="Grab Box" ^
  --policy_type=pi05 ^
  --pretrained_name_or_path=/home/featurize/work/lerobot/outputs/train/pi05_datass/checkpoints/005000/pretrained_model ^
  --policy_device=cuda ^
  --client_device=cpu ^
  --fps=30 ^
  --actions_per_chunk=50 ^
  --chunk_size_threshold=0.5 ^
  --aggregate_fn_name=weighted_average ^
  --debug_visualize_queue_size=True
```

**一行版（复制即用）：**

```cmd
python -m lerobot.async_inference.robot_client --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --policy_type=pi05 --pretrained_name_or_path=/home/featurize/work/lerobot/outputs/train/pi05_datass/checkpoints/005000/pretrained_model --policy_device=cuda --client_device=cpu --fps=30 --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --debug_visualize_queue_size=True
```

#### ⚠️ 三个必须理解的关键点

**① 相机键名必须**直接写成策略期望的名&#x5B57;**，不要用 `front` / `wrist`！**

```
--robot.cameras="{ base_0_rgb: {...}, left_wrist_0_rgb: {...} }"
```

**为什么**：`RobotClient` 构造 `RemotePolicyConfig` 时**根本没有传 `rename_map`**：

```python
# src/lerobot/async_inference/robot_client.py:116-122
self.policy_config = RemotePolicyConfig(
    config.policy_type,
    config.pretrained_name_or_path,
    lerobot_features,
    config.actions_per_chunk,
    config.policy_device,
)          # ← 没有第 6 个参数 rename_map，用默认空 dict
```

而 `RobotClientConfig` 里**也没有 `rename_map` 字段**（`configs.py:102-149`）—— 也就是说**当前版本 client 端无传 rename_map**。

于是服务端会直接用**机器人端的键名**去索引策略的图像特征：

```python
# src/lerobot/async_inference/helpers.py:165-168
image_dict = {
    key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
    for key in image_keys          # key 形如 observation.images.<你的相机名>
}
```

键名对不上的话，`policy_image_features[key]` **直接 KeyError**。

> 所以**训练时用 `--rename_map` 把 `front` → `base_0_rgb`；远程推理时反过来，直接给相机起名叫 `base_0_rgb`。** 看起来不一致，但这是同一件事的两面：**最终进模型的键名必须是 `base_0_rgb` / `left_wrist_0_rgb`**。
>
> 第 3 路 `right_wrist_0_rgb` 你不需要提供 —— 服务端 `_preprocess_images` 会自动补空（`modeling_pi05.py:1243`）。

**② `--pretrained_name_or_path` 写的是【服务端】上的路径。**

这个参数容易看反。它走的是：

```
client 发指令 → server 收到后：
  self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)   # policy_server.py:152
  make_pre_post_processors(self.policy.config, pretrained_path=...同路径...)          # policy_server.py:157
```

所以：

- ✅ `--pretrained_name_or_path=/home/featurize/work/.../005000/pretrained_model`（**云端绝对路径**）
- ❌ `--pretrained_name_or_path=D:\Desktop\pi05_datass`（那是本机路径，服务端读不到）

指向**整个 `pretrained_model` 目录**，因为 `make_pre_post_processors` 要顺带加载里面的归一化统计量。

`--policy_device=cuda` 指的也是**服务端**（哪个 GPU 跑模型）；`--client_device=cpu` 才是本机。

**③ `--policy_type=pi05` 用的是下划线，不是 `--policy.type`。**

别和 `lerobot-rollout` 的 `--policy.path` 混了。这里策略由「类型 + 路径」两个独立参数指定。

#### 其余要点

- **`--task` 必须与数据集一致**：`"Grab Box"`。π₀.₅ 把任务文本和离散化 state 一起写进 prompt（`processor_pi05.py:77-85`）：
  ```
  Task: Grab Box, State: 128 96 141 103 112 87;
  Action:
  ```
- **`--robot.id` 必须带**：标定文件是 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`，**目录名由机器人类属性 `name`（`so_follower`）决定，文件名由 `--robot.id` 决定**。漏了会去读 `None.json`，6 个关节全读成 `2047`（舵机没读通的哨兵值）。
- **`--fps` 默认 30**，与数据一致。
- **没有 `--duration` 参数！** `RobotClientConfig` 里根本没有这个字段（`configs.py:102-149`）。**停止方式就是 Ctrl-C**，程序会在 `finally` 里 `client.stop()` 正常收尾。
- **没有 Rerun 显示**。想看画面得另外开一个进程读相机。`--debug_visualize_queue_size=True` 会在**退出时**用 matplotlib 画出动作队列长度随时间的变化曲线 —— 这是调参的关键依据（§5.8）。

**启动成功的标志：**

```
[robot_client] Robot connected and ready
[robot_client] Sending policy instructions to policy server
[robot_client] Action receiving thread starting
[robot_client] Control loop thread starting
```

服务端那边会打印：

```
[policy_server] Client ... connected and ready
[policy_server] Receiving policy instructions from ... | Policy type: pi05 | ...
[policy_server] Time taken to put policy on cuda: xx.xxxx seconds
[policy_server] Running inference for observation #0 (must_go: True)
[policy_server] Action chunk #0 generated | Total time: 123.45ms
```

### 5.7 参数调优：`actions_per_chunk` 与 `chunk_size_threshold`

这两个参数决定「客户端多久向服务端要一次动作」。官方 `async.mdx` 的说明：

| 参数                     | 默认      | 作用                                    |
| ---------------------- | ------- | ------------------------------------- |
| `actions_per_chunk`    | 50      | 一次要多少个动作（**不能超过策略的 `chunk_size=50`**） |
| `chunk_size_threshold` | **0.5** | 队列剩余比例 **≤ 阈值** 时，发一次新观测（取值 [0,1]）    |

工作逻辑（`robot_client.py:416-419`）：

```python
def _ready_to_send_observation(self):
    with self.action_queue_lock:
        return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
```

| 调整方向                       | 效果                                            |
| -------------------------- | --------------------------------------------- |
| `actions_per_chunk` ↑      | 更不容易"队列空"，但长时段预测误差累积，精度下降                     |
| `chunk_size_threshold` ↑   | 发观测更频繁 → 适应性更强（更接近闭环），但**带宽压力更大**，且重叠部分要做动作聚合 |
| `chunk_size_threshold` → 0 | 退化成同步推理（队列空了才发），**最省带宽**，但更容易卡顿               |
| `chunk_size_threshold` → 1 | 每一步都发观测，带宽爆炸                                  |

**调优方法**：先按默认 `0.5` 跑一次，加 `--debug_visualize_queue_size=True`，**退出后看那条队列长度曲线**：

- 曲线**频繁触底（掉到 0）**→ 动作供应不上 → **提高** `chunk_size_threshold`（更早发观测）或**降低** `--fps`；
- 曲线**一直贴着满值** → 观测发得太频繁、浪费带宽 → **降低** `chunk_size_threshold`。

**动作聚合**：不同 chunk 在重叠时段会有多个动作预测，`--aggregate_fn_name` 决定怎么合（`configs.py:29-34`）：

| 值                      | 公式              | 特点          |
| ---------------------- | --------------- | ----------- |
| `weighted_average`（默认） | `0.3·旧 + 0.7·新` | 偏向新预测，兼顾平滑  |
| `latest_only`          | `新`             | 最激进，响应快、可能抖 |
| `average`              | `0.5·旧 + 0.5·新` | 最平滑、响应慢     |
| `conservative`         | `0.7·旧 + 0.3·新` | 最保守、最平滑     |

### 5.8 带宽估算 —— 这才是远程推理的真正瓶颈

**每次观测要传多少？** client 端 `send_observation` 把**原始分辨率**图像 pickle 后分块发送（resize 在服务端做）：

```
2 张 × 480×640×3 字节 = 1,843,200 B ≈ 1.76 MB
加 pickle 开销 ≈ 1.8 MB / 次观测
```

**发送频率**（`chunk_size_threshold=0.5`，`action_chunk_size=50`）：

```
队列从 50 消耗到 25 → 25 tick ÷ 30 fps = 0.83 秒发一次
→ 上行带宽需求 ≈ 1.8 MB ÷ 0.83 s ≈ 2.2 MB/s ≈ 17.5 Mbps
```

> **普通家宽上行通常只有 10~30 Mbps**，还要和视频会议、云盘同步抢带宽。**这是最先撞到的墙**，比 GPU 算力、比网络延迟都更早出问题。

**优化手段（按推荐顺序）：**

| 手段                                      | 效果                          | 代价                                       |
| --------------------------------------- | --------------------------- | ---------------------------------------- |
| **降低相机采集分辨率** 640×480 → 320×240         | 单次观测降到 ~0.44 MB，**带宽降 4 倍** | letterbox 后仍是 168×224，只是源分辨率低一些，**影响很小** |
| **降低 `--fps`** 30 → 15                  | 控制频率和带宽**同时**降半             | 控制变粗，SO-101 这种舵机臂 15 Hz 通常可接受            |
| **降低 `chunk_size_threshold`** 0.5 → 0.3 | 观测发送频率降低                    | 更接近同步，队列更容易空                             |
| **测一下实际上行**                             | 心里有数                        | —                                        |

```cmd
:: 本机测上行带宽（约 10 MB 数据）
curl -o NUL -w "upload speed: %{speed_upload} bytes/s\n" -X POST --data-binary "@你的大文件" https://httpbin.org/post
```

**另外还要注意网络延迟指标**。client 日志里会打印（`robot_client.py:335-343`）：

```
Received action chunk for step #25 | Network latency (server->client): 32.41ms | Deserialization time: 1.20ms
```

RTC 论文实测的局域网延迟是 **6.9 ms**（非移动）~ **21.2 ms**（移动），而 chunk 缓冲有 1670 ms —— 所以 **30~100 ms 的公网延迟完全不是问题**，只要带宽够。

### 5.9 安全与停机

1. **先空跑**。第一次连接时**把机械臂放在安全位置**，用手扶住，用最小 `--fps` 试。
2. **运行中绝对不要用手重置环境**！π₀.₅ 的 chunk 是 50 步 ≈ 1.67 秒开环执行，你在中途挪动箱子，模型在下一个 chunk 之前是"看不见"的。
3. **停止**：本机 client 窗口 Ctrl-C。云端 server 可以继续留着（下次直接用）。
4. **省钱的开关顺序**：
   ```
   ① 本机 client Ctrl-C（机械臂停）
   ② 云端 server tmux 会话 Ctrl-C（或直接不管）
   ③ 做完实验就【关机】—— 实例一开机就按 ¥1.87/小时计费，挂着不动也在烧钱
   ```
5. **权重留在 `/home/featurize/work`** 就不会丢；`~/.cache` 里的东西关机即蒸发。

> **⚠️ 一个必须知道的现实**：远程推理期间，**实例必须保持开机**。所以"做一次真机实验"的成本 ≈ 开机准备时间 + 实验时间，再乘 ¥1.87。准备 10 分钟 + 实验 20 分钟 ≈ ¥1。**建议把一轮实验的多次测试集中做完**，别开开关关。

---

## 6. 推理方案 B：本地推理（含可行性判断）

### 6.1 先给结论：π₀.₅ 在本机跑不了

| 项目        | 你的本机               | π₀.₅ 需要                                             | 结论        |
| --------- | ------------------ | --------------------------------------------------- | --------- |
| GPU       | RTX 3050 Laptop    | —                                                   |           |
| **显存**    | **4096 MiB（4 GB）** | bf16 权重 ≈ 6.6 GB + 激活/KV cache 约 1 GB → **≥7.5 GB** | ❌         |
| 驱动        | 546.30             | 支持 CUDA 12.x                                        | ✅         |
| 可选：CPU 推理 | 主板内存               | fp32 权重 ≈ 13.4 GB                                   | ⚠️ 勉强，但极慢 |

**所以"本地推理"这一章实际是在讲：本机有什么能落地的替代方案。**

| 子方案                  | 可行性         | 用途                   |
| -------------------- | ----------- | -------------------- |
| **B1：本地跑 ACT**       | ✅ 完全可行      | **推荐的"本地方案"**        |
| **B2：π₀.₅ CPU 推理**   | ⚠️ 能跑但极慢    | **仅用于验证流程打通**，不能真机操作 |
| **B3：升级硬件后本地跑 π₀.₅** | 需 ≥12 GB 显存 | 长期方案                 |

> 如果你想强行在 4 GB 上跑 π₀.₅，只剩 4-bit 量化这条路，但：LeRobot 的 π₀.₅ **没有内置量化支持**；而且 π₀.₅ 的输出是流匹配的**连续速度场**，量化对精度的影响比纯分类模型大得多。**不建议在这上面花时间。**

### 6.2 方案 B1：本地跑 ACT（推荐的本地方案）★

你已经有训练好的 ACT 权重（51.67 M 参数，权重文件 206.7 MB），**4 GB 显存绰绰有余**。

```cmd
lerobot-rollout ^
  --strategy.type=base ^
  --policy.path=D:\Desktop\pretrained_model ^
  --device=cuda ^
  --robot.type=so101_follower ^
  --robot.port=COM7 ^
  --robot.id=zihao_follower_arm ^
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" ^
  --fps=30 ^
  --duration=10
```

一行版：

```cmd
lerobot-rollout --strategy.type=base --policy.path=D:\Desktop\pretrained_model --device=cuda --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --fps=30 --duration=10
```

**与 π₀.₅ 的关键差异**：

|                | ACT（本地）                              | π₀.₅（远程）                                   |
| -------------- | ------------------------------------ | ------------------------------------------ |
| 相机键名           | `front` / `wrist`（**不需要改**，ACT 不挑键名） | `base_0_rgb` / `left_wrist_0_rgb`（**必须改**） |
| `--rename_map` | 不需要                                  | 也不需要（直接在相机名里写对）                            |
| `--task`       | 可省                                   | **必填** `"Grab Box"`                        |
| `--duration`   | ✅ **有**，可以设 10 秒自动停                  | ❌ 没有，只能 Ctrl-C                             |
| 显示             | `--display_data=true` 可开 Rerun       | 无                                          |
| 控制方式           | 每 100 步重规划（约 3.33 秒盲走）               | 队列式，边跑边算                                   |

**ACT 本机性能实测**（你的 3050 Laptop）：单次重规划前向 200~560 ms，控制循环稳定在 28 Hz 左右。峰值显存远低于 4 GB。

> **别按"200~560 ms"做容量规划** —— 那是"一次预测 100 步"的耗时，摊到每步只有 2~5 ms。ACT 的 `select_action()` 只在动作队列空时跑网络，其余 tick 只做 `popleft()`（`modeling_act.py:119-125`）。


**想看画面**：加 `--display_data=true`（需要装 `rerun-sdk`）。

**⚠️ 运行中不要重置环境** —— ACT 一次预测 100 步、开环执行约 3.33 秒。时间轴：

```
t=0.00s  重规划：看一眼画面 → 算出 100 步动作（本机耗时 ~200-560 ms）
t=0.20s  开始盲走 ─────────────────────────────────────┐
           ↑ 你在这里把箱子挪了 5 cm                      │ 这 3.33 秒
t=3.33s  走完第 100 步、队列空了 → 这才重新看一眼画面 ────┘
```

想中途重置，用交互模式：`--interactive=true`，然后按提示输入 `/reset`。

### 6.3 方案 B2：π₀.₅ CPU 推理（**仅用于验证流程**）

如果目的只是"确认这份 checkpoint 能加载、能出动作"，可以在本机用 CPU 跑。**不要用它做真机实验。**

**第一步：把权重取回本机**（7 GB，见 §4 的 scp 方式）。

**第二步：起 server（本机，CPU）**

```cmd
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=30
```

**第三步：起 client（本机，CPU）**

```cmd
python -m lerobot.async_inference.robot_client --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --policy_type=pi05 --pretrained_name_or_path=D:\Desktop\pi05_datass --policy_device=cpu --client_device=cpu --fps=30 --actions_per_chunk=50 --chunk_size_threshold=0.2 --debug_visualize_queue_size=True
```

**关键改动**：`--policy_device=cpu`。

**预期表现（务必管理好预期）：**

| 指标 | 预期 |
| --- | --- |
| 模型加载时间 | **数分钟**（13 GB fp32 权重从磁盘读入内存） |
| 内存占用 | **13~16 GB**（笔记本 16 GB 内存会非常吃紧，可能触发换页） |
| 单次推理耗时 | **数十秒量级**（VLM 前向要在 CPU 上处理 1018 个 token） |
| 机械臂行为 | 每几十秒动一下，**不可用于真实任务** |

> 想只验证"pipeline 能不能跑通"，更省事的做法是**不接机械臂**，直接用 `--robot.type=...` 换成假机器人，或者干脆只跑一个离线推理脚本：
> ```cmd
> python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch; from lerobot.policies.pi05.configuration_pi05 import PI05Config; m=PI05Pytorch(PI05Config()); print('params:', sum(p.numel() for p in m.parameters())/1e9, 'B')"
> ```
> 这个不需要下载权重，能立刻验证「3.3 B 参数量」这个数字。

### 6.4 方案 B3：真要本地跑 π₀.₅，需要什么硬件

| 方案 | 显存 | 说明 |
| --- | --- | --- |
| bf16 推理（推荐基线） | **≥ 12 GB** | 权重 6.6 GB + 激活/KV ≈ 1 GB，留余量 |
| fp32 推理 | ≥ 16 GB | 权重 13.4 GB |
| `torch.compile` 后 | 略高 | 编译会有额外开销，但推理更快（实测 67 ms 级） |
| 参考：云端 4090 | 24 GB | 本手册第 5 章用的就是它 |

**买/换机器时的判断**：只要显存 ≥ 12 GB（如 RTX 3060 12G、4070 Ti Super 16G、4080/4090），就能本地跑 π₀.₅ 推理。**但注意：本地推理可行 ≠ 本地训练可行** —— 训练需要 ≥24 GB 且要冻结 VLM，这个门槛比推理高得多。

**本地推理（假设你有 12 GB+ 显存）的命令**，就是把 §6.3 的 `--policy_device=cpu` 换成 `cuda`：

```cmd
:: server 端
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=30

:: client 端（本机同机）
python -m lerobot.async_inference.robot_client --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --policy_type=pi05 --pretrained_name_or_path=D:\Desktop\pi05_datass --policy_device=cuda --client_device=cpu --fps=30 --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average
```

> **同机跑时为什么还用 server/client 双进程？** 因为：
> ① 架构上是解耦的（相机读取和模型推理在两个线程池里，互不阻塞）；
> ② 从远程切到本地，**只需要把 `--server_address` 和 `--policy_device / --pretrained_name_or_path` 改掉**，client 命令可以原样复用；
> ③ 本机跑通了，将来上云只是换个地址。

### 6.5 方案对比：该选哪个？

| | 方案 A 远程推理 | 方案 B1 本地 ACT | 方案 B2 π₀.₅ CPU |
| --- | --- | --- | --- |
| 本机要求 | 无 GPU 要求 | 4 GB 显存 | 16 GB 内存 |
| 单次推理延迟 | **80~150 ms** | 200~560 ms（每 100 步） | 数十秒 |
| 实时可操作性 | ✅ 好（队列式，无空转） | ✅ 好（开环盲走） | ❌ 不可用 |
| 成本 | ¥1.87/小时 | ¥0 | ¥0 |
| 用哪个模型 | **π₀.₅**（3.3 B） | ACT（51.67 M） | π₀.₅ |
| 适合做什么 | **正式实验、论文数据、面试演示** | 快速迭代、数据采集、链路验证 | 确认权重没坏 |
| 主要风险 | 上行带宽 | 模型能力上限 | 无（反正不能用） |

**推荐组合**：

```
日常迭代 / 调数据 / 快速验证     → 方案 B1（本地 ACT，免费，秒级反馈）
换语言指令 / 加新物体 / 出正式结果 → 方案 A（远程 π₀.₅，¥1.87/小时）
面试前确认模型能跑               → 方案 B2 或纯离线脚本
```

---

## 7. 报错速查

### 7.1 环境 / 依赖

| 报错 / 现象 | 原因 | 解法 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'datasets'` | 只装了 `[pi]` | `pip install -e ".[pi,training]"` |
| `ModuleNotFoundError: No module named 'grpc'` | 没装 async extra | `pip install -e ".[async]"`（**两端都要装**） |
| `GatedRepoError` / 401 / 403 拉 `google/paligemma-3b-pt-224` | 门控分词器未授权 | HF 网页接受许可 + `hf auth login`；镜像拉不动就直连 |
| `torch.cuda.is_available()` 为 `False` | 镜像选错（CPU-only） | 重建实例，选 PyTorch + CUDA 镜像 |
| 权重每次开机都重下 | `HF_HOME` 没设到 `work` 目录 | `export HF_HOME=/home/featurize/work/hf` |

### 7.2 训练

| 报错 / 现象 | 原因 | 解法 |
| --- | --- | --- |
| `ValueError: QUANTILES normalization mode requires q01 and q99 stats` | 数据集缺分位数统计 | 你的数据**已有**；真遇到就跑 `lerobot-edit-dataset --operation.type recompute_stats`，或改用 MEAN_STD 映射 |
| `Cannot specify both --policy.path and --policy.type` | 两者互斥（`parser.py:311`） | 二选一：用 `--policy.path=<权重>` **或** `--policy.type=pi05 --policy.pretrained_path=<权重>` |
| `Feature mismatch between dataset/environment and policy config` | 训练时键名没对上 | 补 `--rename_map`，左边数据集键、右边策略键 |
| `All image features are missing from the batch` | 相机键名**一个都没对上**（策略声明的相机在 batch 里全缺） | 检查 `--rename_map` 的左右两边；用 `python -c` 打印 `policy.config.image_features` 与 `dataset.meta.camera_keys` 对照。**注意**：只缺一部分不会报错，缺的那路会自动补 `-1` + 零掩码（`modeling_pi05.py:1239-1243`） |
| `rename_map requires a pretrained policy checkpoint` | 用了 `--rename_map` 但没指定预训练权重（`train.py:287`） | 加上 `--policy.path=<权重>` **或** `--policy.pretrained_path=<权重>`（两者都能配 rename_map，见 §3.1） |
| `CUDA out of memory` | 3.3 B 模型 + 24 GB 卡 | 依次试：`--policy.train_expert_only=true` → `--policy.gradient_checkpointing=true` → `--policy.dtype=bfloat16` → 减 `--batch_size` |
| `Invalid dtype: xxx` | 只允许 `bfloat16` / `float32` | 按提示改（`configuration_pi05.py:141-142`） |
| 训练很慢 | 3.3 B 模型 + VLA 前向 + 梯度检查点 | 加 `--policy.compile_model=true`；调 `--num_workers` |
| 磁盘写满 / `No space left on device` | checkpoint 累积（7 GB/个） | 减小 `--save_freq`，删中间 checkpoint（§3.7） |
| SSH 断开后训练进程消失 | 前台跑，被 SIGHUP 杀掉 | 用 `nohup ... &` 或 tmux（§3.4） |

### 7.3 远程推理（async）

| 报错 / 现象 | 原因 | 解法 |
| --- | --- | --- |
| client 报 `Failed to connect to policy server` | 隧道没起 / 端口不对 / server 没启动 | 检查 SSH 窗口是否还开着；本机测 `127.0.0.1:8080` 是否通 |
| `KeyError: 'observation.images.front'`（服务端） | 相机键名没改成策略期望的键 | `--robot.cameras` 里直接写 `base_0_rgb` / `left_wrist_0_rgb`（§5.6-①） |
| `All image features are missing from the batch` | 同上，一个相机键名都没对上 | 同上 |
| `Connection refused` from `0.0.0.0` | server 绑的是 `127.0.0.1`，但 client 从外部连 | 走隧道时 server 绑 `127.0.0.1` 即可（隧道是从实例内部访问 localhost） |
| server 一直打印 `Running inference...` 但机械臂不动 | client 动作队列没被消费 / 相机没出图 | 看 client 的 `Avg FPS`；确认相机 index 正确（`lerobot-find-cameras`） |
| 机械臂动作**幅度很小或完全不动** | 队列里动作没更新（`must_go` 没触发） | 降低 `chunk_size_threshold` 让观测发得更勤；或提高 `--fps` |
| 动作**卡顿、一顿一顿** | 网络带宽不足，观测发不出去 | 见 §5.8：降相机分辨率 / 降 fps / 降 threshold |
| 抽风、乱动 | ① 漏了归一化统计量（`pretrained_model` 目录拷全了吗）；② 图像 letterbox 补丁没打（§5.5）；③ `--task` 写错 | 逐条核对 |
| **端口暴露在公网后机械臂被人乱控** | `featurize port export` + `add_insecure_port` 无鉴权 | **立即 `featurize port unexport <端口>`**，改用 SSH 隧道 |

### 7.4 本地推理

| 报错 / 现象 | 原因 | 解法 |
| --- | --- | --- |
| `CUDA out of memory` 加载 π₀.₅ | 本机显存 4 GB | **改用第 5 章远程推理**；或 `--policy_device=cpu` |
| CPU 推理时系统卡死 / 疯狂换页 | fp32 权重 13.4 GB > 可用内存 | 关掉其他程序；或换 32 GB 内存机器 |
| `Device 'cuda' is not available. Switching to 'cpu'` | 装的是 CPU 版 torch（`torch.__version__` 带 `+cpu`） | 若只是跑 client 那无所谓；若要本地 GPU 推理才需要装 CUDA 版 torch |

---

## 8. 一页速查（复制即用）

```bash
# ══════════ ① 云端环境（Featurize 4090 24G）══════════
conda create -n pi05 python=3.12 -y && conda activate pi05   # 或直接用镜像自带环境
cd /home/featurize/work
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[pi,training,async]"
conda install -y ffmpeg -c conda-forge

# 持久化 HF 缓存（关键！否则每次开机重下 7GB）
export HF_HOME=/home/featurize/work/hf
export HF_ENDPOINT=https://hf-mirror.com

# 门控分词器：先到网页接受 google/paligemma-3b-pt-224 的许可，再：
hf auth login

# ══════════ ② 数据（本机打包 → 上传 → 解压）══════════
# 本机 cmd:  cd /d D:\Desktop && tar -czf datass.tar.gz datass
# 上传 datass.tar.gz 到 /home/featurize/work/ 后：
mkdir -p /home/featurize/work/datasets && tar -xzf /home/featurize/work/datass.tar.gz -C /home/featurize/work/datasets

# ══════════ ③ 训练（4090 24G：必须冻结 VLM）══════════
lerobot-train \
  --dataset.repo_id=local/datass \
  --dataset.root=/home/featurize/work/datasets/datass \
  --policy.path=lerobot/pi05_base \
  --rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}' \
  --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_datass \
  --job_name=pi05_datass \
  --batch_size=8 --steps=5000 --save_freq=2500 --log_freq=50 --num_workers=10
# 先 --steps=100 冒烟，再改 5000 正式跑；外面套 nohup

# ── π0 版（把 pi05 换成 pi0）：只有 4 处不同，其余照抄上面 ──
#   --policy.path=lerobot/pi0_base
#   --policy.empty_cameras=1            # π0 的 empty_cameras 默认是 0（configuration_pi0.py:68）
#   --output_dir=outputs/train/pi0_datass  --job_name=pi0_datass
#   --steps=8000 --save_freq=2000       # 2245 步 = 1 epoch；50000 步 = 22 epoch ← 别这么填
# 其余必须一致：train_expert_only=true（全参要 30 GB，4090 必 OOM，见 §3.11）

# ══════════ ④ 推理 A：远程（推荐）══════════
# ── 云端：起 server（建议 tmux）
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=30

# ── 本机：开 SSH 隧道（这个窗口保持开着）
# ssh -p <SSH端口> -L 8080:127.0.0.1:8080 featurize@workspace.featurize.cn

# ── 本机：起 client（★ 相机名直接写策略键名；★ 权重路径是【云端】路径）
python -m lerobot.async_inference.robot_client --server_address=127.0.0.1:8080 --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, left_wrist_0_rgb: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --task="Grab Box" --policy_type=pi05 --pretrained_name_or_path=/home/featurize/work/lerobot/outputs/train/pi05_datass/checkpoints/005000/pretrained_model --policy_device=cuda --client_device=cpu --fps=30 --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --debug_visualize_queue_size=True
# 停止：Ctrl-C（没有 --duration 参数）

# ══════════ ⑤ 推理 B：本地（本机 4GB 显存跑不了 π0.5）══════════
# 能落地的是 ACT：
lerobot-rollout --strategy.type=base --policy.path=D:\Desktop\pretrained_model --device=cuda --robot.type=so101_follower --robot.port=COM7 --robot.id=zihao_follower_arm --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" --fps=30 --duration=10
# π0.5 想本地跑：需 ≥12GB 显存；CPU 版仅供验证（--policy_device=cpu，单次数十秒）

# ══════════ ⑥ 托管到 Hugging Face（详见 §4.1）══════════
unset HF_ENDPOINT                                    # ★ 不清掉会 POST 到只读镜像站
pip install -U "huggingface_hub[cli]"
hf auth login                                        # 粘贴【write】权限 token
SRC='/home/featurize/work/lerobot/~/output_lerobot_train/shake/pi05_A'
mv "$SRC" /home/featurize/work/ckpt_pi05_A           # 把字面量 ~ 目录挪干净
CKPT=/home/featurize/work/ckpt_pi05_A/checkpoints/005000/pretrained_model
ls -lh "$CKPT"                                       # 应看到 7 个文件
hf upload TommyZihao/pi05_shake_A "$CKPT" --repo-type model --private
# 上传后加载：--policy.path=TommyZihao/pi05_shake_A
# 或服务端：--pretrained_name_or_path=TommyZihao/pi05_shake_A
```

**四条最容易翻车的规则：**

| # | 规则 |
| --- | --- |
| 1 | **训练用 `--policy.path` 配 `--rename_map`；推理用 `--robot.cameras` 直接写策略键名。两者说的是同一件事：进模型的键必须是 `base_0_rgb` / `left_wrist_0_rgb`。** |
| 2 | **远程推理必须打 §5.5 的 letterbox 补丁**，否则图像被拉伸变形，成功率下降。 |
| 3 | **`--pretrained_name_or_path` 是【服务端】的路径**，不是本机路径。 |
| 4 | **上传到 HF 时 `pretrained_model/` 里的 7 个文件必须平铺在仓库根目录**，不能带第三个参数套一层子目录；且上传要 `unset HF_ENDPOINT`、用 write token。 |

**π₀.₅ 关键超参（来自 `lerobot/pi05_base` 的 `config.json`）：**

| 参数 | 值 |
| --- | --- |
| `paligemma_variant` / `action_expert_variant` | `gemma_2b` / `gemma_300m` |
| `chunk_size` / `n_action_steps` | 50 / 50 |
| `max_state_dim` / `max_action_dim` | 32 / 32 |
| `num_inference_steps` | 10 |
| `image_resolution` | 224 × 224 |
| `tokenizer_max_length` | 200 |
| 分词器 | `google/paligemma-3b-pt-224`（门控） |
| `normalization_mapping` | VISUAL `IDENTITY` / STATE `QUANTILES` / ACTION `QUANTILES` |
| `optimizer_lr` | 2.5e-5 |
| `optimizer_betas` | (0.9, 0.95) |
| `optimizer_grad_clip_norm` | 1.0 |
| `scheduler` | warmup 1000 / 衰减周期 30000 / 终值 2.5e-6（按 `--steps` 自动缩放） |

**async_inference 关键默认值（`async_inference/configs.py`）：**

| 参数 | 默认 | 位置 |
| --- | --- | --- |
| `PolicyServer --host/--port` | `localhost:8080` | `configs.py:54-55` |
| `--fps` | 30 | `configs.py:58` |
| `--inference_latency` | 1/30 s | `configs.py:59-61` |
| `--obs_queue_timeout` | 2 s | `configs.py:63-65` |
| `--chunk_size_threshold` | **0.5** | `configs.py:137` |
| `--aggregate_fn_name` | `weighted_average` | `configs.py:141-144` |
| `--policy_device` / `--client_device` | `cpu` / `cpu` | `configs.py:128-134` |

> 想深入理解 π₀.₅ 为什么这么设计（PaliGemma 主干、SigLIP、action expert、流匹配、注意力掩码、state 离散化进 prompt、3.3 B 参数量每一处维度的手算、20 题面试演练），请看同目录下的 **《03-Pi05-实现细节与原理详解》**。
>
> ACT 侧的完整实操（Featurize 注册细节、数据集打包、权重取回的 5 种方式、`dtype` 版本报错、标定 2047 诊断、安全重置环境、本次实验结果）见 **《01-LeRobot-ACT-全流程实战手册》**。
