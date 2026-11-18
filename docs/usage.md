# Detection With Association Constraints — MDMT 复现工程

对应 Yang 等，IEEE TGRS 2026，DOI `10.1109/TGRS.2026.3724951`。
已补齐完整 MDMT 的数据审计、训练/验证、断点续训、测试评估和双视角推理。
当前状态是 **本地流程验证通过**；尚未在 A6000 上完成 100 epoch 训练，不能宣称复现了论文精度。

所有命令均在 `Instance-based-Association` 目录执行。

## 已准备的内容

- 你提供的完整数据：`data/raw/MDMT/`。
- 审计及便携 JSONL：`data/processed/MDMT/`；不复制图像，路径相对 manifest。
- 官方测试真值：`data/processed/MDMT/official_test/`，固定官方 Git 提交并记录 SHA256。
- ResNet-50 / ViT-B16 官方 ImageNet 权重：`weights/`，已下载并校验。
- 配置：`configs/resnet50.json`（默认 ResNet）、`configs/vit_b16.json`、`configs/paper_25_queries.json`。
- CPU 短流程配置：`configs/debug_cpu.json`；仅用于检查代码，不比较精度。
- 结果与日志：`runs/`。旧的不完整 HF 镜像在 `data/archive/incomplete_hf/`，新流程不读取它。

## 数据核查结果

| 划分 | 同步帧组 | 双视角图像 | 序列对 | 每帧最大实例数 |
|---|---:|---:|---:|---:|
| train | 12,026 | 24,052 | 25 | 191 |
| val | 1,930 | 3,860 | 5 | 221 |
| test | 5,883 | 11,766 | 14 | 201（XML） |

合计 39,678 张图像；按官方现成划分，序列 ID 无交叉。所有 JPEG 已验证可读，88 个 XML 可解析。

**重要规则：XML 帧号对应排序后图像的零基序号，不能直接用文件名减一。**
例如 `22-1/00000201.jpg` 对应 XML `frame=0`。这与官方 `supplement_MIA.py` 的枚举方式一致。
XML track ID 在视角间共享；官方测试 TXT 的 ID 为 XML ID + 1，TXT frame 为 XML frame + 1。
逐行比对官方测试 TXT 的 **600,923** 条记录，ID/帧号和裁剪后的框坐标均与 XML 对应。
XML 测试框共有 601,270 条，比官方测试 TXT 多 347 条；正式测试使用后者，避免混用真值口径。

遮挡 `occluded=1` 不代表目标不存在，仍参与检测和回归；`outside=1` 才视为不在画面内。
框超出边界时裁剪并计数；不静默丢弃超容量目标。

## 环境

A6000 上新建独立 Python 环境，先安装与服务器驱动兼容的 CUDA 版 torch/torchvision，
再安装 `requirements.txt` 中其余依赖。可从 PyTorch 官方安装选择器获取匹配命令。
不要直接复制本地 macOS 的 `.venv` 到 Linux。

```bash
python -m pip install -r requirements.txt
python tools/check.py --config configs/resnet50.json
python -m unittest discover -s tests -v
```

本地测试环境为 Python 3.13.9 / torch 2.12.0 / torchvision 0.27.0 / pycocotools 2.0.11，
只有 CPU。`check_environment.py` 会打印实际 CUDA、GPU 和显存信息，检查数据、查询容量和权重。
本地 `.venv` 继承已有 torch，仅用于当前工作站检查，不是可移植依赖包。

## 重建数据与权重

当前已生成，无需重复下载。若换位置或重新解压数据，可以执行：

```bash
python prepare.py --verify-images
python tools/prepare_weights.py
```

`prepare.py` 只在缺少官方 TXT 时联网补齐；`tools/prepare_weights.py` 只在缺少权重时下载。训练、评估和推理读取本地文件。
原始目录和 manifest 的相对位置保持一致即可迁移整个项目。

## A6000 训练

```bash
python train.py --config configs/resnet50.json --device cuda --output runs/mdmt_resnet
```

默认 100 epochs，AdamW，lr=1e-4，weight decay=1e-5，batch=4，AMP，
梯度裁剪 0.1；仅训练集使用随机水平翻转、裁剪和颜色抖动。验证/测试无增强。
裁剪后重新计算可见性，所有视角均不可见的实例才移除。
每个 epoch 验证，按验证集 MDA 保存 `best.pt`，同时保存 `last.pt`、优化器、AMP 和随机数状态。
无额外学习率调度，论文未说明调度方案。小 batch 默认固定 ResNet BN 运行统计，仿射参数仍可训练。

若显存不足，显式减小 micro-batch 并做梯度累积，保持有效 batch=4：

```bash
python train.py --config configs/resnet50.json --device cuda --batch-size 2 --accumulation-steps 2 --output runs/mdmt_resnet
```

恢复训练（模型配置须一致；在 epoch 边界恢复，不支持 batch 中途恢复）：

```bash
python train.py --config configs/resnet50.json --device cuda --resume runs/mdmt_resnet/last.pt --output runs/mdmt_resnet
```

ViT 版本：

```bash
python train.py --config configs/vit_b16.json --device cuda --output runs/mdmt_vit
```

首次上服务器可先进行真实分辨率短检查，再开始正式训练：

```bash
python train.py --config configs/resnet50.json --device cuda --limit-train 4 --limit-val 4 --max-train-batches 1 --epochs 1 --output runs/gpu_check
```

短检查生成的权重不能用于报告复现精度；不要从它恢复正式实验。

## 测试与输出

只用验证集选阈值和超参数；测试集用于最终评估。

```bash
python evaluate.py --checkpoint runs/mdmt_resnet/best.pt --device cuda --split test --output runs/mdmt_resnet/test_metrics.json --export-predictions runs/mdmt_resnet/test_predictions
python tools/score_official_mda.py --predictions runs/mdmt_resnet/test_predictions --output runs/mdmt_resnet/official_mda.json
```

`evaluate.py` 输出：一对一匹配 MDA、UCR、关联 precision/recall/F1、标准 COCO mAP/AP50/AP75、损失和逐序列 MDA。
`score_official_mda.py` 调用保存的、未修改匹配逻辑的官方 `mango_eval.py`，单独报告 `MDA_legacy_official`。
两种实现对重复框等边界情况的处理不同，不能互相冒充。
置信阈值默认 0.5，论文未给出，须仅在验证集上调节。

COCO bbox AP 使用标准 maxDets=100。密集图像超过 100 个目标时，即使真值作为预测，
AP 也可能小于 1；`tools/check.py --oracle` 的真实样本检查记录了这一点。
不应修改 maxDets 后继续把结果称为相同口径的标准 AP。

双视角图片推理（JSON + 每视角标框 JPG）：

```bash
python infer.py --checkpoint runs/mdmt_resnet/best.pt --device cuda --images data/raw/MDMT/test/1/26-1/00000001.jpg data/raw/MDMT/test/2/26-2/00000001.jpg --output runs/example
```

同一个 query ID 表示同一时刻跨视角的同一实例，不保证跨视频帧身份连续；本项目不实现时间跟踪。
不执行 NMS 或基于 GT 的推理时关联修正。

## 本地短流程复查

```bash
python train.py --config configs/debug_cpu.json --limit-train 2 --limit-val 2 --max-train-batches 2 --output runs/local_check
python evaluate.py --checkpoint runs/local_check/last.pt --split test --limit 2 --output runs/local_check/test.json
python tools/check.py --oracle
```

`--limit` 和低分辨率仅用于调试。当前 smoke 权重只优化了少量样本，指标为零是预期，
不属于论文复现结果，也没有使用测试集训练。

## 严格复现仍需明确的差异

**25 查询不能覆盖完整 MDMT。** 原论文及原始代码写 N=25，但完整数据单帧最多有 221 个全局实例。
默认工程配置 N=256 以覆盖所有标注；`paper_25_queries.json` 保留 N=25，并在容量不足时明确报错。
不会筛掉密集帧或只挑 25 个目标来伪造一致设置。需要作者说明其 MDMT 目标筛选/查询设置，
才能判断与论文是完全相同的协议。

默认 C=1，沿用公开模型代码默认值，视行人/自行车/车辆为统一“目标”。原 XML 中有少量 bus 和
368 个跨视角 track 类别冲突；类别信息保存在原 XML 和审计报告中。
`--class-mode native` 保留四类，但若同一实例跨视角类别矛盾会报错，不会擅自生成全局类别。

通道拼接降维、三元注意力的可计算形式以及损失权重没有完整公开。
实现选择和论文对应关系详见 [复现说明](reproduction.md)，本地验证详见 [验证报告](reports/local_verification.json)。
因此当前提供的是可执行且有审计的论文方法重建，不能声称已经精确重建了未公开细节。

## 迁移到服务器

需要传输：代码、configs、weights、完整原始数据和 data/processed/MDMT。
可排除 `.git/`、`.venv/`、`runs/`、旧镜像 `data/archive/incomplete_hf/` 以及 `data/raw/MDMT/checkpoints/`。
训练入口不使用那些基线权重。本次未连接服务器、未运行正式训练。


## 多卡 A6000（DDP）

在项目目录运行，先确认 `nvidia-smi` 和 `python tools/check.py` 均能识别GPU。
不需要安装额外分布式框架，使用 PyTorch 自带的 torchrun / DDP。
实现依据：[DDP](https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html)、
[torchrun](https://docs.pytorch.org/docs/2.12/elastic/run.html)。

先用4张卡进行短检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 train.py \
  --config configs/resnet50.json --device cuda --batch-size 4 --workers 0 \
  --epochs 1 --limit-train 16 --limit-val 8 --output runs/gpu_ddp_check
```

确认通过后，从预训练权重重新开始正式训练，不使用短检查权重续训：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 train.py \
  --config configs/resnet50.json --device cuda --batch-size 4 --workers 2 \
  --output runs/mdmt_resnet_ddp
```

后台运行：

```bash
mkdir -p runs/mdmt_resnet_ddp
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup torchrun --standalone --nproc-per-node=4 train.py \
  --config configs/resnet50.json --device cuda --batch-size 4 --workers 2 \
  --output runs/mdmt_resnet_ddp > runs/mdmt_resnet_ddp/train.log 2>&1 &
```

保持相同卡数和训练参数恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 train.py \
  --config configs/resnet50.json --device cuda --batch-size 4 --workers 2 \
  --resume runs/mdmt_resnet_ddp/last.pt --output runs/mdmt_resnet_ddp
```

训练后单卡测试即可，不要用torchrun重复启动完整测试：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --checkpoint runs/mdmt_resnet_ddp/best.pt \
  --device cuda --output runs/mdmt_resnet_ddp/test_metrics.json \
  --export-predictions runs/mdmt_resnet_ddp/test_predictions
python tools/score_official_mda.py --predictions runs/mdmt_resnet_ddp/test_predictions \
  --output runs/mdmt_resnet_ddp/official_mda.json
```

### batch、进程和数据口径

| 卡数/进程数 | --batch-size（全局micro-batch） | 每卡micro-batch | 累积次数 | 有效batch |
|---|---:|---:|---:|---:|
| 1 | 4 | 4 | 1 | 4 |
| 2 | 4 | 2 | 1 | 4 |
| 4 | 4 | 1 | 1 | 4 |
| 2 | 2 | 1 | 2 | 4 |
| 8 | 8 | 1 | 1 | 8（改变实验设置） |

默认学习率仍为1e-4，不按卡数自动放大。`--workers` 是每进程的加载进程数，
例如4卡×workers2共8个加载进程。DDP加载进程显式使用spawn，避免NCCL与fork混用。`--limit-train` 限制分片前的全局数据集，
`--max-train-batches` 则限制每个进程的batch数。
短测的batch限制在BatchSampler中截断索引流，让多进程DataLoader正常耗尽；
不再通过训练循环提前break，避免仍有预取数据时直接析构worker。
DistributedSampler按rank分片，每轮set_epoch重新洗牌；样本总数不能整除卡数时，
使用其标准尾部补齐（可能重复少量样本），数量记录于 `distributed.json`，不会静默丢弃标注。

各rank先独立匹配，再按全局可见框数量归一化回归损失，以匹配DDP的梯度平均。
梯度累积在非更新步骤使用no_sync。验证集不分片、不补齐：仅rank0用未包装模型完整评估，
其余rank等待；避免重复样本和错误地平均各卡AP。
只有rank0写config/history/checkpoint；保存的state_dict不带module前缀，单卡推理直接加载。
检查点记录各rank随机状态，同卡数在epoch边界恢复；改变卡数可以加载权重/优化器，
但会提示随机流重启，不承诺精确续训。CUDA/CPU、不同环境以及worker设置变化也不保证逐位一致。

### 进度条和 batch 容量测试

安装新增依赖：`python -m pip install 'tqdm>=4.66,<5'`。
默认 `--progress auto`：交互终端显示训练/验证进度条，输出重定向（例如nohup）时显示普通日志。
`--progress bar` 强制进度条，`--progress log --log-every 20` 每20批记录一行（首尾批也记录）。
nohup日志不适合用tail显示原地刷新的进度条；要长期保留交互进度条可在tmux会话里运行。
进度条显示batch数、百分比、耗时、ETA、batch/s、当前loss和累计avg；CUDA时显示peak_GiB。
loss/avg是rank0的损失，peak_GiB是rank0本轮训练已分配张量显存峰值，均非多卡合计。
一轮结束输出简短摘要；history.jsonl仍保存完整全局训练损失及验证指标，新增train_seconds、
train_samples_per_second（双视角样本组/秒，含加载/通信）及rank0_train_peak_allocated_gib，便于比较吞吐。
显存峰值不包含所有CUDA/NCCL开销，仍需同时观察nvidia-smi中的两张卡。
启动时保留数据加载、模型搬入设备、DDP初始化及首批数据到达的提示。
已有进程不会热更新，需同步代码后重新启动；若已在正式训练，优先等本轮保存后再重启并resume。

48GB显存不要求必须增大batch。当前默认全局batch4遵循论文所报训练设置；
增大batch属于额外实验，会改变每轮优化步数和梯度统计，不能保证相同精度，代码不自动缩放学习率。
两卡全局batch4/8/16/32分别对应每卡2/4/8/16组双视角样本；显存按每卡独立计算。
先逐档短测，比较吞吐和显存，再决定；不要按剩余显存比例直接推算batch。

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_P2P_DISABLE=1 \
torchrun --standalone --nproc-per-node=2 train.py \
  --config configs/resnet50.json --device cuda \
  --batch-size 16 --workers 0 --epochs 1 \
  --limit-train 640 --max-train-batches 20 --limit-val 64 \
  --progress bar --output runs/batch_probe_16
```

可分别把batch改为4/8/16/32，并对应更换输出目录。每次都从相同预训练骨干开始，
不加载小样例检查点。短测不代表完整数据或长期训练的显存保证，应留出余量。
用户当前服务器曾在默认P2P路径卡住，禁用P2P后两卡短测通过，命令保留该临时绕行设置；
此设置只用于受影响环境，不应无条件推广到其他服务器。

### 本地验证范围

若多轮MDA/mAP仍为0，先检查history.jsonl的完整数值与分项损失。训练摘要现使用6位有效数字，
避免很小的非零指标被四位小数舍入成0。只读诊断命令：

```bash
CUDA_VISIBLE_DEVICES=1 python tools/diagnose.py \
  --checkpoint runs/mdmt_resnet_batch32/last.pt --device cuda \
  --limit 16 --output runs/mdmt_resnet_batch32/diagnosis.json
```

在train/val各取16个均匀分布样本，关闭增强、不更新权重；报告前景/可见性概率、
AP候选数量、MDA过滤后框数、忽略分数的GT最大IoU、query间框坐标标准差和预测/GT宽高。
IoU覆盖不考虑一对一匹配，仅用于定位，不能当作AP或MDA。诊断应优先用last.pt，
若验证指标长期为0，best.pt可能仍是第一轮。运行前停止训练或选择空闲显卡，避免争抢资源。

本地已用CPU/Gloo双进程检查：真实MDMT短训练、累积同步、验证、保存、恢复、
多卡检查点单进程测试，以及不均匀可见目标数下的全局梯度一致性。
本地没有NVIDIA CUDA/NCCL，因此服务器上仍应先执行上述多卡短检查。
新增分布式辅助代码只有 `association/distributed.py`，其余修改位于原train/engine/losses中。
