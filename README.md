# Detection With Association Constraints · MDMT

项目按“运行入口 / 核心实现 / 数据与结果”整理。平时只需要看根目录的四个入口和 `configs/`。

## 目录

```text
Instance-based-Association/
├── prepare.py                # 准备数据：原始 MDMT → 训练/验证/测试清单
├── train.py                  # 训练、验证选优、保存与恢复检查点
├── evaluate.py               # 测试已训练模型，输出指标和预测
├── infer.py                  # 双视角图片推理，输出标框图和 JSON
│
├── association/              # 核心代码；运行入口从这里调用功能
│   ├── data.py               # XML 解析、数据读取与增强
│   ├── models/               # 主干 + MVFF + GIQ + MVOQ
│   ├── losses.py             # Hungarian 匹配及联合损失
│   ├── metrics.py            # MDA、UCR、COCO mAP
│   ├── engine.py             # 建模、DataLoader、公共评估流程
│   └── distributed.py        # torchrun 初始化、全局 batch、各进程随机状态
│
├── configs/                  # 实验参数：resnet50 / vit_b16 / debug_cpu
├── data/
│   ├── raw/MDMT/             # 你下载的完整原始数据，内容保持不变
│   ├── processed/MDMT/       # 转换后的清单、官方 TXT 和数据审计
│   └── archive/              # 旧的不完整镜像，仅留档，训练不读取
├── weights/                  # 官方 ImageNet 预训练权重
├── runs/                     # 训练检查点、日志、指标、预测图片
├── tools/                    # 可选辅助工具：环境检查、权重准备、官方 MDA
├── reference/                # 官方原始代码对照，不是本项目的训练代码
├── tests/                    # 自动测试
├── docs/                     # 详细操作说明、复现差异、验证报告
└── requirements.txt          # 依赖
```

## 最常用的操作

在本项目目录执行，先激活相应 Python 环境。完整数据和预训练权重已经准备好。

```bash
# 检查环境及数据
python tools/check.py

# 需要重新生成清单时使用；不会复制或重新下载原始图像
python prepare.py

# A6000 正式训练
python train.py --config configs/resnet50.json --device cuda --output runs/mdmt_resnet

# 训练结束后，测试验证集选出的最好模型
python evaluate.py --checkpoint runs/mdmt_resnet/best.pt --device cuda --output runs/mdmt_resnet/test.json
```

## 多张 A6000：单机多卡训练

已支持 PyTorch DDP，每张 GPU 一个进程。以 4 张卡为例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 train.py \
  --config configs/resnet50.json --device cuda --batch-size 4 \
  --workers 2 --output runs/mdmt_resnet_ddp
```

这里 `--batch-size 4` 是**所有卡合计**，4卡时每卡1组双视角样本；2卡时每卡2组。
有效 batch = 全局 batch × accumulation_steps。全局 batch 必须能被进程数整除。
若有8卡而想保持总batch=4，只选其中4卡；全用8卡需显式增加总batch，实验设置也随之改变。
验证集只在主进程完整评估，日志和检查点只写一份；最终 `evaluate.py` 仍使用单卡即可。
多卡小检查、续训和实现说明见 [操作手册](docs/usage.md#多卡-a6000-ddp)。

日常修改参数只看 `configs/resnet50.json`；切换 ViT 用 `configs/vit_b16.json`。
`configs/debug_cpu.json` 只用于本地短检查。`paper_25_queries.json` 用于记录论文容量差异，
不能直接覆盖完整 MDMT，不是默认训练配置。

## 代码阅读顺序

1. 看 `configs/resnet50.json`：了解数据、模型、训练参数。
2. 看 `train.py`：理解一个 epoch 的训练和验证。
3. 看 `association/models/model.py`：理解整个模型怎样串联。
4. 再按需要看 `data.py`、各模型模块、`losses.py`、`metrics.py`。

数据流为：**原始数据 → prepare.py → processed 清单 → train.py → best.pt → evaluate.py / infer.py**。

完整命令、断点恢复和迁移服务器见 [操作手册](docs/usage.md)。
模块与论文的对应关系、25→256 查询等差异见 [复现说明](docs/reproduction.md)。
本次结构调整和旧路径对照见 [目录说明](docs/structure.md)。
验证记录放在 [本地流程报告](docs/reports/local_verification.json) 和 [多进程报告](docs/reports/ddp_verification.json)。

当前完成的是本地可执行流程；尚未在 A6000 上完成 100 epoch 正式训练，不能宣称已复现论文精度。
