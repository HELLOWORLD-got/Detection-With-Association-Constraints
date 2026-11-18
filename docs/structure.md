# 为什么这样整理

根目录只放用户直接运行的程序。`association/` 只放算法和公共实现，不包含命令行入口。
例如 `evaluate.py` 从 `association.engine` 获取 DataLoader，不再反过来导入 `train.py`。

## 四个入口

| 文件 | 什么时候用 | 读入 → 产出 |
|---|---|---|
| prepare.py | 首次准备或重新转换数据 | data/raw/MDMT → data/processed/MDMT |
| train.py | 训练和恢复训练 | 配置 + 清单 → runs 下的 best.pt / last.pt |
| evaluate.py | 定量评估已有模型 | 检查点 + 测试集 → 指标 JSON / 预测 JSON |
| infer.py | 查看指定双视角图片的预测 | 检查点 + 两张图片 → 标框图片 / 预测 JSON |

## association 核心包

- `data.py`：负责 XML 标注、图像读取、同步视角及数据增强。
- `models/model.py`：将主干、MVFF、GIQ、MVOQ 串成完整网络；其余文件分别实现这些模块。
- `losses.py`：预测和真值如何做 Hungarian 匹配，以及怎样计算训练损失。
- `metrics.py`：预测怎样转换为 MDA、UCR、COCO mAP。
- `engine.py`：训练、测试共享的建模、数据加载和评估逻辑。
- `distributed.py`：多卡进程初始化、全局batch分配和各rank随机状态；单卡不必阅读。

模型、损失、数据、指标各有一个明确位置，根目录入口只负责组织执行流程。

## 辅助内容

`tools/check.py` 是环境和数据检查，增加 `--oracle` 可检查评估器；
`tools/prepare_weights.py` 准备官方权重；`tools/score_official_mda.py` 跑原版 MDA。
`tools/audit_paper.py` 核对保留的原作者模型代码；`tools/overfit.py` 仅作学习能力诊断。
原作者源码固定于 `reference/author_model/`，与 `association/models/` 的可运行重建实现明确区分。
数据准备各阶段合在 `tools/prepare_data.py`，通过根目录的 `prepare.py` 调用。
旧 HF 镜像下载程序放到 `tools/legacy/`，正常复现不需要打开。
`reference/` 是核对论文和数据协议时保存的官方代码，不修改它来实现训练功能。

## 旧位置 → 新位置

| 原位置 | 新位置 |
|---|---|
| model/ | association/models/ |
| model/loss.py | association/losses.py |
| data.py + mdmt.py | association/data.py |
| runtime.py | association/engine.py |
| metrics.py | association/metrics.py |
| scripts/ 三个数据准备入口 | prepare.py + tools/prepare_data.py |
| scripts/check_environment.py、verify_oracle.py | tools/check.py、tools/check.py --oracle |
| configs/mdmt_full.json | configs/resnet50.json |
| configs/mdmt_vit.json | configs/vit_b16.json |
| configs/smoke_cpu.json | configs/debug_cpu.json |
| 根目录 Multi-Drone-Multi-Object-Detection-and-Tracking/ | data/raw/MDMT/ |
| data/mdmt_complete/ | data/processed/MDMT/ |
| data/MDMT/（旧残缺镜像） | data/archive/incomplete_hf/ |
| docs/*.json | docs/reports/*.json |

数据目录使用同一磁盘上的重命名移动，没有复制大数据，也没有删除原始图像、标注或权重。
转换清单已按新位置重新生成。已有 `.pt` 的模型参数键保持不变；加载时会自动将配置中
旧的 `data/mdmt_complete/` 前缀映射到新位置，因此原检查点仍可评估和恢复训练。
历史 `runs/` 日志保留原文，其中旧命令属于当时的记录；后续使用当前 README 的命令。

这次只整理职责和路径，不调整模型结构、损失、增强、查询数或指标口径。
