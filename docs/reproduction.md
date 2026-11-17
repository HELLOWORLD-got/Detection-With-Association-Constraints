# 论文方法与实现对应

依据：用户提供的 TGRS 2026 PDF（DOI 10.1109/TGRS.2026.3724951），
作者仓库 https://github.com/yyyyyxq/Instance-based-Association，
MDMT 官方 https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking。
MDMT 原版评估脚本保存在 reference/mdmt_official，作为对照而非本项目原创代码。
官方 TXT 使用固定提交 551f90d998087ea2d02df75700e3c7739c4ecbe1。

## 实现映射

| 论文模块 | 实现 | 状态/选择 |
|---|---|---|
| ResNet50 (Eq.5) | association/models/backbone.py | ImageNet V2，stride32 空间特征，1×1 投影到256 |
| ViT (Eq.5) | association/models/backbone.py | ImageNet V1，保留 patch token，移除分类头；矩形位置编码插值与补齐属于实现选择 |
| CA (Eq.6) | association/models/multi_view_feature_fusion.py | 3层，8 heads，跨视角 Q/其他视角 KV，残差+post-LN+FFN；不再错误调用 EncoderLayer(Q,K,V) |
| Concat+SA (Eqs.7–8) | 同上 | 按通道拼接后1×1降至256再3层SA；论文未明确降维算子 |
| GIQ (Eqs.9–10) | association/models/global_instance_query.py | 学习查询、分离x/y的2D sine/cos编码、全局分类；保留公开模型 API |
| MVOQ (Eqs.11–13) | association/models/multi_view_object_query.py | 每视角/每实例学习query，加绑定的全局实例特征，再对本视角空间token做cross attention |
| Hungarian (Eqs.20–25) | association/losses.py | N×N含背景匹配；可见框L1+GIoU；分类和逐视角可见性损失；空实例/空视角掩码 |
| 训练协议 | train.py + association/engine.py | AdamW 1e-4 / wd 1e-5 / batch4 / 100epoch，训练集增强、验证选优、AMP及epoch边界恢复 |
| MDA | association/metrics.py + tools/score_official_mda.py | 明确的一对一版本和未改匹配逻辑的官方旧版并列 |
| UCR | association/metrics.py | 对所有视角presence完全相符，框IoU严格>.6，类别相符；全视角一对一匹配 |
| mAP | association/metrics.py | pycocotools标准COCO bbox AP，maxDets=100，AP50/AP75 |

论文 Eq.11 称 object query 为 Q、单实例为 K、图像特征为 V，
却没有定义不同 token 数的 K/V 如何构成标准注意力，也没有给出可执行的三元算子。
本实现采用条件查询，是合理的可计算重建；不能冒充已确认的作者原算子。
损失 lambda_box/reg/conf/L1/GIoU 目前均为1，论文没有公布数值，配置可修改。
实际训练应只用验证集选这些参数，并保存配置，不用测试集调参。

## 与完整 MDMT 冲突的论文设置

公开代码默认25查询，但真实训练/验证/测试最大实例数分别为191/221/201。
默认改为256以保持全部标注。这是明确的容量扩展，不是严格同参复现。
未找到作者对密集 MDMT 的筛选、裁剪或查询扩展说明；需作者补充才能宣称完全同协议。

公开代码默认类别数为1，默认使用类别无关目标检测/关联。
原 XML 中存在四种标签和368个跨视角track类别冲突。
不擅自把 bus 合为 car 或将不同类别分别当成不同物理实例。

405×720在论文中明确属于自建数据，MDMT训练分辨率未交代；本项目将其作为显式默认，
不宣称它是已经核实的 MDMT 原设置。两张4090的论文实验与单A6000不同。
固定 ResNet BN 运行统计、梯度裁剪0.1、增强范围和置信阈值0.5均为实现选择。
没有作者提供的已训练模型，也不使用 MDMT 包里的 MIA-Net 检测器权重充当复现权重。

## 指标口径

MDA采用TA/(GA+FA+MA)，先按帧/视角对计算，再在序列内平均，最后序列等权平均。
同时输出frame加权平均，避免序列长短带来的混淆。
主实现对预测/GT做IoU>=0.5最大基数一对一匹配；关联不检查类别，符合官方MDMT关联评估。
空GT且无预测的视角对记为1。UCR检查所有真实全局实例，包括只在一个视角存在的实例。
关联 precision/recall/F1 为累计关联计数的 micro 指标，不冒充未明确平均方式的论文AP/AR。

官方 mango_eval.py 的 calAAS 原逻辑会受重复框、多对一和边界情况影响，
因此单独命名 MDA_legacy_official。它不替换主实现，也不静默修正。
默认测试 manifest 用公开TXT真值，XML中额外347个测试框不混入正式评估。
官方TXT的600,923条标注均与本地XML的ID/帧号和框坐标对应（边界裁剪后误差<.01px）。

pycocotools对每视角图像保留最多100个检测，AP预测分数为全局前景类别概率×视角可见性概率，
保留分数>=.001的候选；关联/UCR使用全局argmax非背景且视角可见性>=.5的query。
置信度筛选和argmax规则未在论文中充分交代，属于可配置的实现选择。
所有query在同一视角最多有一个框，跨视角沿用query槽位关联；没有NMS和后处理匹配。

## 论文对照目标（表I，非本项目实测）

| 主干 | MDMT MDA (%) | MDMT mAP (%) | MDMT mAP50 (%) |
|---|---:|---:|---:|
| ResNet | 46.72 | 27.35 | 54.89 |
| ViT | 47.25 | 27.98 | 55.92 |

论文表I的UCR列属于自建数据集，不是MDMT，不能拿它直接对比本项目MDMT UCR。
本地短检查不产生可与上表比较的性能结论；只有完整训练、验证选优和最终测试完成后，
才能填入实测分数并分析差距。即使接近目标，查询容量/未公开算子等差异仍须如实报告。
