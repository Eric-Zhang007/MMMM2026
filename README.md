# 2026 MMMM

## 功能简介
本项目通过 Daning wih he Sars 历史数据，利用神经网络模型反推每周粉丝投票份额。
1. 支持第 1-34 季的不同赛制（Percent 与 Rank）。
2. 实现 Season 28 后的两阶段淘汰机制损失。
3. 自动识别数据面板，支持 GPU/XPU 加速。
4. 集成 Optuna 超参搜索与 TensorBoard 监控。

## 安装环境
pip install torch pandas numpy pyyaml tensorboard optuna scikit-learn

## 快速开始
1. 训练模型: python -m src.train --config configs/default.yaml
2. 续训: python -m src.train --resume_from last

3. 超参搜索: python -m src.tune --n_trials 50
