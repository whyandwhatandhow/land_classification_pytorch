# land_classification_pytorch

基于 PyTorch 的地物（遥感影像）分类仓库，使用 ResNet 系列网络对 EuroSAT 数据集（多类别地表覆盖）进行训练、评估与推理。

**主要功能**
- 支持数据预处理与组织：`data_process.py`
- 模型定义与可选结构：`model.py`, `model_pro.py`, `model_pro_max.py`
- 训练流程：`model_train.py`
- 测试/评估流程：`model_test.py`
- 可视化脚本（预测展示 / 错分查看）：`display.py`
- 若干训练得到的权重文件位于仓库根目录（例如 `best_model_model.pth`）

**目录结构（仓库重要文件/目录）**
- `data/`：样例/数据集（仓库中包含 `eurosat/2750/` 的子目录）
- `model.py`：基础模型定义（ResNet 或其他 backbone 封装）
- `model_pro.py`, `model_pro_max.py`：模型变体或改进版本
- `model_train.py`：训练脚本（训练循环、优化器、日志保存）
- `model_test.py`：测试脚本（加载模型并评估指标）
- `data_process.py`：数据加载、划分、可能的增强与预处理工具
- `display.py`：用于可视化预测结果与错分样本
- `runs/`：TensorBoard 运行/日志输出

快速开始
-----------
下面的步骤假设你在 Linux 环境并使用 `bash`。

1) 克隆/进入仓库

```bash
cd /path/to/your/workspace
# 如果还没克隆：git clone <repo>
cd land_cover_classification
```

2) 创建 Python 虚拟环境并安装依赖

（仓库没有包含 `requirements.txt`，建议使用下面的最小依赖）

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision tensorboard matplotlib numpy pillow scikit-learn tqdm
```

3) 准备数据

- 本项目示例使用 EuroSAT 数据集（RGB 或多光谱已预处理为文件夹结构）。仓库中 `data/eurosat/2750/` 含示例类别目录。
- 若要使用新的数据，请确保按照 "每类一个文件夹" 的格式组织数据：

```
data/eurosat/<images_per_class>/
	ClassA/
		img1.jpg
		img2.jpg
	ClassB/
		img3.jpg
```

- 若需额外预处理（裁剪、重采样、增强等），请查看并修改 `data_process.py` 中的函数。

4) 训练模型

最基本的训练命令（示例）：

```bash
python model_train.py --data_dir data/eurosat/2750 --epochs 50 --batch_size 32 --lr 0.001
```

说明（依据 `model_train.py` 可用的参数而定）：
- `--data_dir`：数据路径
- `--epochs`：训练轮数
- `--batch_size`：批大小
- `--lr`：初始学习率

训练中会在根目录保存最佳模型权重（例如 `best_model_model.pth`、`best_model_model_pro.pth` 等），并在 `runs/` 下写入 TensorBoard 日志。

5) 测试 / 评估

使用训练好的权重评估或在测试集上推理：

```bash
python model_test.py --data_dir data/eurosat/2750 --checkpoint best_model_model.pth --batch_size 32
```

脚本将输出常用指标（准确率、混淆矩阵等），并可将错分样本保存到 `runs/test_misclassified/` 便于查看。

6) 可视化结果

运行 `display.py` 来查看若干推理结果或错分样本：

```bash
python display.py --input_dir runs/test_misclassified --num 20
```

模型与实验说明
----------------
- 已实现多种模型变体，主文件为 `model.py`（基础结构），而 `model_pro.py` 和 `model_pro_max.py` 为实验/增强版。训练脚本会依据传入参数选用不同模型类。
- 仓库附带若干训练好的权重文件以供快速评估：
	- `best_model_model.pth`
	- `best_model_model_pro.pth`
	- `best_model_model_pro_max.pth`

常见问题与调试建议
------------------
- 如果显存不足，请减小 `--batch_size` 或使用更小的输入分辨率。
- 若训练不收敛，尝试：降低学习率、使用学习率衰减 (scheduler)、或启用更强的数据增强。
- 确认 PyTorch 与 CUDA 版本兼容（若使用 GPU）。

再现与可扩展性
----------------
- 为了重现某次实验，记录训练命令、参数、随机种子与模型权重路径是必要的。可在 `model_train.py` 中加入对配置/命令行参数的保存（例如保存到 `runs/` 下的 `config.json`）。
- 若你需要把模型部署到推理服务器或导出为 ONNX，请先在验证集上确认性能再导出。

贡献与联系方式
----------------
- 欢迎提 issue 或 PR：修复 bug、添加配置、增加训练脚本中的参数、或改进模型结构。
- 若需联系作者，可在仓库页面留言。

许可证
------
默认未指定许可证——如果你计划公开发布或允许他人使用，请在仓库根目录添加合适的 LICENSE 文件（例如 MIT）。

--
已更新文件：`README.md`。如需我把 README 再翻为英文或加入具体参数示例（来自 `model_train.py`），我可以继续处理。