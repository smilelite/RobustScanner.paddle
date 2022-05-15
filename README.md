# RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
    - [5.1 基于Inference的推理]()
    - [5.2 基于Serving的服务化部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()

## 1. 简介

RobustScannerz收录于ECVV2020，它研究了注意框架编解码器译码过程的内在机制。发现LSTM的查询特征向量不仅编码了上下文信息，还编码了位置信息，并且在解码靠后的时间步长上，上下文信息主导了查询。由此得出结论，位置信息的确实可能是导致注意力偏移的主要原因。因此引入一种新的位置增强分支和动态融合模块来缓解无上下文场景下的误识别问题。本文采用的位置注意力模块，成为了后续OCR论文常用方式之一。论文在规则和不规则文本识别基准测试上取得了当时最先进的结果，在无上下文基准测试上没有太大的性能下降，从而验证了其在上下文和无上下文应用程序场景中的健壮性。
<p align="center">
 <img src="./doc/robustscanner_framework.png" align="middle" width = "600"/>
<p align="center">

**论文:** [RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition](https://arxiv.org/pdf/2007.07542.pdf)

**参考repo:** [mmocr](https://github.com/open-mmlab/mmocr)

在此非常感谢[mmocr](https://github.com/open-mmlab/mmocr)，提高了本repo复现论文的效率。

本复现采用[ppocr](https://github.com/PaddlePaddle/PaddleOCR),感谢ppocr的开发者。

**aistudio体验教程:** [后续将添加](url)


## 2. 数据集和复现精度

本复现采用的训练数据集和测试数据集参考[mmocr文档](https://mmocr.readthedocs.io/zh_CN/latest/textrecog_models.html#robustscanner)。(其中icdar2011未下载到且其大多数包含在icdar2015中，所以本复现没有使用)

本复现的效果如下
|           |                 数据集                  | IIIT5K | SVT  | IC13 | IC15 | SVTP | CT80 | Avg   |
| --------- | --------------------------------------- | ----- | ---- | ---- | ---- | ---- | ---- | ----- |
| 论文   |          MJ(891W) + ST(726W) + Real        |  95.4  | 89.3 | 94.1 | 79.2 | 82.9 | 92.4 | 88.88 |
| 参考   | MJ(240W) + ST(240W) + SynthAdd(121W) + Real|  95.1  | 89.2 | 93.1 | 77.8 | 80.3 | 90.3 | 87.63 |
| 复现   | MJ(240W) + ST(240W) + SynthAdd(121W) + Real|  95.6  | 90.4 | 93.2 | 77.2 | 81.7 | 88.5 | 87.77 |

模型链接稍后将给出


## 3. 准备数据与环境


### 3.1 准备环境

- 框架：
  - PaddlePaddle >= 2.2.0
- 安装方式
直接使用pip进行安装
`pip install paddlepaddle-gpu`
paddlepaddle安装成功后，使用pip install -r requirements.txt安装依赖。
具体环境配置可参考[ppocr](https://github.com/PaddlePaddle/PaddleOCR)

### 3.2 准备数据

使用的数据集已在AIStudio上公开，地址如下
[训练集](https://aistudio.baidu.com/aistudio/datasetdetail/138433): 真实数据由ICDAR2013, ICDAR2015, IIIT5K, COCO-Text的训练集组成。
合成数据由Synth90K(240W)， SynthAdd(121W)， Synth800K(240W), synthadd组成
[测试集](https://aistudio.baidu.com/aistudio/datasetdetail/138872)：包含规则文本（IIIT5K、SVT、ICDAR2013）和不规则文本（ICDAR2015、SVTP、CUTE80）组成

为方便存储，所有数据都已经打包成lmdb格式。

## 4. 开始使用

本复现基于PaddleOCR框架，需要进行部分修改，主要是加入RobustScanner数据读取方式，backbone, RobustScanner_head，以及在训练和评估脚本中加入RobustScanner字段。
- 数据读取
主要是在./ppocr/data/imaug/rec_img_aug.py中加入了`RobustScannerRecResizeImg`
- backbone
复用./ppocr/modeling/backbones/rec_resnet_31.py,在参数初始化上做了一些修改，这里没有新建一个rec_resnet_31.py,需要注意。
- RobustScanner_head
见./ppocr/modeling/heads/rec_robustscanner_head.py
- loss
复用SARLoss


整体训练流程与PaddleOCR一致，可参考PaddleOCR的流程，下面进行简述。
### 4.1 模型训练
- 修改配置文件，配置文件地址为configs/rec/rec_r31_robustscanner.yml
配置文件格式如下
```
Global:
  ...
  # 添加自定义字典，如修改字典请将路径指向新字典
  character_dict_path: ./ppocr/utils/dict90.txt
  ...
  # 识别空格
  use_space_char: False

...

Train:
  dataset:
    # 数据集格式，支持LMDBDataSet以及SimpleDataSet
    name: LMDBDataSet
    # 数据集路径
    data_dir: ./train_data/
    ...

Eval:
  dataset:
    # 数据集格式，支持LMDBDataSet以及SimpleDataSet
    name: LMDBDataSet
    # 数据集路径
    data_dir: ./train_data
    ...
```
如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false
为了方便体验，本示例提供了toy_data(链接：https://pan.baidu.com/s/1IXVRqRSuGQFouAMLrgNOXA 
提取码：no4x)来简单训练。
若要训练完整数据集，请下载完整数据，并修改配置文件中的data_dir。
```
# GPU训练 支持单卡，多卡训练

#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_r31_robustscanner.yml

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r31_robustscanner.yml
```
根据配置文件中设置的的 `save_model_dir` 和 `save_epoch_step` 字段，会有以下几种参数被保存下来：
```
output/rec/
├── best_accuracy.pdopt  
├── best_accuracy.pdparams  
├── best_accuracy.states  
├── config.yml  
├── iter_epoch_3.pdopt  
├── iter_epoch_3.pdparams  
├── iter_epoch_3.states  
├── latest.pdopt  
├── latest.pdparams  
├── latest.states  
└── train.log
```
其中 best_accuracy.* 是评估集上的最优模型；iter_epoch_x.* 是以 `save_epoch_step` 为间隔保存下来的模型；latest.* 是最后一个epoch的模型。

本复现训练好的模型权重及训练日志地址为(链接：https://pan.baidu.com/s/1IXVRqRSuGQFouAMLrgNOXA 
提取码：no4x), 下载完成后，将文件夹里面的文件全部拷贝到./inference/rec_r31_robustscanner,以方便后续描述

### 4.2 模型评估

评估数据集可以通过 configs/rec/rec_r31_robustscanner.yml 修改Eval中的 data_dir 设置。
```
# GPU 评估， Global.checkpoints 为待测权重
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r31_robustscanner.yml -o Global.checkpoints=./inference/rec_r31_robustscanner/best_accuracy
```
评估日志存储在./log中
### 4.3 模型预测

使用 PaddleOCR 训练好的模型，可以通过以下脚本进行快速预测。

默认预测图片存储在配置文件的 `infer_img` 字段里，通过`Global.infer_img`进行修改。通过 `-o Global.checkpoints` 加载训练好的参数文件：
```
# 预测英文结果
python3 tools/infer_rec.py -c configs/rec/rec_r31_robustscanner.yml -o Global.pretrained_model=./inference/rec_r31_robustscanner/best_accuracy Global.load_static_weights=false Global.infer_img=./inference/rec_inference/word_1.png
```
预测图片：
![](./inference/rec_inference/word_1.png)
得到输入图像的预测结果：

```
infer_img: doc/imgs_words/en/word_1.png
        result: ('joint', 0.9853365)
```
## 5. 模型推理部署

将动态模型转为静态模型
需要注意的是，在模型部署过程中，`PositionAttentionDecoder`中需要生成position_index，这无法进行部署。因此，采用外部传入的方式。见tools/export_model.py的58-73行。

```
# -c 后面设置训练算法的yml配置文件
# -o 配置可选参数
# Global.pretrained_model 参数设置待转换的训练模型地址，不用添加文件后缀 .pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_model.py -c configs/rec/rec_r31_robustscanner.yml -o Global.pretrained_model=./inference/rec_r31_robustscanner/best_accuracy  Global.save_inference_dir=./inference/rec_r31_robustscanner
```
转换成功后，在目录下有三个文件：
```
/inference/rec_r31_robustscanner/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```
- 自定义模型推理

  ```
  python3 tools/infer/predict_rec.py --image_dir=./inference/rec_inference/word_1.png --rec_model_dir=./inference/rec_r31_robustscanner --rec_image_shape="3, 48, 48, 160" --rec_char_dict_path=./ppocr/utils/dict90.txt --use_space_char=False --rec_algorithm="RobustScanner"
  ```

## 6. 自动化测试脚本

飞桨除了基本的模型训练和预测，还提供了训推一体全流程（Training and Inference Pipeline Criterion(TIPC)）信息和测试工具，方便用户查阅每种模型的训练推理部署打通情况，并可以进行一键测试。
测试单项功能仅需两行命令，**如需测试不同模型/功能，替换配置文件即可**，命令格式如下：
```shell
# 功能：准备数据
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/prepare.sh  configs/[model_name]/[params_file_name]  [Mode]

# 功能：运行测试
# 格式：bash + 运行脚本 + 参数1: 配置文件选择 + 参数2: 模式选择
bash test_tipc/test_train_inference_python.sh configs/[model_name]/[params_file_name]  [Mode]
```

例如，测试基本训练预测功能的`lite_train_lite_infer`模式，运行：
```shell
# 准备数据
bash test_tipc/prepare.sh ./test_tipc/configs/ch_ppocr_mobile_v2.0_det/train_infer_python.txt 'lite_train_lite_infer'
# 运行测试
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/ch_ppocr_mobile_v2.0_det/train_infer_python.txt 'lite_train_lite_infer'
```
更多信息可查看[基础训练预测使用文档](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/test_tipc/docs/test_train_inference_python.md#22-%E5%8A%9F%E8%83%BD%E6%B5%8B%E8%AF%95)。

关于本复现，tipc配置文件已经给出，暂不提供数据准备和下载
test_tipc/configs/rec_r31_robustscanner，可以通过查看train_infer_python.txt的内容来了解tipc的具体流程和配置。

需要注意的是在train_infer_python.txt中有关于模型和推理图片的配置
`train_infer_img_dir:./inference/rec_inference`
`train_model:./inference/rec_r31_robustscanner/best_accuracy`


```shell
# 运行lite_train_lite_infer模式，
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/rec_r31_robustscanner/train_infer_python.txt 'lite_train_lite_infer'
```

```shell
# 运行whole_infer模式，
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/rec_r31_robustscanner/train_infer_python.txt 'whole_infer'
```

## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。
