****# 对比音频-视觉掩码自编码器

- [介绍](#介绍)
- [引用](#引用)
- [存储库包含什么？](#存储库包含什么)
- [CAV-MAE模型](#cav-mae模型)
- [数据准备](#数据准备)
    - [步骤1. 从视频中提取音频轨道和图像帧](#步骤1-从视频中提取音频轨道和图像帧)
    - [步骤2. 为数据集构建标签集和json文件](#步骤2-为数据集构建标签集和json文件)
- [CAV-MAE预训练](#cav-mae预训练)
    - [调整vision-MAE检查点](#调整vision-mae检查点)
    - [构建虚拟环境并安装软件包](#构建虚拟环境并安装软件包)
    - [运行CAV-MAE预训练](#运行cav-mae预训练)
    - [在AS-2M预训练CAV-MAE基础上进行其他预训练](#在as-2m预训练cav-mae基础上进行其他预训练)
- [音频-视觉事件分类](#音频-视觉事件分类)
  - [AudioSet](#audioset)
  - [VGGSound](#vggsound)
- [检索](#检索)
- [修复](#修复)
- [预训练模型](#预训练模型)
    - [CAV-MAE预训练模型（主要）](#cav-mae预训练模型主要)
    - [CAV-MAE预训练模型（消融研究）](#cav-mae预训练模型消融研究)
    - [CAV-MAE预训练+微调模型](#cav-mae预训练微调模型)
    - [AudioSet和VGGSound数据列表](#audioset和vggsound数据列表)

## 介绍

本存储库包含了 **对比音频-视觉掩码自编码器(CAV-MAE)** 的官方实现（基于PyTorch），该模型在ICLR 2023论文[《Contrastive Audio-Visual Masked Autoencoder》](https://openreview.net/forum?id=QPtMRyk5rb)（Yuan Gong, Andrew Rouditchenko, Alexander H. Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, James Glass）中提出。

CAV-MAE结合了两个重要的自监督学习框架：**对比学习**和**掩码数据建模**，以学习一个联合和协调的音频-视觉表示。我们的实验证明，对比音频-视觉对应学习目标不仅使模型能够执行音频-视觉检索任务，而且还帮助模型学习更好的联合表示。CAV-MAE在VGGSound上达到了新的SOTA准确率65.9％，在音频-视觉事件分类任务上与之前最佳的监督预训练模型在AudioSet上具有可比性。

**评论：**我们在[OpenReview](https://openreview.net/forum?id=QPtMRyk5rb)上收到了该论文的审稿意见，并对审稿人的宝贵意见表示感谢。

## 引用

如果您发现此存储库有用，请引用我们的论文。

```
@inproceedings{gong2023contrastive,
    title={Contrastive Audio-Visual Masked Autoencoder},
    author={Yuan Gong and Andrew Rouditchenko and Alexander H. Liu and David Harwath and Leonid Karlinsky and Hilde Kuehne and James R. Glass},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=QPtMRyk5rb}
}
```

## 存储库包含什么？

这个存储库包含了重现我们的实验和进一步适应CAV-MAE预训练到您的任务所需的所有内容。

具体来说，

- `CAVMAE`和`CAVMAEFT`模型脚本位于[`src/models/cav-mae.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/models/cav_mae.py)中。
- 数据预处理脚本位于[`src/preprocess/`](https://github.com/YuanGongND/cav-mae/tree/master/src/preprocess)中。
- 训练管道位于[`src/run_cavmae_pretrain.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/run_cavmae_pretrain.py)和[`src/traintest_cavmae.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/traintest_cavmae.py)中（用于CAV-MAE自监督预训练）；以及[`src/run_cavmae_ft.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/run_cavmae_ft.py)和[`src/traintest_ft.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/traintest_ft.py)（用于分类微调）。
- [`egs/{audioset,vggsound}`](https://github.com/YuanGongND/cav-mae/tree/master/egs)中包含了AudioSet和VGGSound的训练脚本和日志。
- [`src/retrieval.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/retrieval.py)中包含了检索实验脚本。
- [`src/inpaint.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/inpaint.py)中包含了修复实验脚本。
- [`src/extract_audio_representation.py`](https://github.com/YuanGongND/cav-mae/blob/master/src/extract_audio_representation.py)中包含了音频表示提取脚本。
- 预训练模型和数据列表的详细列表见[[这里]](#pretrained-models)。

## CAV-MAE模型

提出的CAV-MAE模型位于`src/models/cav-mae.py`中，其中包括两个模型：`CAVMAE`（用于预训练，具有解码器）和`CAVMAEFT`（用于微调，没有解码器）。代码是自包含且有注释的，了解详细信息的最佳方法是阅读代码。

`CAVMAE`和`CAVMAEFT`的输入应该是一对（音频，图像）。音频的形状应为[批次大小，帧数，梅尔频谱数量]，例如[批次大小，1000，128]；图像的形状应为[批次大小，通道数，高度，宽度]，例如[批次大小，3，224，224]。

`CAVMAE`模型使用其默认参数是我们在论文中使用的模型。但是，我们还实现了其他一些功能，例如，
- 音频和图像的掩码比率可以不相同。
- 我们在论文中使用了无结构掩码，但也实现了结构化掩码策略（例如，时间，频率，时间-频率），您可以通过更改`forward`函数中的`mask_mode`来更改它。
- 我们在论文中使用了单向对比损失，但也实现了双向对比损失。
- 您可以通过更改`forward`函数中的`mae_loss_weight`和`contrast_loss_weight`来跳过对比损失或MAE损失。

## 数据准备

如果您要使用我们的预训练模型，您当然可以使用自己的训练管道，但如果要使用我们的预训练模型，输入数据仍然需要与我们的数据相同。

否则，如果您希望使用我们的管道（包括预训练、分类、检索和修复中的任何一个）。您将需要以与我们相同的格式准备数据。

#### 步骤1. 从视频中提取音频轨道和图像帧

假设您有一组视频（例如，以`.mp4`格式），您首先需要离线提取音频轨道和图像帧，并将它们保存在磁盘上。在运行时提取通常会大大增加数据加载开销。在`src/preprocess/extract_{audio,video_frame}.py`中，我们包含了我们的提取代码。这两个脚本都很简单，您将需要准备一个包含视频路径列表的`csv`文件（有关示例，请参见`src/preprocess/sample_video_extract_list.csv`），以及您要保存输出的`target_fold`（一个单一路径）。

默认情况下，我们假设`video_id`是视频的名称（即没有扩展名和路径的视频名称），例如，视频`/path/test12345.mp4`的`video_id`是`test12345`。图像帧将保存在`target_fold/frame_{0-9}/video_id.jpg`，音频轨道将保存在`target_fold/video_id.wav`中。

我们提供了一个最小示例。视频和列表在此存储库中提供，您只需运行即可生成帧和音频：
```python
cd cav-mae/src/preprocess
# extract video frames
python extract_video_frame.py -input_file_list sample_video_extract_list.csv -target_fold ./sample_frames
# extract audio tracks
python extract_audio.py  -input_file_list sample_video_extract_list.csv -target_fold ./sample_audio
```

#### 步骤2. 为数据集构建标签集和json文件

您将需要两个文件：

- 一个标签csv文件列出了所有标签（请参见`src/preprocess/sample_datafiles/class_labels_indices_as.csv`作为示例）。
- 一个json文件，每个样本有四个键（请参见`src/preprocess/sample_datafiles/sample_json_as.json`作为示例）：
  - `wav`：先前步骤中提取的音频轨道的绝对路径，例如`/data/sls/audioset/--4gqARaEJE.flac`。
  - `video_id`：视频ID（即视频文件名没有扩展名），例如，视频`--4gqARaEJE.mp4`的`video_id`是`--4gqARaEJE`。
  - `video_path`：您在之前步骤中使用的`target_fold`，例如`/data/sls/audioset/`。我们的管道将从`video_path/frame_{0-9}/video_id.jpg`加载，而不是`video_path/video_id.jpg`，所以**确保`video_path/frame_{0-9}/video_id.jpg`包含您的图像帧。**
  - `labels`：此样本的所有标签，如果有多个，请使用`,`分隔，必须与标签csv文件一致。
  - 您可以查看我们使用`src/preprocess/create_json_as.py`如何自动生成此类json文件。

为了使这更容易，我们在这里分享了我们的AudioSet和VGGSound数据文件，您可以基于我们的文件使用/修改。共享的数据文件还显示了我们在实验中使用的确切样本ID，这对于重现目的可能很有帮助。

## CAV-MAE预训练

#### 调整vision-MAE检查点

正如我们在论文中提到的，使用ImageNet预训练检查点初始化CAV-MAE会提高性能。然而，原始的MAE检查点是针对单模态的，而CAV-MAE是多模态的。我们使用一个脚本来适应原始的MAE检查点以用于CAV-MAE。您不需要自己执行此操作，因为我们的管道会处理它。但是，如果您有兴趣，可以查看`src/adapt_vmae_weights.py`是如何执行的。

#### 构建虚拟环境并安装软件包

在运行任何实验之前，请为此项目安装必要的软件包：

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

#### 运行CAV-MAE预训练

预训练脚本是`egs/audioset/run_cavmae_pretrain_scale++.sh`，只需通过`./run_cavmae_pretrain_scale++.sh`运行它，它将调用`src/run_cavmae_pretrain.py`，它将调用`src/traintest_cavmae.py`。

注意，`egs/audioset/run_cavmae_pretrain_scale++.sh`应复现`CAV-MAE-Scale++`（不在论文中），它需要更大的GPU（4 X 48GB GPU）；对于较小的GPU（4 X {24,12}GB GPU），我们还包括`egs/audioset/run_cavmae_pretrain_{scale+,base).sh`，应复现`CAV-MAE-BASE`。

您需要将以下内容更改为之前步骤中准备的自己的数据文件：
- `tr_data`：训练数据json文件。
- `te_data`：测试数据json文件。
- `label-csv`：标签csv文件。

注意：即使不使用，自监督预训练仍然需要标签。这是因为我们使用单个数据加载器。您可以为真-无标签数据提供虚拟标签。

预训练脚本中的一些值得注意的参数：
- `masking_ratio=0.75`：音频和视觉的掩码比率。
- `contrast_loss_weight=0.01`：对比损失的权重，可以为0。
- `mae_loss_weight=1.0`：MAE损失的权重，可以为0。
- `tr_pos=False`：使用可训练的位置嵌入，应设置为`False`。
- `norm_pix_loss=True`：使用像素归一化的MAE。修复模型使用`False`，其他情况使用`True`。
- `pretrain_path=None`：基于另一个预训练检查点预训练。

#### 在AS-2M预训练CAV-MAE基础上进行其他预训练

一个典型的情况是您有一个新的数据集（例如，VGGSound），您当然可以在新的数据集上微调AudioSet预训练的CAV-MAE。但是在本文中，我们发现首先在您的数据集上进行另一轮的自监督预训练，然后进行监督微调效果更好。`egs/vggsound/run_cavmae_pretrain_as.sh`是如何执行此操作的很好的示例。

## 音频-视觉事件分类

### AudioSet

AudioSet分类脚本位于`egs/audioset/`中

- `run_cavmae_ft_full.sh`在完整的AudioSet-2M上进行微调，同时使用音频和视觉数据。应复现51.2 mAP。
- `run_cavmae_ft_bal.sh`在平衡的AudioSet-20K上进行微调，同时使用音频和视觉数据。应复现42.2（在论文中是42.0 mAP）。
- `run_cavmae_ft_bal_audioonly.sh`在平衡的AudioSet-20K上进行音频微调。应复现38.3 mAP（在论文中是37.7 mAP）。

您将需要将以下内容更改为之前步骤中准备的自己的数据文件：
- `tr_data`：训练数据json文件。
- `te_data`：测试数据json文件。
- `label-csv`：标签csv文件。

一些值得注意的参数是：

- `ftmode=multimodal`：使用多模态数据还是不使用，将`audioonly`设置为微调仅使用音频模型。
- `pretrain_path`：预训练的CAV-MAE检查点路径。
- `freeze_base`：冻结CAV-MAE并仅训练新初始化的MLP层，即线性探测。端到端微调应该是False。
- `head_lr`：新初始化的MLP层参数/预训练的CAV-MAE参数之比。始终设置为> 1。
- `lrscheduler_start`，`lrscheduler_decay`和`lrscheduler_step`：学习率调度程序，从`lrscheduler_start`轮开始，学习率将每隔`lrscheduler_step`轮衰减`lrscheduler_decay`倍。
- `wa`，`wa_start`，`wa_end`：模型权重平均化参数。如果`wa=True`，将在评估之前对`wa_start`到`wa_end`之间的检查点进行加权平均。`wa_end`应小于总轮数。
- `dataset_mean`和`dataset_std`：音频频谱数据集级均值和标准差，对于相似的数据集（例如，我们在AudioSet和VGGSound上使用相同的值）可以使用我们的值。
- `target_length`：输入音频长度（以帧计），例如，10s音频的输入音频长度为1000。
- `freqm`和`timem`：音频的specaug参数。`freqm`应该大约为48，`timem`应该是`target_length`的20％。

您将获得比论文中稍微好一些的结果，因为默认情况下，我们基于我们最新的GPU预训练了`CAV-MAE-Scale++`模型。但是如果您希望，您也可以复现论文中的（较低）数字，因为我们还提供了`CAV-MAE-Scale+`和`CAV-MAE-Base`的检查点。请注意，这些模型的大小相同，但是使用不同的批量大小进行训练，换句话说，微调成本是相同的，因此除非用于消融研究目的，否则建议使用我们最强大的预训练CAV-MAE模型。

一些训练日志也提供在`egs/audioset/training_logs`上以帮助复现。

### VGGSound

与AudioSet非常相似，脚本位于`egs/vggsound/run_cavmae_ft.sh`。这应该复现65.8%的准确率（没有额外的VGGSound预训练）和65.9%的准确率（具有额外的VGGSound预训练）。

训练日志也提供在`egs/vggsound/training_logs`上以帮助复现。

## 检索

音频-视觉检索脚本位于`src/retrieval.py`，代码是自包含的。您只需要一个CAV-MAE检查点和一个数据集。

## 修复

音频-视觉修复脚本位于`src/inpaint.py`，代码是自包含的。您只需要一个没有像素归一化的CAV-MAE检查点和一个数据集。

## 预训练模型

#### CAV-MAE预训练模型（主要）

我们提供了以下CAV-MAE模型。请使用以下脚本加载带有解码器的CAV-MAE模型：

```python3
import torch,timm
from models import CAVMAE
assert timm.__version__ == '0.4.5' # it is important to have right version of timm
model_path = 'the path to your model location'
# CAV-MAE model with decoder
audio_model = CAVMAE(audio_length=1024, \ # all models trained with 10s audio
                     modality_specific_depth=11, \ # all models trained with 11 modality-specific layers and 1 shared layer
                     norm_pix_loss=True, tr_pos=False) # most models are trained with pixel normalization and non-trainabe positional embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
print(miss, unexpected) # check if all weights are correctly loaded
```

|      模型名称     | 批次大小 |   Lambda_c   |      掩码比率     |                     用途                     |
|:-------------------:|:----------:|:--------:|:----------------:|:---------------------------------------------:|
|   **[CAV-MAE-Scale++（推荐）](https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1)** <br />[[中国镜像链接]](https://share.weiyun.com/vGfP1YZt) |     256    |   0.01   | 75%无结构化 | 除修复之外的所有目的都建议使用。 |
|    [CAV-MAE-Scale+](https://www.dropbox.com/s/xu8bfie6hz86oev/audio_model.25.pth?dl=1)   |     108    |   0.01   | 75%无结构化 |            复现精确的论文结果。     |
|     [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)    |     48     |   0.01   | 75%无结构化 |            复现精确的论文结果。     |
| [CAV-MAE-Base-NoNorm](https://www.dropbox.com/s/arrg7cb3e4hhjwu/cav-mae-base-nonorm.pth?dl=1) |     48     |   0.01   | 75%无结构化 |                   修复                   |

#### CAV-MAE预训练模型（消融研究）

除了上述模型之外，我们还发布了以下用于消融研究的模型。在每个表格中，这些模型在超参数上具有相同的设置，但与感兴趣的超参数不同。

*掩码比率*

|    模型名称   | 批次大小 |   Lambda_c   |      掩码比率     |
|:---------------:|:----------:|:--------:|:----------------:|
| [CAV-MAE-Base-50](https://www.dropbox.com/s/dgorer0ybdbnvgf/50.pth?dl=1) |     48     |   0.01   | 50%无结构化 |
| [CAV-MAE-Base-65](https://www.dropbox.com/s/xmuyksqch6l6g87/65.pth?dl=1) |     48     |   0.01   | 65%无结构化 |
|   [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)  |     48     |   0.01   | 75%无结构化 |
| [CAV-MAE-Base-85](https://www.dropbox.com/s/y3o9zggpecmrbnu/85.pth?dl=1) |     48     |   0.01   | 85%无结构化 |

*音频掩码方法*

|         模型名称        | 批次大小 |   Lambda_c   |        掩码比率       |
|:-------------------------:|:----------:|:--------:|:----------------:|
| [CAV-MAE-Base-Unstructured](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1) |     48     |   0.01   |      75%无结构化      |
|     [CAV-MAE-Base-TF](https://www.dropbox.com/s/madv6ynuy5113zh/time_freq_75.pth?dl=1)      |     48     |   0.01   | 75%时域频域掩码 |
|     [CAV-MAE-Base-TF-50](https://www.dropbox.com/s/nd7qlagn8je6zjn/time_freq_50.pth?dl=1)    |     48     |   0.01   | 50%时域频域掩码 |
|       [CAV-MAE-Base-T](https://www.dropbox.com/s/hfehd7m379ehr0y/time_75.pth?dl=1)      |     48     |   0.01   |      75%时域掩码      |
|       [CAV-MAE-Base-F](https://www.dropbox.com/s/ad4fhzt6d3xre5p/freq_75.pth?dl=1)      |     48     |   0.01   |    75%频域掩码   |

*对比损失权重*

|      模型名称     | 批次大小 |   Lambda_c   |      掩码比率     |
|:-------------------:|:----------:|:--------:|:----------------:|
| [CAV-MAE-Base-C0.001](https://www.dropbox.com/s/4j4qyiyjcmbtc6u/cav-mae-base-c0.001.pth?dl=1) |     48     |   0.001  | 75%无结构化 |
|     [CAV-MAE-Base](https://www.dropbox.com/s/wxrjgr86gdhc5k8/cav-mae-base.pth?dl=1)    |     48     |   0.01   | 75%无结构化 |
|  [CAV-MAE-Base-C0.1](https://www.dropbox.com/s/wf54ver9rs9fl1b/cav-mae-base-c0.1.pth?dl=1)  |     48     |   0.1   | 75%无结构化 |

*对称（双向）对比损失*

|      模型名称     | 批次大小 |   Lambda_c   | 对比损失 |
|:-------------------:|:----------:|:--------:|:----------------:|
|    [CAV-MAE-Scale+](https://www.dropbox.com/s/mvhmg7eda410phr/single_direction.pth?dl=1)   |     120    |   0.01   | 单向 |
| [CAV-MAE-Symc-Scale+](https://www.dropbox.com/s/ute6ydkw4hdv7rn/symc.pth?dl=1) |     120    |   0.01   |  双向  |

#### CAV-MAE预训练+微调模型

使用以下脚本加载没有解码器的CAV-MAE模型（通常是微调模型）：

```python3
import torch,timm
from models import CAVMAEFT
assert timm.__version__ == '0.4.5' # it is important to have right version of timm
model_path = 'the path to your model location'
n_class = 527 # 527 for audioset finetuned models, 309 for vggsound finetuned models
# CAV-MAE model without decoder
audio_model = models.CAVMAEFT(label_dim=n_class, \
                              modality_specific_depth=11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mdl_weight = torch.load(model_path, map_location=device)
audio_model = torch.nn.DataParallel(audio_model) # it is important to convert the model to dataparallel object as all weights are saved in dataparallel format (i.e., in module.xxx)
miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
print(miss, unexpected) # check if all weights are correctly loaded, if you are loading a model with decoder, you will see decoders are unexpected and missed are newly initialized classification heads
```

*多模态音频-视觉模型*

| 预训练模型 |  预训练数据 | 微调数据 |  性能  | 下载链接 |
|:----------------:|:----------------:|:--------------:|:-------------:|:--------:|
|  CAV-MAE-Scale+  |    AudioSet-2M   |   AudioSet-2M  |  51.2 mAP   |  [link](https://www.dropbox.com/s/hejj49zdgyyfsuh/as-full-51.2.pth?dl=1)  |
|  CAV-MAE-Scale+  |    AudioSet-2M   |  AudioSet-20K  |  42.0 mAP   |  [link](https://www.dropbox.com/s/9x6y642so3zve5y/as-20k-42.0.pth?dl=1)  |
|  CAV-MAE-Scale+  |    AudioSet-2M   |    VGGSound    |65.5 accuracy |  [link](https://www.dropbox.com/s/f4wrbxv2unewss9/vgg_65.5.pth?dl=1)  |
|  CAV-MAE-Scale+  | AS-2M + VGGSound |    VGGSound    |65.9 accuracy |  [link](https://www.dropbox.com/s/q66s3mu526mj2x6/vgg_66.0.pth?dl=1)  |

*仅音频或仅视觉模型*

本文的一个结论是多模态预训练有助于单模态性能。我们在下面发布了使用多模态数据预训练的模型，但在仅输入一个模态的推理中进行微调时，这些模型也有效。

| 预训练模型 |      预训练数据      |    微调数据    |  性能  | 下载链接 |
|:----------------:|:-------------------------:|:--------------------:|:-------------:|:--------:|
|  CAV-MAE-Scale+  | AudioSet-2M（多模态） |  AudioSet-2M（音频） |  46.6 mAP   |  [link](https://www.dropbox.com/s/itfw7p0ueq7z9og/as_46.6.pth?dl=1)  |
|  CAV-MAE-Scale+  | AudioSet-2M（多模态） | AudioSet-20K（音频） |  37.7 mAP   |  [link](https://www.dropbox.com/s/pariabyh1iyayda/as_37.7.pth?dl=1)  |
|  CAV-MAE-Scale++ <br> (请注意是++)  | AudioSet-2M（多模态） | AudioSet-20K（视觉） | 20.0 mAP |  [link](https://www.dropbox.com/s/9ngkq9ygwqecxz5/as_20.0.pth?dl=1)  |
|  CAV-MAE-Scale+  | AS-2M + VGGSound（多模态） |   VGGSound（音频） | 59.8 accuracy |  [link](https://www.dropbox.com/s/l4rj0sgpnt08bp2/vgg_59.8.pth?dl=1)  |

#### AudioSet和VGGSound数据列表

我们还发布了以下AudioSet和VGGSound数据列表。这些数据文件可用于1）将其用作数据准备的示例；2）检查我们在论文中使用的确切样本ID以进行复现。
由于版权问题，我们无法提供这两个数据集的原始数据。

*AudioSet*

|          文件名         |                              内容                             |
|:-------------------------:|:----------------------------------------------------------------:|
|   [AudioSet标签CSV文件](https://www.dropbox.com/s/z3ko8bv9b7738t1/class_labels_indices.csv?dl=1)      |                     AudioSet的标签集                    |
|   [AudioSet-2M Json文件](https://www.dropbox.com/s/18hoeq92juqsg2g/audioset_2m_cleaned.json?dl=1)   |   我们在本文中使用的完整的AudioSet训练集样本信息   |
|   [AudioSet-2M样本权重列表](https://www.dropbox.com/s/y3omts