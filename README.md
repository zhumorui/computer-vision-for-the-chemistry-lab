# computer-vision-for-the-chemistry-lab

This project is based on the vessel instance segmentation, which is implemented by [@sagieppel](https://github.com/sagieppel).

```bash
python main.py
```

## Resource links

Videos Resources link: https://drive.google.com/drive/folders/1xhWR0Rzq6qwpHGN3Rm18-Rby7yZWiOq-?usp=sharing

Weight files link: https://drive.google.com/drive/folders/1SBjdqPskKwYyABW4xGnT_k6p7i7hskcn?usp=sharing

## Put weight files in right location:

"utils/Semantic/logs/1000000_Semantic_withCOCO_AllSets.torch"

"utils/InstanceVessel/logs/Vessel_Coco_610000_Trained_on_All_Sets.torch"

## Configuration

default mode: GPU mode # mode can be modified in "utils/Vessel_detect.py"

## Function Description

## 功能描述(中文版)

计算机视觉辅助实验自动化框架用于对实验现象过程进行离线或在线智能分析. 支持离线视频与在线RTSP流推理, 实现了透明玻璃容器检测、分液检测、主要颜色分析、颜色变化检测等功能. 

main.py为demo脚本, 其流程包括视频流的获取、视频流id的定义、视频分析对象的创建、分析结果的返回、实时显示、结束分析. 所有的子算法(可选)都基于主算法(透明玻璃容器mask)提取ROI区域实现, 如果未检测到玻璃容器, 算法将在对应的等待时间后再次进行检测, 直到检测到玻璃容器为止, 子算法才能继续被触发.

主文件夹包括utils(包括一系列函数和脚本工具)、Video_Resources(可选,离线或测试时使用)、output(程序运行时自动生成)等文件夹. 主文件夹包括main.py(demo脚本), Exp_v3.py(类文件). 

脚本对应函数与参数的细节在代码对应的注释区域.

## 安装说明(中文版)

1. 下载对应文件

    权重文件下载链接:

    Weight files link: https://drive.google.com/drive/folders/1SBjdqPskKwYyABW4xGnT_k6p7i7hskcn?usp=sharing

    测试用视频下载链接:

    https://drive.google.com/drive/folders/1xhWR0Rzq6qwpHGN3Rm18-Rby7yZWiOq-?usp=sharing

2. 将权重文件放到以下的路径

    "utils/Semantic/logs/1000000_Semantic_withCOCO_AllSets.torch"

    "utils/InstanceVessel/logs/Vessel_Coco_610000_Trained_on_All_Sets.torch"

3. 按照需要自行修改main.py中创建类时的默认设置, 然后运行main.py函数
4. 分液检测与颜色变化检测的结果会在output文件夹中找到, 文件夹和文件会根据当地时间戳与对应的视频流id命名, 同时当检测到分液过程时, 终端也会有对应的输出.
