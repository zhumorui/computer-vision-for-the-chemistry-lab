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

The Computer Vision Assisted Experimental Automation Framework is designed for intelligent analysis of experimental phenomena processes, either offline or online. It supports offline video and online RTSP stream inference, and features functionalities such as transparent glass container detection, liquid separation detection, primary color analysis, and color change detection.

The main.py serves as a demo script. Its workflow includes video stream acquisition, video stream ID definition, creation of video analysis objects, return of analysis results, real-time display, and analysis termination. All sub-algorithms (optional) are based on the main algorithm (transparent glass container mask) to extract the ROI area. If no glass container is detected, the algorithm will retry after a specified waiting time until a glass container is found, after which the sub-algorithms can be triggered.

The main directory includes folders such as utils (which contains a series of functions and script tools), Video_Resources (optional, used for offline or testing), and output (automatically generated during program execution). The main directory contains main.py (demo script) and Exp_v3.py (class file).

Details of the script corresponding functions and parameters can be found in the respective comment sections of the code.

## Installation Instruction

1. Download the respective files:

    Weight files download link:

    https://drive.google.com/drive/folders/1SBjdqPskKwYyABW4xGnT_k6p7i7hskcn?usp=sharing

    Test video download link:

    https://drive.google.com/drive/folders/1xhWR0Rzq6qwpHGN3Rm18-Rby7yZWiOq-?usp=sharing

2. Place the weight files in the following paths:

    "utils/Semantic/logs/1000000_Semantic_withCOCO_AllSets.torch"

    "utils/InstanceVessel/logs/Vessel_Coco_610000_Trained_on_All_Sets.torch"

3. Modify the default settings in main.py as needed when creating the class, and then run the main.py function.
4. Results for liquid separation detection and color change detection can be found in the output folder. The folders and files will be named based on the local timestamp and the corresponding video stream ID. Additionally, when a liquid separation process is detected, there will be corresponding outputs in the terminal.
