# Non-static-Objects-Selection-for-SLAM
An official code for IEEE ACCESS paper "Coarse Semantic-based Keypoints Selection for Robust Mapping in Dynamic Environments"
## Installation

### CenterNet
Please refer to README.md in CenterNet for installation instructions.

### ORB_SLAM
Please refer to README.md in ORB_SLAM for installation instructions.

## Usage

This is the offline-mode of our proposed method in this paper, the on-line one will be released in the future.

1. Download the models (We use [ctdet_coco_dla_2x](https://drive.google.com/open?id=1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT) for detection in this paper) 
from the [Model zoo](readme/MODEL_ZOO.md) and put them in `CenterNet_ROOT/models/`.

2. For object detection on images, run:

~~~
python src/demo_im.py ctdet --dataset_path /path/to/image_folder/ --load_model ../models/ctdet_coco_dla_2x.pth --output_path /path/to/output_folder/ 
~~~

The results of the object detection are saved to the file result.csv which is the semantic_file input of the modified SLAM system.

result.csv: [frame_id, category_id, x1, y1, x2, y2]

3. Run the modified SLAM system:

~~~
python associate.py /path/to/TUM/datasets/rgb.txt /path/to/TUM/datasets/depth.txt -> /path/to/TUM/datasets/associations.txt
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.bin Examples/RGB-D/TUMX.yaml /path/to/TUM/datasets /path/to/TUM/datasets/associations.txt /path/to/TUM/datasets/semantic_result/results.csv
~~~

## Benchmark Evaluation

We provide two tools for the evaluation of the trajectory estimation.

evo: [https://github.com/MichaelGrupp/evo]

rgbd_benchmark_tools: [https://github.com/jbriales/rgbd_benchmark_tools]
