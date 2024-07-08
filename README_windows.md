
github repo: https://github.com/hsiong/detect-cow-yoloworld

# Before your start
```shell
cd code 
pip install .
```

### build-essential 

linux: sudo apt install build-essential 

https://github.com/AILab-CVC/YOLO-World/blob/master/docs/installation.md

### requirements(windows)

mmcv:2.1.0 https://mmcv.readthedocs.io/en/latest/get_started/installation.html

```
pip install path/mmcv-2.1.0-cp38-cp38-win_amd64.whl
备选: pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.2/index.html
```

cuda: 11.8

torch: 2.1.2

python: 3.8 (64bit)

+ windows: “crtdbg.h”: No such file or directory

  ```
  install Visual Studio Installer -> windows 11 sdk
  ```

+ ERROR: torch-2.1.2+cu118-cp38-cp38-win_amd64.whl is not a supported wheel on this platform.

  ```
  python 64 bit
  
  pip install weights/torch-2.1.2+cu118-cp38-cp38-win_amd64.whl
  ```

+ Torch not compiled with CUDA enabled (cuda 12.2 对应 pytorch 未发布)
  pip uninstall torch torchvision torchaudio

  ```
  pip install torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
  ```

+ unsupported Microsoft Visual Studio version! Only the versio ns between 2017 and 2022 (inclusive) are supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.

  **Visual Studio 2017 (版本 15.x)**

  **Visual Studio 2019 (版本 16.x)**

  **Visual Studio 2022 (版本 17.x)**

+ ModuleNotFoundError: No module named 'mmcv._ext'

  compile mmcv

  https://github.com/open-mmlab/mmcv/archive/refs/tags

  cd mmcv
  pip install -e .

+ C2429 "requires compiler flag /std:c++17"
  extra_compile_args['cxx'] = ['/std:c++14'] -> extra_compile_args['cxx'] = ['/std:c++17']

+ https://github.com/AILab-CVC/YOLO-World/issues/157
  What are the minimum requriements to run YOLO-World? #157

  ```
  dependencies = [
  "wheel",
  "torch==2.1.2",
  "torchvision==0.16.2",
  "transformers",
  "tokenizers",
  "numpy",
  "opencv-python",
  "supervision==0.19.0",
  "openmim",
  "mmcv-lite==2.0.1",
  "mmdet==3.3.0",
  "mmengine==0.10.4",
  "mmcv==2.1.0",
  
  'mmyolo @ git+https://github.com/onuralpszr/mmyolo.git',
  
  ]
  
  ```

### requirements-linux



### other issue

+ Incorrect path_or_model_id: '../pretrained_models/clip-vit-base-patch32-projection'. Please provide either the path to a local folder or the repo_id of a model on the Hub

  ```
  https://github.com/AILab-CVC/YOLO-World/issues/206
  
  you can comment line 18 and uncomment line 19 in your config file.
  ```

+ 'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'
  ``````
  pip install git+https://github.com/lvis-dataset/lvis-api.git
  ``````

+ No such file or directory: 'data/coco/lvis/lvis_v1_minival_inserted_image_name.json'

  ```
  https://github.com/AILab-CVC/YOLO-World/issues/308 
  
  下载后, 修改 yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.json
  
  - 删除 coco_val_dataset 的 data_root='data/coco/',
  - 修改  的 lvis_v1_minival_inserted_image_name.json 的路径
  ```

+ ./third_party/mmyolo/configs/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco.py' FileNotFoundError

  ```
  修改 yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.json 的 yolov8_x_syncbn_fast_8xb16-500e_coco.py 的路径为 yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.json 的相对路径
  ```

+ KeyError: 'YOLOWorldDetector is not in the mmyolo::model registry. Please check whether the value of `YOLOWorldDetector` is correct or it was registered as expected
  ```
  sys.path.append('yolo_world') # Adjust this path
  ```

+ 多卡推理: https://github.com/AILab-CVC/YOLO-World/issues/246

+ ImportError: Failed to import custom modules from {‘imports’: [‘yolo_world’], ‘allow_failed_imports’: False}, the current sys.path is:
  You should set PYTHONPATH to make sys.path include the directory which contains your custom module

  ```
  https://blog.csdn.net/ITdaka/article/details/138863017?spm=1001.2014.3001.5501
  
  解决这个问题的关键就是找到这个自定义的yolo_world包在哪里，并且放在对应位置。
  打开yolo_world_v2_s.py的配置文件不难发现，最上边有一个自定义的导包custom_imports = dict(imports=[‘yolo_world’],
  allow_failed_imports=False)，点击yolo_world便进入了最外层目录的yolo_world的init.py初始化文件，这个yolo_world就是我们要的自定义包
  ```

+ pycharm无法访问网络(pip huggingface...): 

  VPN节点问题

+ ImportError: Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".

  pip install git+https://github.com/lvis-dataset/lvis-api.git

### fps
YOLO-World achieves 35.4 AP with 52.0 FPS on V100
https://arxiv.org/pdf/2401.17270
https://github.com/AILab-CVC/YOLO-World/issues/177

## lable

### X-AnyLabeling  

https://github.com/CVHub520/X-AnyLabeling    一个巨好用的标注工具

+ nvcc -> CUDA Version: 11.8 , python >= 3.10

  https://pypi.org/project/onnx/#history 16.1

  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

+ cuda 11.8

  ```
  sudo apt-get purge 'nvidia-*'
  sudo apt-get purge 'cuda-*'
  sudo apt-get autoremove
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository ppa:graphics-drivers/ppa
  sudo apt-get update
  sudo ubuntu-drivers autoinstall
  sudo reboot
  ```

  ```
  sudo apt install build-essential dkms
  wget -c https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
  sudo sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit # 只想安装 CUDA Toolkit 而不安装驱动; 自动接受协议并进行安装
  
  # 添加以下内容到你的 ~/.bashrc 文件中：
  echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
  source ~/.bashrc
  
  nvcc --version
  ```

+ 
