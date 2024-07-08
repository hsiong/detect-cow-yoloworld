
# install guide
+ [windows: README_windows.md](README_windows.md)
+ [ubuntu: README_ubuntu.md](README_ubuntu.md)

# License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

# Acknowledgements

This project includes software from [YOLO-World](https://github.com/AILab-CVC/YOLO-World) by AILab-CVC, licensed under the GNU General Public License v3.0.

# Citation

If you use this project in your research, please cite the following paper:

```bibtex
@inproceedings{Cheng2024YOLOWorld,
  title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
  author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
  booktitle={Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

# fps
YOLO-World achieves 35.4 AP with 52.0 FPS on V100
https://arxiv.org/pdf/2401.17270
https://github.com/AILab-CVC/YOLO-World/issues/177

# lable

## X-AnyLabeling  

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
  
  >  添加以下内容到你的 ~/.bashrc 文件中：
  echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
  source ~/.bashrc
  
  nvcc --version
  ```

+ 
