import glob
import os
import os.path as osp
import sys
import time
from typing import List

import cv2
import torch
from PIL import Image, ImageDraw
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmdet.utils import get_test_pipeline_cfg


def inference(model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
	image = cv2.imread(image)
	image = image[:, :, [2, 1, 0]]
	data_info = dict(img=image, img_id=0, texts=texts)
	data_info = test_pipeline(data_info)
	data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
					  data_samples=[data_info['data_samples']])
	with torch.no_grad():
		output = model.test_step(data_batch)[0]
	pred_instances = output.pred_instances
	# score thresholding
	pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
	# max detections
	if len(pred_instances.scores) > max_dets:
		indices = pred_instances.scores.float().topk(max_dets)[1]
		pred_instances = pred_instances[indices]
	
	pred_instances = pred_instances.cpu().numpy()
	boxes = pred_instances['bboxes']
	labels = pred_instances['labels']
	scores = pred_instances['scores']
	label_texts = [texts[x][0] for x in labels]
	return boxes, labels, label_texts, scores
	
def list_img(directory_path):
	files = os.listdir(directory_path)
	file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif',
					   '.JPG', '.JPEG', '.PNG', '.BMP', '.GIF']  # 常见图片格式
	images = []
	for file_extension in file_extensions:
		search_pattern = os.path.join(directory_path, f"*{file_extension}")  # 搜索模式
		files = glob.glob(search_pattern)  # 获取符合模式的文件
		absolute_files = [os.path.abspath(file) for file in files]  # 转换为绝对路径
		images.extend(absolute_files)  # 添加到结果列表中
	return images


class DetectedObject:
	def __init__(self, object_id, bbox, label, class_name, score):
		self.object_id = object_id
		self.bbox = bbox
		self.label = label
		self.class_name = class_name
		self.score = score
	
	def __repr__(self):
		return (f"DetectedObject(object_id={self.object_id}, bbox={self.bbox}, "
				f"label={self.label}, class_name={self.class_name}, score={self.score})")


def draw_img(detected_objects: List[DetectedObject], image_path):
	image = Image.open(image_path)
	draw = ImageDraw.Draw(image)
	
	for obj in detected_objects:
		bbox = obj.bbox
		draw.rectangle(bbox, outline="red", width=2)
		draw.text((bbox[0], bbox[1]), f"{obj.class_name} {obj.score:.2f}", fill="red")
	
	# 生成保存路径
	original_dir = os.path.dirname(image_path)
	parent_dir = os.path.basename(original_dir)
	detect_dir_name = f"{parent_dir}_detect"
	new_dir = os.path.join(original_dir, "..", detect_dir_name)
	os.makedirs(new_dir, exist_ok=True)
	
	original_filename = os.path.basename(image_path)
	filename, ext = os.path.splitext(original_filename)
	new_filename = f"{filename}_detect{ext}"
	save_path = os.path.join(new_dir, new_filename)
	
	#image.show()  # 显示图像
	# 保存图像
	image.save(save_path)
	print(f"Image saved to {save_path}")

def detect(model, image_absolute_path, texts, test_pipeline):
	start_time = time.time()
	
	print(f"starting to detect: {image_absolute_path}")
	results = inference(model, image_absolute_path, texts, test_pipeline)
	print(f"detecting results: {results}")
	
	# format_str = [
	# 	f"obj-{idx}: {box}, label-{lbl}, class-{lbl_text}, score-{score}"
	# 	for idx, (box, lbl, lbl_text, score) in enumerate(zip(*results))
	# ]
	# for q in format_str:
	# 	print(q)
	detect_list: List[DetectedObject] = []
	for idx, (box, lbl, lbl_text, score) in enumerate(zip(*results)):
		detect_list.append(DetectedObject(idx, box, lbl, lbl_text, score))
	
	draw_img(detect_list, image_absolute_path)
	
	end_time = time.time()
	
	elapsed_time = end_time - start_time
	print(f"Elapsed time: {elapsed_time} seconds")

def test_detect_cow():
	# 设置PYTHONPATH环境变量
	
	config_file = "config/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py"
	checkpoint = "weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth"
	
	print(f'start test...')
	cfg = Config.fromfile(config_file)
	cfg.work_dir = osp.join('./work_dirs')
	print(f'get cfg...')
	# init model
	cfg.load_from = checkpoint
	model = init_detector(cfg, checkpoint=checkpoint, device='cuda:0')
	print(f'get model...')
	test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
	test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
	test_pipeline = Compose(test_pipeline_cfg)
	print(f'get test_pipeline...')
	
	texts = [['cow']]
	
	directory_path = '../detect-cow/captured_frames'
	files = list_img(directory_path)
	for file in files:
		detect(model, file, texts, test_pipeline) 


