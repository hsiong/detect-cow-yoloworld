import mmcv
from mmcv.ops import nms
import torch

def test_cuda():
	print(f"PyTorch version: {torch.__version__}")
	print(f"CUDA available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		print(f"CUDA version: {torch.version.cuda}")
		print(f"Number of CUDA devices: {torch.cuda.device_count()}")
		print(f"Current CUDA device: {torch.cuda.current_device()}")
		print(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
	else:
		print("CUDA is not available. Please check your installation.")
		
def test_mmcv():
	
	boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32).cuda()
	scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32).cuda()
	
	dets, keep = nms(boxes, scores, 0.5)
	print(dets)
	print(keep)