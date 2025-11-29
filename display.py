import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import json
import os
import torch.nn as nn
import model
import model_pro

# 使用你提供的类别列表（仅名称）
CLASSES = [
    "AnnualCrop（农作物）",
	"Forest（森林）",
	"HerbaceousVegetation（草地）",
	"Highway（公路）",
	"Industrial（工厂）",
	"Pasture（牧场）",
	"PermanentCrop（永久性作物）",
	"Residential（住宅）",
	"River（河流）",
	"SeaLake（湖海）",
]


def load_model(model_path):
	if not model_path:
		return None, "no_path"

	# 如果传入的是文件对象（Gradio File），尝试使用其 name
	path = model_path
	if hasattr(model_path, 'name'):
		path = model_path.name

	if not os.path.exists(path):
		return None, f"模型文件未找到: {path}"

	try:
		ckpt = torch.load(path, map_location='cpu')
	except Exception as e:
		return None, f"载入文件失败: {e}"

	# 提取可能的 state_dict
	state_dict = None
	if isinstance(ckpt, dict):
		if 'model_state_dict' in ckpt:
			state_dict = ckpt['model_state_dict']
		elif 'state_dict' in ckpt:
			state_dict = ckpt['state_dict']
		else:
			# 可能直接就是 state_dict
			# Heuristic: check if values are tensors
			vals = list(ckpt.values())
			if len(vals) > 0 and hasattr(vals[0], 'shape'):
				state_dict = ckpt
			else:
				# fallback: try to find a nested dict that looks like state_dict
				for v in ckpt.values():
					if isinstance(v, dict):
						state_dict = v
						break
	else:
		# Unexpected format
		return None, "checkpoint 格式未知"

	if state_dict is None:
		return None, "无法找到权重字典"

	# 决定使用哪个模型架构
	first_key = next(iter(state_dict.keys()))
	try:
		if first_key.startswith('b1') or first_key.startswith('b1.') or '.b1.' in first_key:
			# 自定义 ResNet
			net = model.ResNet(in_channels=3, num_classes=len(CLASSES))
		elif first_key.startswith('conv1') or first_key.startswith('layer1') or first_key.startswith('fc'):
			# torchvision resnet style
			net = model_pro.get_resnet50(num_classes=len(CLASSES), in_channels=3)
		else:
			# 最后尝试自定义 ResNet
			net = model.ResNet(in_channels=3, num_classes=len(CLASSES))

		# 先尝试严格加载，否则降级为 strict=False
		try:
			net.load_state_dict(state_dict)
		except Exception:
			net.load_state_dict(state_dict, strict=False)
			print("警告: 使用非严格模式加载模型权重，可能存在不匹配的键。")
		net.eval()
		return net, None
	except Exception as e:
		return None, f"模型实例化或加载失败: {e}"


def get_pth_files(search_dirs=None):
	# 在当前目录及 checkpoints/ 中查找 .pth 文件
	files = []
	search_dirs = search_dirs or ['.','checkpoints']
	for d in search_dirs:
		if not os.path.isdir(d):
			continue
		for f in os.listdir(d):
			if f.endswith('.pth'):
				files.append(os.path.join(d, f))
	# 返回文件名（不含路径）以便下拉显示
	return files


def predict(image, model_path):
	if image is None:
		return "请上传图片", {c: 0.0 for c in CLASSES}

	if not model_path:
		return "请选择模型", {c: 0.0 for c in CLASSES}

	model, err = load_model(model_path)
	if model is None:
		return f"模型加载失败: {err}", {c: 0.0 for c in CLASSES}

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	try:
		img_tensor = transform(image).unsqueeze(0)
		with torch.no_grad():
			outputs = model(img_tensor)
			probs = torch.softmax(outputs, dim=1)[0]
	except Exception as e:
		return f"预测失败: {e}", {c: 0.0 for c in CLASSES}

	# 如果模型输出类别数与 CLASSES 不匹配，报告错误
	if probs.shape[0] != len(CLASSES):
		return f"类别数不匹配: 模型输出={probs.shape[0]} vs CLASSES={len(CLASSES)}", {c: 0.0 for c in CLASSES}

	predicted = int(torch.argmax(probs).item())
	result_text = f"预测类别: {CLASSES[predicted]}"
	prob_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
	return result_text, prob_dict


# --- Gradio 界面 ---
def _list_models_for_dropdown():
	files = get_pth_files()
	# 显示相对路径或文件名
	names = [os.path.relpath(f) for f in files]
	return names


with gr.Blocks(title="地物分类") as demo:
	gr.Markdown("# 地物分类预测")

	with gr.Row():
		with gr.Column():
			model_dropdown = gr.Dropdown(label="选择模型 (.pth)", choices=_list_models_for_dropdown(), value=_list_models_for_dropdown()[0] if _list_models_for_dropdown() else None)
			image_input = gr.Image(label="上传图片", type="pil")
			submit_btn = gr.Button("预测", variant="primary")

		with gr.Column():
			result_text = gr.Textbox(label="预测结果", interactive=False)
			result_chart = gr.Label(label="各类别概率")

	submit_btn.click(
		fn=predict,
		inputs=[image_input, model_dropdown],
		outputs=[result_text, result_chart]
	)


if __name__ == "__main__":
	demo.launch(share=False,server_name="10.13.3.242",server_port=7860)

