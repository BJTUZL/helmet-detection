import numpy as np
import onnxruntime as ort
import cv2
import torch
import torchvision

# 加载 ONNX 模型
model_path = 'C:\\Users\lenovo\Desktop\helmet-detection\Smart_Construction-master\weights\\best.onnx'
session = ort.InferenceSession(model_path)

# 加载图片并预处理
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()  # 保存原始图片以便后续绘制
    image = cv2.resize(image, (640, 640))  # YOLOv5 通常使用 640x640 大小
    image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    image = np.transpose(image, (2, 0, 1))  # 转换为 CXY 格式
    image = image[np.newaxis, :]  # 增加批次维度
    return image, original_image

# 进行推理
def infer(image_path):
    input_tensor, original_image = preprocess_image(image_path)
    print(input_tensor.shape)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 执行推理
    result = session.run([output_name], {input_name: input_tensor})
    return result, original_image

def xywh2xyxy(x):
    """Convert (center_x, center_y, width, height) to (x1, y1, x2, y2) format."""
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def box_iou(box1, box2):
    """Calculate the IoU of two boxes."""
    # box1: (N, 4), box2: (M, 4)
    inter = (torch.min(box1[:, None, 2], box2[:, 2]) - torch.max(box1[:, None, 0], box2[:, 0])).clamp(0) * \
            (torch.min(box1[:, None, 3], box2[:, 3]) - torch.max(box1[:, None, 1], box2[:, 1])).clamp(0)
    return inter / (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])[:, None] + \
           (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]) - inter

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    print("prediction:")
    print(prediction.shape)
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]

    return output

# 保存结果到图片
def save_result(image, output, output_path='output_image.jpg'):
    print("检测结果:", output)
    for detections in output:
        if detections is not None:
            for detection in detections:
                x_min, y_min, x_max, y_max, conf, class_id = detection
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                # 绘制边界框
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(image, f'Class: {int(class_id)}, Conf: {conf:.2f}', 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存处理后的图片
    cv2.imwrite(output_path, image)
    print(f"结果图片已保存为: {output_path}")

# 主程序
input_image_path = "C:\\Users\lenovo\Desktop\helmet-detection\Smart_Construction-master\data\VOC2028\JPEGImages\\000012.jpg"
output, original_image = infer(input_image_path)

# 应用非极大值抑制
nms_output = non_max_suppression(output[0], conf_thres=0.5, iou_thres=0.6)
print(nms_output)
# 保存结果
save_result(original_image, nms_output)