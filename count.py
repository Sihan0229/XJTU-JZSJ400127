from subprocess import list2cmdline
import sys

from numpy import ndarray
sys.path.insert(0, './yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.models.process_image_and_extract_polygons import clean_and_fit_polygons
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

import matplotlib
matplotlib.use('Agg')


########################################
start_time = time.time()  # 记录开始时间

source_dir = 'inference/input/test3.mp4' # '0'    # 要打开的文件。
output_dir = 'inference/output/change_line_1' # 要保存到的文件夹
show_video = False   # 运行时是否显示
save_video = True   # 是否保存运行结果视频
save_text = True    # 是否保存结果数据到txt文件中，
                    # result.txt的格式是(帧序号,框序号,框到左边距离,框到顶上距离,框横长,框竖高,-1,-1,-1,-1)，
                    # number.txt的内容是统计到第几帧时每条线沿两个方向的跨线物体数
class_list = [2]    # 类别序号，在coco_classes.txt中查看（注意是序号不是行号），可以有一个或多个类别
point_idx = 0       # 方框的检测点位置(0, 1, 2, 3, 4)，看下边的图，当一个方框的检测点跨过检测线时，统计数会+1

####################################################
# 输出average_frame背景路径
# output_image_path = "/root/autodl-tmp/XJTU-JZSJ400127/inference/output/average_frame.png"

# 打开输入视频
cap = cv2.VideoCapture(source_dir)

# 获取视频的基本信息
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))       # 视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 视频高度

# 初始化一个数组用于累加所有帧
sum_frames = np.zeros((height, width, 3), dtype=np.float32)

# 遍历视频帧
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 累加帧
    sum_frames += frame.astype(np.float32)
    count += 1
# 释放视频资源
cap.release()
# 计算平均帧
average_frame = sum_frames / count

# 将平均帧转换为uint8格式（像素值范围0-255）
average_frame = np.clip(average_frame, 0, 255).astype(np.uint8)

####################################################
output_point_image_path = "/root/autodl-tmp/longest_boundaries_output.png"
all_polygons = clean_and_fit_polygons(average_frame, output_point_image_path)

####################################################
def polygons_to_lines(all_polygons, line_color=(0, 255, 255), line_thickness=5):
    """
    将四个顶点的多边形顶点转化为线段列表，按指定顺序连接线段：2 -> 1 -> 3 -> 4 -> 2。
    :param all_polygons: 包含多个多边形的顶点列表，每个多边形是一个顶点的数组。
    :param line_color: 线条的颜色，默认为黄色 (R, G, B) 格式。
    :param line_thickness: 线条的粗细，默认为 5。
    :return: 一个包含所有线段的列表，每条线段的格式为 [x1, y1, x2, y2, (R, G, B), 粗细]。
    """
    lines = []

    for polygon in all_polygons:
        # 确保只有四个顶点（可以根据需要调整此限制）
        if len(polygon) != 4:
            print("警告：当前多边形不是四个顶点的多边形，跳过该多边形。")
            continue

        # 按照顺序：2 -> 1 -> 3 -> 4 -> 2
        # 即：polygon[1] -> polygon[0] -> polygon[2] -> polygon[3] -> polygon[1]
        lines.append([polygon[1][0], polygon[1][1], polygon[0][0], polygon[0][1], line_color, line_thickness])  # 2 -> 1
        lines.append([polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1], line_color, line_thickness])  # 1 -> 3
        lines.append([polygon[2][0], polygon[2][1], polygon[3][0], polygon[3][1], line_color, line_thickness])  # 3 -> 4
        lines.append([polygon[3][0], polygon[3][1], polygon[1][0], polygon[1][1], line_color, line_thickness])  # 4 -> 2

    return lines
###########

lines = polygons_to_lines(all_polygons)
# [            # 在这里定义检测线
#     # 一条线就是一个list，内容为[x1, y1, x2, y2, (R, G, B), 线的粗细]，例如：
#     [142, 789, 1091, 544, (0,255,255), 5],
#     [1091, 544, 1715, 903, (0,255,0), 5],

# ]

total_time = time.time() - start_time  # 计算总时间
print(f"总预处理时间：{total_time:.2f}秒")
########################################
# 一些参数的定义
# x是点到左边的距离，y是点到顶上的距离
# 线的小侧是线与x轴所夹的锐角区域

# 方框检测点的序号
#    1__________________2
#    |                  |
#    |                  |
#    |      0(中心点)   |
#    |                  |
#    |__________________|
#    4                  3


#    |-------> x轴
#    |
#    |
#    V
#    y轴

########################################
# 判断点是否位于线的大侧
def big_side(line, x, y) -> bool:
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    if y1 == y2:
        if y > y1:
            return True
        elif y <= y1:
            return False

    if x1 == x2:
        if x > x1:
            return True
        elif x <= x1:
            return False

    if (x - x1)/(x2 - x1) > (y - y1)/(y2 - y1):
        return True
    else:
        return False

# 每条线添加一个list，统计大->小、小->大两个方向穿过检测线的物体数，下标为6
for line in lines:
    line.append([0,0])

########################################



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img



# 在调用detect()函数进行检测时，记得加上
# with torch.no_grad():
#     detect(args)
def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

#####################################################
    # 参数设置
    show_vid = show_video
    save_vid = save_video
    save_txt = save_text

#####################################################
    # 获取视频的信息
    a = cv2.VideoCapture(source)
    frame_num = int(a.get(7))   # 总帧数
    frame_rate = a.get(5)       # 帧速率
    frame_w = a.get(3)          # 帧宽
    frame_h = a.get(4)          # 帧高
    print(frame_num, frame_rate, frame_w, frame_h)
    a.release()

#####################################################

    # point_list统计所有点与线之间的位置关系
    point_list = []
    for i in range(len(lines)):
        point_list.append([[],[]]) # 分别统计位于大侧、小侧的点


#####################################################

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    ##################################
    # 打印使用的设备
    print(device)
    ##################################
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    #############################################
                    # 这里tlwh_bboxs是list，里边包着的也是list
                    # tlwh_bboxs的元素中的四个值分别是框到左边、顶上距离和框横长和竖高
                    # 而outputs是ndarray
                    # outputs中每一个子数组中的五个数分别是每一个框的左上角xy和右下角xy坐标和框序号
                    # x是点到左边的距离，y是点到顶上的距离
                    #############################################

                    for point in outputs:
                        # 计算检测点坐标
                        if point_idx == 0:
                            point_x = int(point[0]+point[2])/2
                            point_y = int(point[1]+point[3])/2
                        elif point_idx == 1:
                            point_x = point[0]
                            point_y = point[1]
                        elif point_idx == 2:
                            point_x = point[2]
                            point_y = point[1]
                        elif point_idx == 3:
                            point_x = point[2]
                            point_y = point[3]
                        elif point_idx == 4:
                            point_x = point[0]
                            point_y = point[3]
                        
                        # 计算检测点与每条线的位置关系
                        for line_idx, line in enumerate(lines):
                            if big_side(line, point_x, point_y):                # 点此刻位于大侧
                                if point[-1] not in point_list[line_idx][0]:    # 若不在大侧list，则加入
                                    point_list[line_idx][0].append(point[-1])
                                if point[-1] in point_list[line_idx][1]:        # 若此前位于小侧list，说明沿着小->大方向穿过了检测线
                                    line[6][1] += 1                             # 统计数+1，并从小侧list移除
                                    point_list[line_idx][1].remove(point[-1])
                            
                            else:
                                if point[-1] not in point_list[line_idx][1]:
                                    point_list[line_idx][1].append(point[-1])
                                if point[-1] in point_list[line_idx][0]:
                                    line[6][0] += 1
                                    point_list[line_idx][0].remove(point[-1])


                    #############################################

                    # Write MOT compliant results to file
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            # bbox_top = tlwh_bbox[0]
                            # bbox_left = tlwh_bbox[1]
                            bbox_left = tlwh_bbox[0]
                            bbox_top = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                # f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                #                             bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                                                        # 修改后的格式为：帧序号、框序号、框到左边距离、框到顶上距离、框横长、框竖高，原命名应该是把顶上和左边命名写反了
            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            #########################################################
            # 画线
            for line in lines:
                cv2.line(im0, (line[0], line[1]), (line[2], line[3]), line[4], line[5])   # 画布、起点坐标、终点坐标、线颜色、线粗细
            # 标注文字
            gap = int(frame_w / len(lines))
            for line_idx, line in enumerate(lines):
                cv2.putText(im0, f'dir1 = {line[6][0]}', (gap*line_idx+25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, line[4], 2)    # 画布、内容、左下角坐标、字体、字号（数字越大字越大）、字颜色、笔画粗细
                cv2.putText(im0, f'dir2 = {line[6][1]}', (gap*line_idx+25, 50), cv2.FONT_HERSHEY_COMPLEX, 1, line[4], 2)    # 画布、内容、左下角坐标、字体、字号（数字越大字越大）、字颜色、笔画粗细

            # 写入保存文件
            if save_txt:
                with open(out+'/number.txt', 'a') as f:
                    f.write(f'frame{frame_idx}:\n')
                    for line_idx, line in enumerate(lines):
                        f.write(f'line{line_idx}\tdirection1:{line[6][0]}, direction2:{line[6][1]}\n')
                    f.write('\n')



            #########################################################

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=source_dir, help='source')
    parser.add_argument('--output', type=str, default=output_dir, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', default=class_list, type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
