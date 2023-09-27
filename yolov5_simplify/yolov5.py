import cv2
import torch

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors


class YOLOv5(object):

    def __init__(self,
                 weights='models/yolov5m.pt',  # model.pt path(s)
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 view_img=True
                 ):

        self.model = DetectMultiBackend(weights, device=device)

        self.imgsz = check_img_size(imgsz, s=self.model.stride)  # check image size

        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.view = view_img

    def detect(self, dataset, webcam, half):
        self.model.model.half() if half else self.model.model.float()

        # Data input
        bs = len(dataset)  # batch_size
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=half)  # warmup

        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = self.model(im, augment=False, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)

            for i, det in enumerate(pred):  # per image
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                annotator = Annotator(im0, line_width=3, example=str(self.model.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.view:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{self.model.names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                im0 = annotator.result()
                if self.view:
                    cv2.imshow(str(p), im0)
                    if webcam:
                        cv2.waitKey(1)  # 1 millisecond
                    else:
                        cv2.waitKey(0)  # wait for check


