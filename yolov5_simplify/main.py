from pathlib import Path

from torch.backends import cudnn

from utils.datasets import LoadStreams, LoadImages
from yolov5 import YOLOv5

IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes

if __name__ == '__main__':
    source = '0'  # webcam
    # source = 'dir/test.jpg'

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

    # Dataloader
    if webcam:
        # view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source)
        bs = 1  # batch_size

    detect_model = YOLOv5()

    detect_model.detect(dataset, webcam, half=True)
