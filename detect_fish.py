import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    imgnum = 22680
    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference      img = cv2.imread(img)
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Write results
                for *xyxy, conf, cls in det:
                    if cls == 1 or cls == 2:
                        if 1:  # Write to file
                            abcd = []
                            abcd= torch.tensor(xyxy).tolist()
                            min_side = 160
                            min_side2 = 320

                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            x,y,w,h=abcd
                            # print(x)
                            # print(y)
                            # print(w)
                            # print(h)
                            x=int(x)
                            y=int(y)
                            x1=int(w)
                            y1=int(h)

                            crop_img = im0[y:y1,x:x1]
                            
                            # cv2.imwrite('momo/20220714_ori/'+str(imgnum)+'.jpg',crop_img)
                            w=x1-x
                            h=y1-y
                            scale = max(w,h)/float(min_side)                            
                            scale2 = max(w,h)/float(min_side2)                            
                            new_w,new_h = int(w/scale), int(h/scale)         
                            new_w2,new_h2 = int(w/scale2), int(h/scale2)         
                            # print("scale",scale)   
                            # print("\n"+"wh"+"\n",w,h,new_w,new_h)

                            crop_img2 = im0[y:(y+max(w,h)),x:(x+max(w,h))]


                            resize_img = cv2.resize(crop_img,(new_w,new_h), interpolation=cv2.INTER_AREA)
                            resize_img2 = cv2.resize(crop_img,(new_w2,new_h2), interpolation=cv2.INTER_AREA)
                            resize_img3 = cv2.resize(crop_img2,(min_side,min_side), interpolation=cv2.INTER_AREA)
                            resize_img4 = cv2.resize(crop_img2,(min_side2,min_side2), interpolation=cv2.INTER_AREA)

                            # cv2.imwrite('momo/20220714_160/'+str(imgnum)+'.jpg',resize_img3)
                            # cv2.imwrite('momo/20220714_320/'+str(imgnum)+'.jpg',resize_img4)

                            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)                         
                            # cv2.imshow('test', resize_img4)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            
                            # cv2.imshow("im0",im0)
                            # cv2.imshow("cropped", crop_img)
                            if new_w % 2 != 0 and new_h % 2 == 0:
                                top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2,(min_side-new_w)/2 + 1,(min_side-new_w)/2
                            elif new_w % 2 == 0 and new_h % 2 != 0:
                                top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2,(min_side-new_w)/2 ,(min_side-new_w)/2
                            elif new_w % 2 == 0 and new_h % 2 == 0:
                                top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2,(min_side-new_w)/2 ,(min_side-new_w)/2
                            else :
                                top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2,(min_side-new_w)/2 + 1,(min_side-new_w)/2
                            pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value = [128,128,128])

                            if new_w2 % 2 != 0 and new_h2 % 2 == 0:
                                top2, bottom2, left2, right2 = (min_side2-new_h2)/2, (min_side2-new_h2)/2,(min_side2-new_w2)/2 + 1,(min_side2-new_w2)/2
                            elif new_w2 % 2 == 0 and new_h2 % 2 != 0:
                                top2, bottom2, left2, right2 = (min_side2-new_h2)/2 + 1, (min_side2-new_h2)/2,(min_side2-new_w2)/2 ,(min_side2-new_w2)/2
                            elif new_w2 % 2 == 0 and new_h2 % 2 == 0:
                                top2, bottom2, left2, right2 = (min_side2-new_h2)/2, (min_side2-new_h2)/2,(min_side2-new_w2)/2 ,(min_side2-new_w2)/2
                            else :
                                top2, bottom2, left2, right2 = (min_side2-new_h2)/2 + 1, (min_side2-new_h2)/2,(min_side2-new_w2)/2 + 1,(min_side2-new_w2)/2                        
                            # print("*******************",top,bottom,left,right)
                            pad_img2 = cv2.copyMakeBorder(resize_img2, int(top2), int(bottom2), int(left2), int(right2), cv2.BORDER_CONSTANT, value = [128,128,128])
                            
                            
                            # crop_img = im0[y1:h1,x1:w1]
                            # crop_img = im0[int((y-0.05)*480):int((xywh[1])*480)+int((xywh[3]+0.05)*480), int((xywh[0]-0.05)*640):int(xywh[0]*640)+int((xywh[2]+0.05)*640)]
                            # cv2.imshow("im0",im0)
                            # cv2.imshow("padded", pad_img)
                            # cv2.waitKey(1000000)
                            # crop_img = cv2.resize(crop_img,(160,160), interpolation=cv2.INTER_AREA)
                            # cv2.imwrite('momo/20220714_160_pad/'+str(imgnum)+'.jpg',pad_img)
                            # cv2.imwrite('momo/20220714_320_pad/'+str(imgnum)+'.jpg',pad_img2)
                            
                            # cv2.destroyWINDOW()
                            # print(xywh[0])
                            imgnum = imgnum +1
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='momo/best_epoch250_20211005.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='momo/dataset', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='momo/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.9, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='momo/yolor_p6_fish.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='momo/fish.names', help='*.cfg path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
