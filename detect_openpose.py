import argparse
from math import degrees
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
from openpose.demo import run_img, infer_fast ,sexual
from openpose.modules.pose import Pose, track_poses
from openpose.models.with_mobilenet import PoseEstimationWithMobileNet
from openpose.FHRCNN.RBFNET_train import FHRCNN, LinearClassifier
from openpose.modules.load_state import load_state

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def putback_y(y,point,top,scale):
    point = y+(point-top)*scale
    return point
def putback_x(x,point,left,scale):
    point = x+(point-left)*scale
    return point

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    imgnum = 1
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
    f=open('momo/data_0.08.csv','w')
    f.close()
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
            plot = []
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
                            # resize_img2 = cv2.resize(crop_img,(new_w2,new_h2), interpolation=cv2.INTER_AREA)
                            # resize_img3 = cv2.resize(crop_img2,(min_side,min_side), interpolation=cv2.INTER_AREA)
                            # resize_img4 = cv2.resize(crop_img2,(min_side2,min_side2), interpolation=cv2.INTER_AREA)

                            if new_w % 2 != 0 and new_h % 2 == 0:
                                top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2,(min_side-new_w)/2 + 1,(min_side-new_w)/2
                            elif new_w % 2 == 0 and new_h % 2 != 0:
                                top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2,(min_side-new_w)/2 ,(min_side-new_w)/2
                            elif new_w % 2 == 0 and new_h % 2 == 0:
                                top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2,(min_side-new_w)/2 ,(min_side-new_w)/2
                            else :
                                top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2,(min_side-new_w)/2 + 1,(min_side-new_w)/2
                            pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value = [128,128,128])

                                                                                                     
                            
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                            net = PoseEstimationWithMobileNet()
                            net2 = LinearClassifier().cuda()
                            net3 = FHRCNN().cuda()
                            
                            
                            checkpoint = torch.load('./openpose/default_checkpoints/checkpoint_iter_17000.pth')
                            load_state(net, checkpoint)

                            checkpoint2 = torch.load('./openpose/FHRCNN/DNN/model_2999.pth')
                            checkpoint3 = torch.load('./openpose/FHRCNN/model_FHRCNN/model_2999.pth')

                            net2.load_state_dict(checkpoint2)
                            net3.load_state_dict(checkpoint3)

                            text,x_2, y_2, x_0, y_0, p1min_x, p1min_y, p1max_x, p1max_y, rate, degree = run_img(net, net2, pad_img, min_side, opt.device,imgnum)  
                            # text,x_2, y_2, x_0, y_0, p1min_x, p1min_y, p1max_x, p1max_y = run_img(net, net3, pad_img, min_side, opt.device,imgnum) 
                            
                            # cv2.circle(pad_img, (int(x_2), int(y_2)), 3, [0, 0, 255], -1)
                            # cv2.circle(pad_img, (int(x_0), int(y_0)), 3, [0, 0, 255], -1)                   
                            # cv2.circle(pad_img, (int(p1min_x), int(p1min_y)), 3, [0, 0, 255], -1)
                            # cv2.circle(pad_img, (int(p1max_x), int(p1max_y)), 3, [0, 0, 255], -1)                    
                            # cv2.line(pad_img, (int(x_2), int(y_2)), (int(x_0), int(y_0)), Pose.color, 1)
                            # cv2.line(pad_img, (int(p1min_x), int(p1min_y)), (int(p1max_x), int(p1max_y)), Pose.color, 1)
                            # cv2.line(pad_img, (int(x_2), int(y_2)), (int(p1min_x), int(p1min_y)), Pose.color, 1)
                            # cv2.line(pad_img, (int(x_2), int(y_2)), (int(p1max_x), int(p1max_y)), Pose.color, 1)
                            # cv2.putText(pad_img,'{}'.format(text), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255)) 
                            # cv2.imwrite("./20220729/"+str(imgnum)+".jpg",pad_img)                           
                            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)                         
                            # cv2.imshow('test', pad_img)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            imgnum = imgnum +1

                            x_2 = putback_x(x,x_2,left,scale)
                            x_0 = putback_x(x,x_0,left,scale)
                            p1min_x = putback_x(x,p1min_x,left,scale)
                            p1max_x = putback_x(x,p1max_x,left,scale)
                            y_2 = putback_y(y,y_2,top,scale)
                            y_0 = putback_y(y,y_0,top,scale)
                            p1min_y = putback_y(y,p1min_y,top,scale)
                            p1max_y = putback_y(y,p1max_y,top,scale)
                            
                            place_x = [x_2, x_0, p1min_x, p1max_x]
                            place_y = [y_2, y_0, p1min_y, p1max_y]
                            place_x.sort()
                            place_y.sort() 
                        
                            w1 =  pow((pow((p1min_x-p1max_x),2)+pow((p1min_y-p1max_y),2)),0.5)
                            h1 =  pow((pow((x_2-x_0),2)+pow((y_2-y_0),2)),0.5)
                            # print(w1,h1)
                            # if text != 'error':
                            if text != 'error' and w1 > 22 and h1 > 22:
                                f=open('momo/data_0.08.csv','a')
                                f.write(str(rate)+','+str(degree)+','+str(text)+'\n')
                                f.close()
                                x_text = torch.tensor([[rate, degree]]).cuda()
                                x_text = x_text.to(torch.float32) 
                                tep = net2(x_text)
                                output = tep.argmax()
                                text = output
                                point = [text,x_2, y_2, x_0, y_0, p1min_x, p1min_y, p1max_x, p1max_y]
                                plot.append(point)
                                # print(w1,h1)      
                                # cv2.circle(im0, (int(x_2), int(y_2)), 3, [0, 0, 255], -1)
                                # cv2.circle(im0, (int(x_0), int(y_0)), 3, [0, 0, 255], -1)                   
                                # cv2.circle(im0, (int(p1min_x), int(p1min_y)), 3, [0, 0, 255], -1)
                                # cv2.circle(im0, (int(p1max_x), int(p1max_y)), 3, [0, 0, 255], -1)                    
                                # cv2.line(im0, (int(x_2), int(y_2)), (int(x_0), int(y_0)), Pose.color, 1)
                                # cv2.line(im0, (int(p1min_x), int(p1min_y)), (int(p1max_x), int(p1max_y)), Pose.color, 1)
                                # cv2.line(im0, (int(x_2), int(y_2)), (int(p1min_x), int(p1min_y)), Pose.color, 1)
                                # cv2.line(im0, (int(x_2), int(y_2)), (int(p1max_x), int(p1max_y)), Pose.color, 1)
                                # # print(rate,degree)



                                # cv2.putText(im0,'{}'.format(text), (int(x_0)+5, int(y_0)+5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 180)) 

                                # # cv2.imwrite("./20220708/"+str(imgnum)+".jpg",im0)                                                    

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            for i in plot:
                cv2.circle(im0, (int(i[1]), int(i[2])), 3, [0, 0, 255], -1)
                cv2.circle(im0, (int(i[3]), int(i[4])), 3, [0, 0, 255], -1)                   
                cv2.circle(im0, (int(i[5]), int(i[6])), 3, [0, 0, 255], -1)
                cv2.circle(im0, (int(i[7]), int(i[8])), 3, [0, 0, 255], -1)                    
                cv2.line(im0, (int(i[1]), int(i[2])), (int(i[3]), int(i[4])), Pose.color, 1)
                cv2.line(im0, (int(i[5]), int(i[6])), (int(i[7]), int(i[8])), Pose.color, 1)
                cv2.line(im0, (int(i[1]), int(i[2])), (int(i[5]), int(i[6])), Pose.color, 1)
                cv2.line(im0, (int(i[1]), int(i[2])), (int(i[7]), int(i[8])), Pose.color, 1)
                cv2.putText(im0,'{}'.format(i[0]), (int(i[3])+5, int(i[4])+5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 180)) 
              
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
        
        # cv2.namedWindow("test", cv2.WINDOW_NORMAL)                         
        # cv2.imshow('test', im0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='momo/best_epoch250_20211005.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='momo/dataset', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='momo/output_test', help='output folder')  # output folder
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
