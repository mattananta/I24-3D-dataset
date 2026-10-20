from scene import Scene

import numpy as np
import torch
import _pickle as pickle
import copy
import cv2
import os
import time
import gc

start_time = time.time()


def available_scene_ids(video_directory):
    """Return scene numbers whose directories contain at least one video."""
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    scene_ids = []
    if not os.path.isdir(video_directory):
        return scene_ids
    for entry in os.scandir(video_directory):
        if not entry.is_dir() or not entry.name.startswith("scene"):
            continue
        suffix = entry.name[len("scene"):]
        if suffix.isdigit() and any(
            video.is_file()
            and os.path.splitext(video.name)[1].lower() in video_extensions
            for video in os.scandir(entry.path)
        ):
            scene_ids.append(int(suffix))
    return sorted(scene_ids)

def create_summary(directories,out_dir):
    all_data = []
    for directory in directories:
        
        im_dir    = directory[0]
        label_dir = directory[1]
        
        if not os.path.isdir(im_dir) or not os.path.isdir(label_dir):
            continue
        ims = os.listdir(im_dir)
        labels = os.listdir(label_dir)
        if not ims:
            continue
          
        if os.path.isdir(os.path.join(im_dir,ims[0])): 
            for cam in ims:
                im_dir_new = os.path.join(im_dir,cam)
                label_dir_new = os.path.join(label_dir,cam)
                
                ims_new = os.listdir(im_dir_new)
                labels_new = os.listdir(label_dir_new)
                
                for im in ims_new:
                    label = im.split(".png")[0] + ".cpkl"
                    if label in labels_new:
                        all_data.append([os.path.join(im_dir_new,im),os.path.join(label_dir_new,label)])
            
        else:
            for im in ims:
                label = im.split(".png")[0] + ".cpkl"
                if label in labels:
                    all_data.append([os.path.join(im_dir,im),os.path.join(label_dir,label)])
    
    save_label_path = "{}/data_summary.cpkl".format(out_dir)
    with open(save_label_path,"wb") as f:
        pickle.dump(all_data,f)
    print("Saved dataset summary with {} items at {}".format(len(all_data),save_label_path))

def cache_frames(ann,output_directory, start_frame = 0,how_many = 8,append_extra = False,write_frames = True):
    os.makedirs(output_directory, exist_ok=True)
    all_data = []
    
    if not append_extra:
        # for each frame idx
        for f_idx in np.arange(start_frame,ann.last_frame):
            
            if len(ann.data[f_idx]) == 0: # past last frame
                break
            
            # for each camera
            if f_idx % how_many == 0: # only cache every 20th frame
                
                for c_idx in range(len(ann.cameras)):
                    cam_frame_annotations = []
                    
                    camera = ann.cameras[c_idx]
                    cam_name = ann.cameras[c_idx].name
                    
                    # get set of objects labeled within that camera
                    ts_data = list(ann.data[f_idx].values())
                    ts_data = list(filter(lambda x: x["camera"] == camera.name,ts_data))
                    
                    if len(ts_data) == 0:
                        continue
                    ids = [item["id"] for item in ts_data]
                    if len(ts_data) > 0:
                        if False: #--------------------------------------------------------------------------------------------------------------------------------------------- single or double box for trailers
                            boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in ts_data])
                            # convert into image space
                            im_boxes = ann.hg.state_to_im(boxes,name = camera.name)
                            
                            for i in range (len(ts_data)):
                                # for each object, append the image-space box to the ts_data annotation
                                ts_data[i]["im_box"] = im_boxes[i]
                                ts_data[i]["box"] = boxes[i]
                                ts_data[i]["frame_idx"] = f_idx
                                if "gen" not in ts_data[i].keys(): ts_data[i]["gen"] = "Manual"
                                cam_frame_annotations.append(ts_data[i])
                        
                        else:
                            # separate cab and trailer boxes
                            trailers = []
                            for obj in ts_data:
                                if obj.get("cab_length", 0) > 0:
                                    item = obj
                                    front = torch.tensor([item["x"]+item["l"]*item["direction"] - item["cab_length"]*item["direction"],item["y"],item["cab_length"],item["w"],item["h"],item["direction"]]).float()
                                    back =  torch.tensor([item["x"],item["y"],item["l"] - item["cab_length"],item["w"],item["h"],item["direction"]]).float()
                                    front_im = ann.hg.state_to_im(front.unsqueeze(0),name = camera.name)
                                    back_im =  ann.hg.state_to_im(back.unsqueeze(0),name = camera.name)
                                    
                                    obj_new = copy.deepcopy(obj)
                                    
                                    obj["box"] = front
                                    obj["im_box"] = front_im
                                    obj["frame_idx"] = f_idx
                                    
                                    obj_new["box"] = back
                                    obj_new["im_box"] = back_im
                                    obj_new["frame_idx"] = f_idx
                                    
                                    if "semi" in obj["class"]: # front = original, back = trailer_s
                                        obj_new["class"] = "trailer_sem"
                                    else:   # front = orighinal, back = trailer
                                        obj_new["class"] = "trailer"

                                    trailers.append(obj_new)
                                    
                                else:
                                    box = torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() 
                                    im_box = ann.hg.state_to_im(box.unsqueeze(0),name = camera.name)
                                    obj["box"] = box
                                    obj["im_box"] = im_box
                                    obj["frame_idx"] = f_idx

                            ts_data += trailers
                            cam_frame_annotations = ts_data
                        
                        
                        
                         
                    # generate image path
                    im_directory = "{}/im".format(output_directory)
                    if not os.path.exists(im_directory):
                            os.mkdir(im_directory)
                    scene_directory = "{}/im/scene_{}".format(output_directory,ann.scene_id)            
                    if not os.path.exists(scene_directory):
                            os.mkdir(scene_directory)
                    cam_directory   = "{}/im/scene_{}/{}".format(output_directory,ann.scene_id,cam_name)            
                    if not os.path.exists(cam_directory):
                            os.mkdir(cam_directory)
                   
                    
                                     
                    im_path = "{}/{}.png".format(cam_directory,str(f_idx).zfill(4))
                    if write_frames:  
                        if not os.path.exists(im_path):
                            frame = ann.buffer[ann.buffer_frame_idx][c_idx].copy() # second item is timestamp or dummy value
                            frame = cv2.resize(frame,(1920,1080))
                            cv2.imwrite(im_path,frame)
                              
                                
                      
                    # generate label path
                    lab_directory = "{}/label".format(output_directory)
                    if not os.path.exists(lab_directory):
                            os.mkdir(lab_directory)
                    scene_directory = "{}/label/scene_{}".format(output_directory,ann.scene_id)            
                    if not os.path.exists(scene_directory):
                            os.mkdir(scene_directory)
                    cam_directory   = "{}/label/scene_{}/{}".format(output_directory,ann.scene_id,cam_name)            
                    if not os.path.exists(cam_directory):
                            os.mkdir(cam_directory)
                            
                    label_directory = "{}/label/scene_{}/{}".format(output_directory,ann.scene_id,cam_name)
                    if not os.path.exists(label_directory):
                        os.mkdir(label_directory)
                    lab_path = "{}/{}.cpkl".format(label_directory,str(f_idx).zfill(4))
                    
                    with open(lab_path,"wb") as f:
                        pickle.dump(cam_frame_annotations,f)
                    
                    # after all objects have been  added
                    all_data.append([im_path,lab_path])
        
       
                    
            
        
            # advance cameras
            if write_frames:
                ann.next()
            if f_idx % 100 == 0:
                t_est = time.time() - start_time
                print("Cached scene {} frame {}, {} fps cache rate".format(ann.scene_id,f_idx,f_idx/t_est))
    
            # if f_idx > 10:
            #     break
        
    
    else: # add additional bonus frames
        bonus_path = "/home/worklab/Documents/datasets/more_3D_frames"
        
        ims = os.listdir(bonus_path + "/im")
        labels = os.listdir(bonus_path + "/label_export")
        
        for im_path in ims:
            label_path = im_path.split(".png")[0] + ".cpkl"
            if label_path in labels:
                all_data.append([os.path.join(bonus_path, "im", im_path),os.path.join(bonus_path, "label_export", label_path)])

     
    save_label_path = "{}/data_summary.cpkl".format(output_directory)
    with open(save_label_path,"wb") as f:
        pickle.dump(all_data,f)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache I24 video frames and annotations.")
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get("I24_DATASET_DIR", "/data/I24"),
        help="Directory containing video/ and data/ (default: /data/I24)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("I24_OUTPUT_DIR", "/output"),
        help="Directory for cached images, labels, and data_summary.cpkl (default: /output)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Video decode device, e.g. cuda:0; use cpu to disable GPU decoding",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    video_dir = os.path.join(dataset_dir, "video")
    data_dir = os.path.join(dataset_dir, "data")
    out_dir = args.output_dir

    scene_ids = available_scene_ids(video_dir)
    if not scene_ids:
        raise FileNotFoundError("No scene folders containing videos found in {}".format(video_dir))

    for scene_id in scene_ids:
        ann = Scene(video_dir, data_dir, scene_id=scene_id, start_frame=0, device=args.device)
        ann.correct_curve()
        # Keep the in-memory video buffer bounded for CPU decoding, especially
        # when source videos are 4K and the GPU decoder is unavailable.
        ann.buffer_lim = 10
        cache_frames(ann, output_directory=out_dir, start_frame=0)
        del ann
        gc.collect()

    summary_directories = [
        (os.path.join(out_dir, "im", "scene_{}".format(scene_id)),
         os.path.join(out_dir, "label", "scene_{}".format(scene_id)))
        for scene_id in scene_ids
    ]
    create_summary(summary_directories, out_dir)
