'''
You must install Grounded-SAM-2 model from github.
link: https://github.com/idea-research/grounded-sam-2
After downloading and installing all files from this github, copy and paste this code under ~/Grounded-SAM-2/
Edit or confirm lines 30, 32, 77, 78, 181, 182 then run this code.
(python3 Grounded-SAM-2/01_door_segmentation_GSAMbased.py)
'''



import copy
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import threading
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import time

# time.sleep(20000)

class MyDetector:
    def __init__(
        self,
        grounding_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_ckpt_path="/home/jb/workspace/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt", ## find the directory where you downloaded this file.
        device="cuda",
        prompt_text="door.", # wall. window. elevator. glass. pillar. column.", ## object class
    ):
        self.device = device
        self.prompt_text = prompt_text

        # GroundingDINO
        print(">>> Load processor")
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        print(">>> Load dino_model")
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)

        # SAM2
        print(">>> Build sam model")
        sam_model = build_sam2(sam2_model_cfg, sam2_ckpt_path, device=device)
        print(">>> Build sam predictor")
        self.sam_predictor = SAM2ImagePredictor(sam_model)

        # Video predictor (tracking module)
        print(">>> Build video predictor")
        self.video_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_ckpt_path)
        print(">>> Init state")
        self.inference_state = self.video_predictor.init_state()
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=device)
        self.inference_state["video_height"] = None
        self.inference_state["video_width"] = None

        self.last_mask_dict = MaskDictionaryModel()
        self.objects_count = 0
        print(">>> MyDetector __init__ successed")

    def detect(self, image_np: np.ndarray):
        img_pil = Image.fromarray(image_np)

        # GroundingDINO detection
        inputs = self.processor(images=img_pil, text=self.prompt_text, return_tensors="pt").to(self.device)
        print("inputs.device:", inputs.pixel_values.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        print("DINO outputs:", outputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.45, ## box threshold
            text_threshold=0.45, ## text threshold
            target_sizes=[img_pil.size[::-1]],
        )
        print("Results after post_process:", results)
        boxes = results[0]["boxes"]
        labels = results[0]["labels"]
        print(boxes)
        print(labels)
        if boxes.shape[0] == 0:
            return []

        # SAM2 segmentation
        self.sam_predictor.set_image(image_np)
        print("SAM2 set_image done")

        masks, scores, logits = self.sam_predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )
        print("SAM2 predict done")
        # === 🟢 Resize masks to original image shape ===
        H_img, W_img = image_np.shape[:2]
        resized_masks = []

        for i in range(masks.shape[0]):
            mask = masks[i]

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = np.squeeze(mask, axis=0)

            print(f"mask.shape before resize: {mask.shape}")

            mask = mask.astype(np.uint8) * 255

            H_img, W_img = image_np.shape[:2]
            resized = cv2.resize(mask, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
            resized = (resized > 0).astype(np.uint8)
            resized_masks.append(resized)

        resized_masks = np.stack(resized_masks, axis=0)

        # Build MaskDictionaryModel
        mask_dict = MaskDictionaryModel()
        mask_dict.add_new_frame_annotation(
            mask_list=torch.tensor(resized_masks).to(self.device),
            box_list=boxes,
            label_list=labels,
        )

        # Update masks (assign IDs)
        self.objects_count = mask_dict.update_masks(
            tracking_annotation_dict=self.last_mask_dict,
            iou_threshold=0.3,
            objects_count=self.objects_count,
        )
        self.last_mask_dict = copy.deepcopy(mask_dict)

        results_list = []


        for obj_id, obj_info in self.last_mask_dict.labels.items():
            obj_info.update_box()

            # mask = self.last_mask_dict.masks[obj_id].cpu().numpy()
            self.mask_index_per_obj_id = {obj_id: i for i, obj_id in enumerate(mask_dict.labels.keys())}
            mask = resized_masks[self.mask_index_per_obj_id[obj_id]]

            ys, xs = np.where(mask == 1)
            if len(xs) == 0 or len(ys) == 0:
                continue

            b_neg1 = xs + ys
            b_pos1 = ys - xs

            idx_b_neg1_max = np.argmax(b_neg1)
            idx_b_neg1_min = np.argmin(b_neg1)
            idx_b_pos1_max = np.argmax(b_pos1)
            idx_b_pos1_min = np.argmin(b_pos1)

            corner_neg1_max = [float(xs[idx_b_neg1_max]), float(ys[idx_b_neg1_max])]
            corner_neg1_min = [float(xs[idx_b_neg1_min]), float(ys[idx_b_neg1_min])]
            corner_pos1_max = [float(xs[idx_b_pos1_max]), float(ys[idx_b_pos1_max])]
            corner_pos1_min = [float(xs[idx_b_pos1_min]), float(ys[idx_b_pos1_min])]

            corners = [
                corner_neg1_min,
                corner_pos1_min,  
                corner_neg1_max, 
                corner_pos1_max, 
            ]

            results_list.append({
                "id": int(obj_id),
                "corners": corners
            })


        return results_list


detector = MyDetector()
import cv2, os, json


imageDIR = "/home/jb/workspace/360video/videofiles/FWH_4thfloor_JH_07222025/" ## original images' directory (000000.jpg, 000001.jpg, ...)
jsonDIR = os.path.join("/home/jb/workspace/360video/graph_codes/GSAM_result/00_threshold_4545/","FWH_4thfloor_JH_07222025") ## where the detection and segmentation results will be stored
os.makedirs(jsonDIR, exist_ok=True)

imgs = os.listdir(imageDIR)
imgs.sort()
for imgfile in imgs:

    imgDIR = os.path.join(imageDIR,imgfile)
    img = cv2.imread(imgDIR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect(img_rgb)

    with open(os.path.join(jsonDIR, imgfile.replace(".jpg",".json")), "w") as f:
        json.dump(results, f)
    print("JSON Completed:", os.path.join(jsonDIR, imgfile.replace(".jpg",".json")))

