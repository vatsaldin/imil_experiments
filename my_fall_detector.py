# ===============================================================================================================#
# Copyright 2024 Infosys Ltd.                                                                                    #
# Use of this source code is governed by Apache License Version 2.0 that can be found in the LICENSE file or at  #
# http://www.apache.org/licenses/                                                                                #
# ===============================================================================================================#

import base64
import io
import time
from milapi.utils import get_mtp, get_datetime_utc
from modelloader.base_model_loader import BaseModelLoader
import torch


class CustomFallDetectionModelLoader(BaseModelLoader):
    def __init__(self, config, model_name):
        super().__init__(config, model_name)
        #model_path = None if config[model_name]['model_path'][0] == '' \
        #    else json.loads(config[model_name]['model_path'])
        
        
        from ultralytics import YOLO
        from PIL import Image
        # self.model_obj = YOLO(model_path if model_path is None else model_path[0])
        self.model_obj = YOLO("yolov8s.pt")
        self.Image = Image
        self.torch = torch
        
        device = config[model_name]['device']
        self.logger = config[model_name].get('logger')

        if device != None or device == "":
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.logger:
            self.logger.debug("device : " + str(self.device))
            self.logger.debug("VideoSearch Clip model loaded successfully")
    
    def did_person_fall(self, x1, y1, x2, y2):
        height = y2 - y1
        width = x2 - x1
        threshold  = height - width

        if threshold < 0:
            return True        
        else:
            return False



    def predict(self, base64_image, prompt, confidence_threshold=0.5):
        st1 = time.time()
        print("in predict method")
        image = self.Image.open(io.BytesIO(base64.b64decode(base64_image)))
        image = image.convert('RGB') if image.mode != 'RGB' else image
        
        results = self.model_obj(image)  # Run inference

        # Get detection results
        detections = results[0]  # Get the first (and only) set of detections

        # Process results (example of extracting boxes and classes)
        boxes = detections.boxes.data  # Get boxes (xyxy, conf, cls)
        detected_classes = []
        confidences = []

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            class_id = int(cls)
            class_name = self.model_obj.names[class_id]  # Get class name from model's class names

            if class_name == 'person':
                print(f"Information that Person fell is----> {self.did_person_fall(x1, y1, x2, y2)}")

            detected_classes.append(class_name)
            confidences.append(float(conf))
        
        print("detected_classes", detected_classes)
        

    def predict_request(self, req_data):
        output = []
        print("in predict_request method")
        # Extraction
        Tid = req_data["Tid"]
        DeviceId = req_data["Did"]
        Fid = req_data["Fid"]
        Cs = req_data["C_threshold"]
        Base_64 = req_data["Base_64"]
        Per = req_data["Per"]
        mtp = req_data["Mtp"]
        Ts_ntp = req_data["Ts_ntp"]
        Ts = req_data["Ts"]
        Inf_ver = req_data["Inf_ver"]
        Msg_ver = req_data["Msg_ver"]
        Model = req_data["Model"]
        Ad = req_data["Ad"]
        Lfp = req_data["Lfp"]
        Ltsize = req_data["Ltsize"]
        Ffp = req_data["Ffp"]
        prompt = req_data["Prompt"]

        start_time = get_datetime_utc()
        predicted_fs_list = self.predict(Base_64, prompt, Cs)
        end_time = get_datetime_utc()
        mtp = get_mtp(mtp, start_time, end_time, Model)

        output.append({"Tid": Tid, "Did": DeviceId, "Fid": Fid, "Fs": predicted_fs_list, "Mtp": mtp,
                       "Ts": Ts, "Ts_ntp": Ts_ntp, "Msg_ver": Msg_ver, "Inf_ver": Inf_ver,
                       "Obase_64": [], "Img_url": [],
                       "Rc": "200", "Rm": "Success", "Ad": Ad, "Lfp": Lfp, "Ffp": Ffp, "Ltsize": Ltsize})
        return output[0]

        
