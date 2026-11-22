"""Object detector wrapper (ultralytics).

mode "yolo"       : COCO-pretrained YOLO (80 classes) - "what is around me, where?"
mode "yoloworld"  : open-vocabulary YOLO-World; call set_vocab([...]) with any text
                    labels (keys, wallet, spectacles ...) to detect non-COCO things.
"""
from typing import List, Optional

import numpy as np

from .schema import Det


class Detector:
    def __init__(self, weights="yolov8n.pt", conf=0.35, max_det=20, mode="yolo",
                 device=None):
        from ultralytics import YOLO, YOLOWorld
        self.mode = mode
        self.conf = conf
        self.max_det = max_det
        self.device = device
        self.model = YOLOWorld(weights) if mode == "yoloworld" else YOLO(weights)

    def set_vocab(self, names: List[str]):
        if self.mode != "yoloworld":
            raise RuntimeError("set_vocab needs mode='yoloworld'")
        self.model.set_classes(list(names))

    def detect(self, bgr: np.ndarray, classes: Optional[List[str]] = None) -> List[Det]:
        res = self.model.predict(bgr, conf=self.conf, max_det=self.max_det,
                                 device=self.device, verbose=False,
                                 agnostic_nms=True)[0]       # one box per object
        out = []
        names = res.names
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, conf, cls):
            label = names[int(k)]
            if classes and label not in classes:
                continue
            out.append(Det(tuple(float(v) for v in b), label, float(c)))
        return out
