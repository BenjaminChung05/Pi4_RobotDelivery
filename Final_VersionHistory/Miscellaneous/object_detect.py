from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class ObjectDetectorConfig:
    class_file: str = "/home/pi/Desktop/Object_Detection_Files/coco.names"
    config_path: str = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weights_path: str = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
    input_size: tuple[int, int] = (320, 320)
    input_scale: float = 1.0 / 127.5
    input_mean: tuple[float, float, float] = (127.5, 127.5, 127.5)
    swap_rb: bool = True
    conf_threshold: float = 0.45
    nms_threshold: float = 0.20


@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


class OpenCVDnnObjectDetector:
    def __init__(self, config: ObjectDetectorConfig | None = None) -> None:
        self.config = config or ObjectDetectorConfig()
        self.enabled = False
        self.reason = ""
        self.class_names: list[str] = []
        self.net = None

        class_path = Path(self.config.class_file)
        config_path = Path(self.config.config_path)
        weights_path = Path(self.config.weights_path)

        if not class_path.exists():
            self.reason = f"Missing class file: {class_path}"
            return
        if not config_path.exists():
            self.reason = f"Missing config file: {config_path}"
            return
        if not weights_path.exists():
            self.reason = f"Missing weights file: {weights_path}"
            return

        self.class_names = class_path.read_text().splitlines()
        self.net = cv2.dnn_DetectionModel(str(weights_path), str(config_path))
        self.net.setInputSize(*self.config.input_size)
        self.net.setInputScale(self.config.input_scale)
        self.net.setInputMean(self.config.input_mean)
        self.net.setInputSwapRB(self.config.swap_rb)
        self.enabled = True

    def detect(self, frame_bgr, allowed_classes: set[str] | None = None) -> list[DetectedObject]:
        if not self.enabled or self.net is None:
            return []

        class_ids, confidences, boxes = self.net.detect(
            frame_bgr,
            confThreshold=self.config.conf_threshold,
            nmsThreshold=self.config.nms_threshold,
        )

        detections: list[DetectedObject] = []
        if len(class_ids) == 0:
            return detections

        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            index = int(class_id) - 1
            if index < 0 or index >= len(self.class_names):
                continue
            class_name = self.class_names[index].strip()
            if allowed_classes is not None and class_name not in allowed_classes:
                continue
            detections.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                )
            )
        return detections


def draw_object_debug(frame_bgr, detections: list[DetectedObject]) -> None:
    for detection in detections:
        x, y, w, h = detection.bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 140, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 140, 0),
            2,
        )