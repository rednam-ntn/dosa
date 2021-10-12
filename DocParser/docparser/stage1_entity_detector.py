import logging.config
import os
import time
from typing import List

import cv2 as cv
import mrcnn.model as modellib
import tensorflow as tf
from docparser.utils.data_utils import DocsDataset
from docparser.utils.experiment_utils import DocparserDefaultConfig, TimeHistory

# logging.config.fileConfig(os.path.join(os.path.dirname(__file__), "logging.conf"))
logger = logging.getLogger(__name__)
tf.get_logger().setLevel("ERROR")


class EntityDetector(object):
    def __init__(self, nms_threshold=None, resize_mode="square", detection_max_instances=100):

        self.classes = DocsDataset.ALL_CLASSES

        self.dataset_num_all_classes = len(DocsDataset.ALL_CLASSES) + 1  # one extra background class

        class InferenceConfig(DocparserDefaultConfig):
            NAME = "docparser_inference"
            DETECTION_MAX_INSTANCES = detection_max_instances
            IMAGE_RESIZE_MODE = resize_mode
            if nms_threshold:
                DETECTION_NMS_THRESHOLD = nms_threshold
            NUM_CLASSES = self.dataset_num_all_classes
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.update_train_config()
        self.inference_config = InferenceConfig()

    def init_model(
        self,
        model_log_dir="logs",
        model_weights=None,
        train_mode=False,
    ):

        if train_mode:
            config = self.train_config
            mode = "training"
        else:
            config = self.inference_config
            mode = "inference"

        self.model = modellib.MaskRCNN(mode=mode, config=config, model_dir=model_log_dir)
        if model_weights is None:
            model_weights_path = os.path.join(
                os.path.dirname(__file__),
                "default_models",
                "docparser_0009.h5",
            )
        else:
            model_weights_path = model_weights

        logger.info("loading model weights from {}".format(model_weights_path))
        self.model.load_weights(model_weights_path, by_name=True)

    def load_model_weights(self, model_weights_path):
        self.model.load_weights(model_weights_path, by_name=True)

    def get_config(self):
        return self.model.config

    def predict(self, image):
        st = time.time()
        results = self.model.detect([image])
        et = time.time()

        result_dict = results[0]
        r = result_dict
        pred_bboxes = r["rois"]
        pred_class_ids = r["class_ids"]
        pred_scores = r["scores"]
        classes_with_background = ["background"] + self.classes

        orig_shape = list(image.shape)

        num_preds = len(pred_bboxes)
        prediction_list = []
        for pred_nr in range(num_preds):
            class_name = classes_with_background[pred_class_ids[pred_nr]]
            pred_bbox = pred_bboxes[pred_nr]
            pred_score = pred_scores[pred_nr]
            prediction_list.append(
                {
                    "pred_nr": pred_nr,
                    "class_name": class_name,
                    "pred_score": pred_score,
                    "bbox_orig_coords": pred_bbox,
                    "orig_img_shape": orig_shape,
                }
            )

        predictions = {"prediction_list": prediction_list, "orig_img_shape": orig_shape}

        print(f"Found: {num_preds} bboxes in: {et - st:.3f} seconds")

        return predictions

    @staticmethod
    def write_predictions_to_file(predictions, target_dir, file_name, use_original_img_coords=True):
        detections_textfile = os.path.join(target_dir, file_name)
        detections_output_lines = []
        logger.debug("saving predictions to {}".format(detections_textfile))

        if len(predictions["orig_img_shape"]) == 2:
            logger.debug("Expanding dimension for image size")
            predictions["orig_img_shape"].append(1)
        detections_output_lines.append(
            "orig_height:{};orig_width:{};orig_depth:{}".format(*predictions["orig_img_shape"])
        )
        for pred in predictions["prediction_list"]:
            pred_nr = pred["pred_nr"]
            class_name = pred["class_name"]
            pred_score = pred["pred_score"]
            if use_original_img_coords:
                pred_bbox = pred["bbox_orig_coords"]
            else:
                pred_bbox = pred["pred_bbox"]
            y1, x1, y2, x2 = pred_bbox
            pred_output_line = "{} {} {} {} {} {} {}".format(pred_nr, class_name, pred_score, x1, y1, x2, y2)
            detections_output_lines.append(pred_output_line)

        with open(detections_textfile, "w") as out_file:
            for line in detections_output_lines:
                out_file.write("{}\n".format(line))

    def save_predictions_to_file(self, predictions, target_dir, file_name, use_original_img_coords=True):
        EntityDetector.write_predictions_to_file(
            predictions,
            target_dir,
            file_name,
            use_original_img_coords=use_original_img_coords,
        )

    @staticmethod
    def visualize(
        img,
        predictions,
        output_filepath: str,
        filter_classes: List[str] = None,
        score_thresh: float = 0.5,
    ):
        for pred in predictions["prediction_list"]:
            class_name = pred["class_name"]
            pred_score = pred["pred_score"]

            if filter_classes and class_name not in filter_classes:
                continue

            if not pred_score >= score_thresh:
                continue

            pred_bbox = pred["bbox_orig_coords"]

            y1, x1, y2, x2 = pred_bbox
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(
                img,
                f"{class_name} - {pred_score:.3f}",
                (x1, y1 - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        cv.imwrite(output_filepath, img)

    def update_train_config(
        self,
        gpu_count=1,
        nms_threshold=None,
        train_rois_per_image=200,
        max_gt_instances=100,
        detection_max_instances=100,
        steps_per_epoch=2000,
        validation_steps=200,
        learning_rate=0.001,
        resize_mode="square",
        name="docparser_default",
    ):
        class TrainConfig(DocparserDefaultConfig):
            TRAIN_ROIS_PER_IMAGE = train_rois_per_image
            MAX_GT_INSTANCES = max_gt_instances
            DETECTION_MAX_INSTANCES = detection_max_instances
            STEPS_PER_EPOCH = steps_per_epoch
            VALIDATION_STEPS = validation_steps
            LEARNING_RATE = learning_rate
            GPU_COUNT = gpu_count
            IMAGE_RESIZE_MODE = resize_mode
            NAME = name
            if nms_threshold:
                DETECTION_NMS_THRESHOLD = nms_threshold
            NUM_CLASSES = self.dataset_num_all_classes

        self.train_config = TrainConfig()

    def train(
        self,
        dataset_train,
        dataset_val,
        custom_callbacks=[],
        augmentation=None,
        epochs1=20,
        epochs2=60,
        epochs3=80,
    ):

        custom_callbacks = [TimeHistory()] + custom_callbacks

        if epochs1 > 0:
            # Training - Stage 1
            logger.info("Training network heads")
            self.model.train(
                dataset_train,
                dataset_val,
                learning_rate=self.train_config.LEARNING_RATE,
                epochs=epochs1,  # 40
                layers="heads",
                augmentation=augmentation,
                custom_callbacks=custom_callbacks,
            )

        if epochs2 > 0:
            # Training - Stage 2
            logger.info("Fine tune Resnet stage 4 and up")
            self.model.train(
                dataset_train,
                dataset_val,
                learning_rate=self.train_config.LEARNING_RATE,
                epochs=epochs2,  # 120
                layers="4+",
                augmentation=augmentation,
                custom_callbacks=custom_callbacks,
            )

        if epochs3 > 0:
            # Training - Stage 3
            logger.info("Fine tune all layers")
            self.model.train(
                dataset_train,
                dataset_val,
                learning_rate=self.train_config.LEARNING_RATE / 10,
                epochs=epochs3,  # 160
                layers="all",
                augmentation=augmentation,
                custom_callbacks=custom_callbacks,
            )
