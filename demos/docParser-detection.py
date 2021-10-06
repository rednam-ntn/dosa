from datetime import datetime as dt
from pathlib import Path

import cv2
from docparser import stage1_entity_detector
from tqdm import tqdm


def init_paths():
    data_dir = Path("./data")
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = data_dir / "pred" / f"{dt.now().timestamp()}"
    pred_dir.mkdir(parents=True, exist_ok=True)

    return log_dir, pred_dir


if __name__ == "__main__":
    log_dir, pred_dir = init_paths()

    entity_detector = stage1_entity_detector.EntityDetector()
    entity_detector.init_model(model_log_dir=str(log_dir.absolute()), default_weights="highlevel_wsft")

    # Input image should be sized around (1684, 1192)
    samples_path = Path("/home/shine/rednam/source/mrcnn-doc-parser/data/sample/resized")
    for img_path in tqdm(list(samples_path.glob("*.png"))):
        pred_file_path = pred_dir / img_path.name

        # Load Image
        image = cv2.imread(str(img_path.absolute()))

        # Detect entities
        entity_predictions = entity_detector.predict(image)

        entity_detector.visualize(
            image,
            entity_predictions,
            str(pred_file_path.absolute()),
            filter_classes=["figure_graphic", "tabular"],
            score_thresh=0.7,
        )
