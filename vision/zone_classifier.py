"""Our own trained AI: which zone is the camera looking at?

ZoneClassifier - loads the MobileNetV2 we trained (training/train_zone.py) and predicts
                 (zone label, confidence) for one camera frame, on CPU.
ZoneSmoother   - pure Python. One frame can be wrong, so a zone only counts after
                 ZONE_CONSECUTIVE predictions in a row agree with confidence >= ZONE_CONF.

torch is imported only inside ZoneClassifier, so the smoother (and its tests) never need it.
"""

import config


class ZoneClassifier:
    def __init__(self, model_path=None):
        import torch
        from PIL import Image

        # Same model definition and same preprocessing as training - no mismatch possible.
        from training.train_zone import build_model, eval_transform

        self._torch, self._Image = torch, Image
        ckpt = torch.load(model_path or config.ZONE_MODEL_PATH, map_location="cpu", weights_only=True)
        self.class_names = ckpt["class_names"]
        self.model = build_model(len(self.class_names), ckpt.get("dropout", 0.0), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.transform = eval_transform()

    def predict(self, frame_bgr):
        """One camera frame (OpenCV BGR) -> (label, confidence between 0 and 1)."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(self._Image.fromarray(rgb)).unsqueeze(0)
        with self._torch.no_grad():
            probs = self._torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.max(0)
        return self.class_names[int(idx)], float(conf)


class ZoneSmoother:
    """Confirms a zone only after several confident predictions in a row."""

    def __init__(self, needed=None, min_conf=None, ignore=None):
        self.needed = config.ZONE_CONSECUTIVE if needed is None else needed
        self.min_conf = config.ZONE_CONF if min_conf is None else min_conf
        self.ignore = config.EXTRA_CLASS if ignore is None else ignore
        self.label = None     # label of the current streak
        self.count = 0        # length of the current streak

    def update(self, label, confidence):
        """Feed one prediction. Returns the confirmed zone name, or None."""
        if label == self.ignore or confidence < self.min_conf:
            self.label, self.count = None, 0      # "Other" or unsure: streak broken
            return None
        if label == self.label:
            self.count += 1
        else:
            self.label, self.count = label, 1     # a different zone starts a new streak
        return self.label if self.count >= self.needed else None
