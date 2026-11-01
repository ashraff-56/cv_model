"""Find the plate or bowl in a food photo and measure it, so the plate can be the ruler for grams.

SmartBite's users never say how much food there is or how big their plate is, and a photo alone has no
scale: a close-up of two gulab jamun looks like a full bowl. The plate is the one object of roughly known
size in almost every meal photo, so it supplies the scale (and the camera tilt) the photo lacks.

1. OWLv2 [Minderer et al., NeurIPS 2023], an open-vocabulary detector, proposes "plate" / "bowl" boxes.
2. SAM 2.1 [Ravi et al., 2024] turns the best box into a pixel mask.
3. Shape moments of the filled mask give tilt and size. Seen head-on, a round plate and a square plate (any
   shape with 3+-fold symmetry) both have an isotropic second-moment matrix; seen at an angle the matrix
   stretches. Stretching the image back until the moments are isotropic undoes the tilt (weak-perspective
   approximation), and the spread along the long axis is the plate's size in pixels.

Both models are Apache-2.0 (OWLv2 model card; SAM 2 README). Too heavy for a phone, fine on the server.

    python -m smartbite_cv.plate meal.jpg other.jpg --viz reports/plate_viz
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

OWL_ID = "google/owlv2-base-patch16-ensemble"
SAM_ID = "facebook/sam2.1-hiera-tiny"
QUERIES = (("plate", "a plate"), ("plate", "a dinner plate"), ("bowl", "a bowl"))
MAX_STRETCH = 4.0   # tilt beyond ~76 degrees from overhead is too oblique to rectify
MAX_BORDER = 0.20   # share of the outline allowed on the image edge (plate partly out of frame)


@dataclass
class PlateGeometry:
    cx: float
    cy: float
    sigma_major: float   # px: spread of the filled outline along its long axis
    sigma_minor: float
    angle_deg: float     # direction of the long axis (image x to the right, y down)
    area_px: float

    @property
    def stretch(self) -> float:
        return self.sigma_major / max(self.sigma_minor, 1e-6)

    @property
    def tilt_deg(self) -> float:
        """Camera tilt away from straight overhead implied by the foreshortening."""
        return math.degrees(math.acos(min(1.0, 1.0 / self.stretch)))


@dataclass
class Plate:
    kind: str            # "plate" or "bowl"
    query: str
    score: float         # OWLv2 box score
    box: tuple[float, float, float, float]
    mask: np.ndarray     # bool HxW, filled outline
    geometry: PlateGeometry
    touches_border: float  # fraction of the outline on the image edge: >0 means the plate is cut off

    @property
    def reliable(self) -> bool:
        """Mostly in frame (at most 20% of the outline on the image edge) and not too oblique. A plate with one
        edge clipped keeps its long axis, which carries the scale; a stricter rule would mostly reject the large
        plates and bias everything trained on reliable plates toward small ones."""
        return self.touches_border <= MAX_BORDER and self.geometry.stretch <= MAX_STRETCH


# ---------------------------------------------------------------- geometry (no model weights needed)
def fill_outline(mask: np.ndarray) -> np.ndarray:
    """Largest connected region with holes and dents filled, so food on the plate never punches holes."""
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    hull = cv2.convexHull(cv2.findNonZero((lab == k).astype(np.uint8)))
    out = np.zeros_like(m)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def mask_geometry(mask: np.ndarray) -> PlateGeometry:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        raise ValueError("empty mask")
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"]
    evals, evecs = np.linalg.eigh(cov)   # ascending
    major = evecs[:, 1]
    return PlateGeometry(cx=m["m10"] / m["m00"], cy=m["m01"] / m["m00"],
                         sigma_major=math.sqrt(evals[1]), sigma_minor=math.sqrt(max(evals[0], 0.0)),
                         angle_deg=math.degrees(math.atan2(major[1], major[0])), area_px=m["m00"])


def border_fraction(mask: np.ndarray) -> float:
    """Share of the outline's perimeter that lies on the image border."""
    edge = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).sum()
    perim = cv2.arcLength(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)[0][0], True) if mask.any() else 0.0
    return float(edge / perim) if perim else 0.0


def rectify_affine(geom: PlateGeometry, sigma_out: float, center_out: tuple[float, float]) -> np.ndarray:
    """2x3 affine: plate centre -> center_out, short axis stretched to match the long one, long-axis
    spread -> sigma_out px. The long axis is not foreshortened by tilt, so it carries the true size."""
    t = math.radians(geom.angle_deg)
    u, v = np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])
    a = (np.outer(u, u) + min(geom.stretch, MAX_STRETCH) * np.outer(v, v)) * (sigma_out / geom.sigma_major)
    b = np.asarray(center_out, float) - a @ np.array([geom.cx, geom.cy])
    return np.hstack([a, b[:, None]])


def rectify(image: np.ndarray, geom: PlateGeometry, sigma_out: float, out_hw: tuple[int, int],
            border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Top-down view with the plate centred and its long-axis spread at sigma_out px (uint8 in/out)."""
    scale = sigma_out / geom.sigma_major
    if scale < 0.5:   # warpAffine only interpolates bilinearly; shrink big photos with area averaging first
        pre = 2 * scale
        image = cv2.resize(image, None, fx=pre, fy=pre, interpolation=cv2.INTER_AREA)
        geom = PlateGeometry(geom.cx * pre, geom.cy * pre, geom.sigma_major * pre, geom.sigma_minor * pre,
                             geom.angle_deg, geom.area_px * pre * pre)
    h, w = out_hw
    m = rectify_affine(geom, sigma_out, (w / 2, h / 2))
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)


PLATE_VIEW_SIZE, PLATE_VIEW_SIGMA = 320, 64.0   # a round plate fills 80% of the width, a square one fits
GRAY = (124, 116, 104)                          # ImageNet mean colour: "nothing here" for the network


def plate_only_view(image: np.ndarray, plate: "Plate", size: int = PLATE_VIEW_SIZE,
                    sigma: float = PLATE_VIEW_SIGMA, margin: float = 0.06) -> np.ndarray:
    """Top-down view of the plate alone, always drawn at the same size, everything else grey.

    The plate becomes the only ruler: the table, the room and (in Nutrition5k) the scanning rig's fixed-size
    QR codes are blanked, so the network can't learn scale from anything a user's photo won't have.
    `margin` (fraction of the plate's width) keeps food that hangs over the rim."""
    view = rectify(image, plate.geometry, sigma, (size, size))
    mask = rectify(plate.mask.astype(np.uint8) * 255, plate.geometry, sigma, (size, size), cv2.BORDER_CONSTANT) > 127
    r = max(1, round(margin * 4 * sigma))
    mask = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    view[~mask] = GRAY
    return view


def portion_view(image: np.ndarray, plate: Plate, ref_sigma: dict[str, float], plate_scale: float = 1.0,
                 out_hw: tuple[int, int] = (480, 640)) -> np.ndarray:
    """What the portion model expects: a top-down 640x480 view with the plate drawn as large as a reference
    plate appears in Nutrition5k's fixed overhead camera. `ref_sigma` holds that size per container kind,
    measured on training photos; `plate_scale` > 1 says this plate is known to be bigger than the reference."""
    return rectify(image, plate.geometry, ref_sigma.get(plate.kind, ref_sigma["plate"]) * plate_scale, out_hw)


# ---------------------------------------------------------------- detector
def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _mask_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


class PlateDetector:
    """OWLv2 box -> SAM 2.1 mask -> filled outline + geometry. Weights download from Hugging Face once."""

    def __init__(self, owl_id: str = OWL_ID, sam_id: str = SAM_ID, device: str | None = None,
                 min_score: float = 0.1):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.owl_proc = Owlv2Processor.from_pretrained(owl_id)
        self.owl = Owlv2ForObjectDetection.from_pretrained(owl_id).to(self.device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(sam_id)
        self.sam = Sam2Model.from_pretrained(sam_id).to(self.device).eval()
        self.min_score = min_score

    def find_box(self, img) -> tuple[str, str, float, tuple] | None:
        """Best plate/bowl box (OWLv2 pads to a square, so boxes are scaled by the longer side)."""
        queries = [q for _, q in QUERIES]
        inputs = self.owl_proc(text=[queries], images=img, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.owl(**inputs)
        side = max(img.size)
        res = self.owl_proc.post_process_grounded_object_detection(
            out, threshold=self.min_score, target_sizes=[(side, side)], text_labels=[queries])[0]
        if len(res["scores"]) == 0:
            return None
        i = int(res["scores"].argmax())
        w, h = img.size
        x0, y0, x1, y1 = res["boxes"][i].tolist()
        box = (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1))
        kind, query = QUERIES[int(res["labels"][i])]
        return kind, query, float(res["scores"][i]), box

    def segment(self, img, box) -> np.ndarray:
        """SAM 2.1 mask for the box; of its three candidates, keep the one whose filled outline fits the box."""
        inputs = self.sam_proc(images=img, input_boxes=[[list(box)]], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.sam(**inputs, multimask_output=True)
        masks = self.sam_proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0]
        best, best_fit = None, -1.0
        for m in masks.numpy().astype(bool):
            if not m.any():
                continue
            filled = fill_outline(m)
            fit = _box_iou(_mask_box(filled), box)
            if fit > best_fit:
                best, best_fit = filled, fit
        return best if best is not None else np.zeros((img.height, img.width), bool)

    def detect(self, img) -> Plate | None:
        found = self.find_box(img)
        if found is None:
            return None
        kind, query, score, box = found
        mask = self.segment(img, box)
        if mask.sum() < 0.01 * mask.size:
            return None
        return Plate(kind, query, score, box, mask, mask_geometry(mask), border_fraction(mask))


# ---------------------------------------------------------------- detection cache
def plate_record(plate: Plate | None) -> dict:
    """JSON-safe record. The filled outline is a convex polygon, so its hull rebuilds the mask exactly."""
    if plate is None:
        return {"found": False}
    hull = cv2.convexHull(cv2.findNonZero(plate.mask.astype(np.uint8)))[:, 0]
    return {"found": True, "kind": plate.kind, "query": plate.query, "score": round(plate.score, 4),
            "box": [round(v, 1) for v in plate.box], "hull": hull.tolist(),
            "hw": list(plate.mask.shape), "touches_border": round(plate.touches_border, 4)}


def plate_from_record(rec: dict) -> Plate | None:
    if not rec.get("found"):
        return None
    mask = np.zeros(tuple(rec["hw"]), np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(rec["hull"], np.int32), 1)
    mask = mask.astype(bool)
    return Plate(rec["kind"], rec["query"], rec["score"], tuple(rec["box"]), mask, mask_geometry(mask),
                 rec["touches_border"])


class DetectionCache:
    """Append-only JSON-lines cache (key -> detection), so each photo is run through OWLv2 + SAM once."""

    def __init__(self, path: Path | None):
        import json
        self.json, self.path, self.records = json, path, {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    def detect(self, det: "PlateDetector", key: str, img) -> Plate | None:
        if key in self.records:
            return plate_from_record(self.records[key])
        plate = det.detect(img)
        rec = {"key": key, **plate_record(plate)}
        self.records[key] = rec
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(self.json.dumps(rec) + "\n")
        return plate


# ---------------------------------------------------------------- visual check
def draw(img_rgb: np.ndarray, plate: Plate | None) -> np.ndarray:
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if plate is None:
        cv2.putText(out, "no plate", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return out
    cnt = cv2.findContours(plate.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    lw = max(2, out.shape[1] // 400)
    cv2.drawContours(out, cnt, -1, (0, 255, 0), lw)
    x0, y0, x1, y1 = (int(v) for v in plate.box)
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 128, 0), lw)
    g = plate.geometry
    t = math.radians(g.angle_deg)
    for s, ang in ((2 * g.sigma_major, t), (2 * g.sigma_minor, t + math.pi / 2)):
        dx, dy = s * math.cos(ang), s * math.sin(ang)
        cv2.line(out, (int(g.cx - dx), int(g.cy - dy)), (int(g.cx + dx), int(g.cy + dy)), (0, 0, 255), lw)
    txt = f"{plate.kind} {plate.score:.2f} tilt {g.tilt_deg:.0f}deg sigma {g.sigma_major:.0f}px" + \
          ("" if plate.reliable else " UNRELIABLE")
    cv2.putText(out, txt, (10, 10 + 30 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * lw, (0, 0, 255), lw)
    return out


def main(argv: list[str] | None = None) -> None:
    from .imageio import load_rgb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--viz", type=Path, help="write annotated images and top-down views here")
    args = ap.parse_args(argv)
    det = PlateDetector()
    if args.viz:
        args.viz.mkdir(parents=True, exist_ok=True)
    for p in args.images:
        img = load_rgb(p)
        plate = det.detect(img)
        if plate is None:
            print(f"{p.name}: no plate found")
        else:
            g = plate.geometry
            print(f"{p.name}: {plate.kind} ({plate.query}) score {plate.score:.2f} tilt {g.tilt_deg:.0f} deg "
                  f"long-axis sigma {g.sigma_major:.0f}px border {plate.touches_border:.2f} "
                  f"{'ok' if plate.reliable else 'UNRELIABLE'}")
        if args.viz:
            arr = np.asarray(img)
            cv2.imwrite(str(args.viz / f"{p.stem}_plate.jpg"), draw(arr, plate))
            if plate is not None:
                top = rectify(arr, plate.geometry, 110.0, (480, 640))
                cv2.imwrite(str(args.viz / f"{p.stem}_topdown.jpg"), cv2.cvtColor(top, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
