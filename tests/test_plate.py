import math

import cv2
import numpy as np
import pytest

from smartbite_cv.plate import (MAX_STRETCH, PlateGeometry, border_fraction, fill_outline, mask_geometry,
                                rectify, rectify_affine)


def ellipse_mask(h=600, w=800, center=(400, 300), axes=(200, 100), angle=30.0):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, center, axes, angle, 0, 360, 1, -1)
    return m.astype(bool)


def test_circle_seen_at_60_degrees_gives_tilt_60():
    g = mask_geometry(ellipse_mask(axes=(200, 100)))   # minor/major = cos(60)
    assert g.tilt_deg == pytest.approx(60, abs=1)
    assert g.sigma_major == pytest.approx(200 / 2, rel=0.01)   # disk of radius r: sigma = r / 2
    assert g.angle_deg % 180 == pytest.approx(30, abs=1)


def test_square_plate_tilt_is_independent_of_its_rotation():
    """A square's moments are isotropic at any in-plane rotation, so only the camera tilt shows."""
    for rot in (0, 20, 45):
        sq = cv2.boxPoints(((0, 0), (300, 300), rot))
        sq[:, 1] *= math.cos(math.radians(45))   # camera tilted 45 degrees about the x axis
        m = np.zeros((600, 800), np.uint8)
        cv2.fillConvexPoly(m, (sq + (400, 300)).astype(np.int32), 1)
        assert mask_geometry(m.astype(bool)).tilt_deg == pytest.approx(45, abs=2)


def test_rectify_makes_the_plate_round_and_sets_its_size():
    m = ellipse_mask(axes=(200, 90), angle=-20)
    g = mask_geometry(m)
    img = np.repeat(m[..., None].astype(np.uint8) * 255, 3, axis=2)
    top = rectify(img, g, sigma_out=60.0, out_hw=(480, 640))[..., 0] > 127
    g2 = mask_geometry(top)
    assert g2.tilt_deg < 5
    assert g2.sigma_major == pytest.approx(60, rel=0.03)
    assert (g2.cx, g2.cy) == pytest.approx((320, 240), abs=2)


def test_rectify_affine_caps_extreme_tilt():
    g = PlateGeometry(0, 0, 100, 5, 0, 1)   # stretch 20
    a = rectify_affine(g, 100, (0, 0))[:, :2]
    assert np.linalg.svd(a, compute_uv=False).max() == pytest.approx(MAX_STRETCH, rel=1e-6)


def test_fill_outline_fills_food_holes_and_drops_specks():
    m = ellipse_mask().astype(np.uint8)
    cv2.circle(m, (400, 300), 50, 0, -1)    # food region carved out of the plate
    m[5:10, 5:10] = 1                       # speck far away
    f = fill_outline(m.astype(bool))
    assert f[300, 400] and not f[7, 7]


def test_border_fraction_flags_cut_off_plates():
    assert border_fraction(ellipse_mask()) == 0
    assert border_fraction(ellipse_mask(center=(400, 0))) > 0.1


def test_zoom_keeps_size_and_scales_content():
    from smartbite_cv.plate_bench import zoom
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
    for z in (0.6, 1.25):
        out = zoom(img, z)
        assert out.shape == img.shape
        g = mask_geometry(out[..., 0] > 127)
        assert g.sigma_major == pytest.approx(50 * z, rel=0.03)


def test_summarise_pairs_plate_against_raw_by_dish():
    from smartbite_cv.plate_bench import summarise
    rows = []
    for i in range(20):
        true = 100.0 + i
        for cam in "AB":
            rows.append({"condition": "side", "dish_id": f"d{i}", "camera": cam, "true_g": true, "found": 1,
                         "reliable": 1, "raw_g": true * 1.5, "plate_g": true * 1.1, "plate_any_g": true * 1.1})
    s = summarise(rows, constant_g=100.0)["side"]
    assert s["dishes"] == 20 and s["photos"] == 40
    assert s["plate"]["mae_g"] < s["raw"]["mae_g"]
    assert s["plate_minus_raw_mae_g"]["ci95"][1] < 0
