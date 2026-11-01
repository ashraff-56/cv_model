import numpy as np
import torch

from smartbite_cv.portion import (PortionNet, fit_sigma_scale, fuse_with_serving, gaussian_nll, mass_metrics,
                                  read_dish_metadata, split_ids)


def test_read_dish_metadata_parses_totals(tmp_path):
    p = tmp_path / "dish_metadata_cafe1.csv"
    p.write_text("dish_1,300.5,193.0,12.4,28.2,18.6,ingr_1,soy sauce,3.4,1.8,0.02,0.17,0.28\n"
                 "dish_2,0,0,0,0,0\n", encoding="utf-8")
    meta = read_dish_metadata([p])
    assert meta["dish_1"] == {"mass": 193.0, "kcal": 300.5, "fat": 12.4, "carb": 28.2, "protein": 18.6}
    assert meta["dish_2"]["mass"] == 0.0


def test_split_ids_disjoint_deterministic(tmp_path):
    d = tmp_path / "dish_ids" / "splits"
    d.mkdir(parents=True)
    (d / "tr.txt").write_text("\n".join(f"dish_{i}" for i in range(100)), encoding="utf-8")
    (d / "te.txt").write_text("\n".join(f"dish_{i}" for i in range(100, 120)), encoding="utf-8")
    cfg = {"seed": 1, "data": {"metadata_dir": str(tmp_path), "train_ids": "tr.txt", "test_ids": "te.txt",
                               "val_fraction": 0.1}}
    a, b = split_ids(cfg), split_ids(cfg)
    assert a == b and len(a["val"]) == 10 and len(a["train"]) == 90 and len(a["test"]) == 20
    assert not (set(a["train"]) & set(a["val"])) and not (set(a["train"]) & set(a["test"]))


def test_mass_metrics_matches_nutrition5k_definition():
    true = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 150.0, 300.0])
    m = mass_metrics(pred, true)
    assert abs(m["mae_g"] - 20.0) < 1e-9
    assert abs(m["mae_pct_of_mean"] - 10.0) < 1e-9          # 20 / mean(200) * 100
    assert abs(m["within_20pct"] - 2 / 3) < 1e-9                    # errors 10%, 25%, 0%


def test_sigma_scale_recovers_underestimated_uncertainty():
    rng = np.random.default_rng(0)
    mu = np.zeros(20000)
    logvar = np.full(20000, np.log(0.1 ** 2))               # model claims sigma 0.1
    y = rng.normal(0, 0.2, 20000)                            # truth spread is 0.2
    assert abs(fit_sigma_scale(mu, logvar, y) - 2.0) < 0.05


def test_fusion_weights_by_confidence():
    serving = 300.0
    m, s = fuse_with_serving(np.log(150.0), 0.3, serving, 0.3)
    assert abs(np.exp(m) - np.sqrt(150 * 300)) < 1e-6 and s < 0.3   # equal confidence -> geometric mean
    m, _ = fuse_with_serving(np.log(150.0), 1e-3, serving, 0.3)
    assert abs(np.exp(m) - 150.0) < 0.1                              # confident photo estimate wins
    m, _ = fuse_with_serving(np.log(150.0), 10.0, serving, 0.3)
    assert abs(np.exp(m) - 300.0) < 1.0                              # unsure photo estimate -> typical serving


def test_portion_net_outputs_mean_and_clamped_logvar():
    net = PortionNet({"name": "efficientnet_b0", "init_log_mass": 5.0}, pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 96, 128))
    assert out.shape == (2, 2)
    assert torch.allclose(out[:, 0], torch.full((2,), 5.0))          # zero-init head starts at the train mean
    loss = gaussian_nll(out, torch.full((2,), 5.0))
    assert torch.isfinite(loss)
