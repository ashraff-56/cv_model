import hashlib

import yaml

from smartbite_cv.modeling import checkpoint_meta


def test_checkpoint_meta_preserves_selected_v3_class_definitions(tmp_path):
    class_file = tmp_path / "classes_v3.yaml"
    class_file.write_text(yaml.safe_dump([
        {"slug": "apple", "display": "Apple", "variants": ["Red apple"], "category": "fruit"},
        {"slug": "biryani", "display": "Biryani", "variants": ["Chicken biryani"], "category": "dish"},
    ]), encoding="utf-8")
    cfg = {
        "model": {"name": "efficientnet_b0.ra_in1k"},
        "data": {"classes": str(class_file)},
        "img_size": 224,
    }

    meta = checkpoint_meta(cfg, ["biryani", "apple"])

    assert meta["classes"] == ["biryani", "apple"]
    assert [item["slug"] for item in meta["class_metadata"]] == ["biryani", "apple"]
    assert meta["class_metadata"][0]["display"] == "Biryani"
    assert meta["class_metadata"][1]["variants"] == ["Red apple"]
    assert meta["class_config_sha256"] == hashlib.sha256(class_file.read_bytes()).hexdigest()
