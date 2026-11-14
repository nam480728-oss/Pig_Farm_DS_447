"""Kiem tra khung du an: cac goi import duoc va file cau hinh doc duoc."""
from pathlib import Path

import yaml

GOC = Path(__file__).resolve().parents[1]


def test_doc_cau_hinh():
    cfg = yaml.safe_load((GOC / "config" / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["dem"]["dot_phut"] > 0


def test_import_cac_module():
    import ai.modules.bo_an  # noqa: F401
    import ai.modules.dem  # noqa: F401
    import ai.modules.nhip_tho  # noqa: F401
