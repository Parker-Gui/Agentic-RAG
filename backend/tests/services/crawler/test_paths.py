from pathlib import Path

from app.services.crawler.common.paths import DATA_DIR, SITES_PATH


def test_default_data_dir_points_to_project_root() -> None:
    expected_root = Path(__file__).resolve().parents[4]

    assert DATA_DIR == expected_root
    assert SITES_PATH == expected_root / "config" / "monitor_sites.json"
