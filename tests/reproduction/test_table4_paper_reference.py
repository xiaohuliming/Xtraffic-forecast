from scripts.reproduction.igstgnn_table4.manifest import METHODS, REGIONS
from scripts.reproduction.igstgnn_table4.paper_reference import (
    DEFAULT_REFERENCE_PATH,
    load_paper_table4_reference,
)


def test_paper_table4_reference_contains_the_exact_ordered_39_rows():
    reference = load_paper_table4_reference()

    assert reference["status"] == "external_paper_reference_verified"
    assert reference["row_count"] == 39
    assert reference["source_lock"]["arxiv_version"] == 2
    assert reference["source_lock"]["pdf_sha256"] == (
        "e2bf111af6b8fafbb5bb41a320d37e920dbdaca5122feac0600af6cc114414a7"
    )
    assert [(row["region"], row["method"]) for row in reference["rows"]] == [
        (region, method) for region in REGIONS for method in METHODS
    ]
    assert len(reference["source_file_sha256"]) == 64
    assert DEFAULT_REFERENCE_PATH.is_file()


def test_paper_table4_reference_matches_selected_published_cells():
    rows = {
        (row["region"], row["method"]): row
        for row in load_paper_table4_reference()["rows"]
    }

    assert rows[("Alameda", "igstgnn")]["h3_mae"] == 11.80
    assert rows[("Contra_Costa", "stgcn")]["avg_mape_percent"] == 15.46
    assert rows[("Orange", "igstgnn")]["avg_mae"] == 13.13
    assert rows[("Orange", "hl")]["h12_rmse"] == 42.49
