from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from torch_training.history import History


@pytest.fixture
def history() -> History:
    return History()


def test_empty_history(history: History) -> None:
    assert history.training == dict()
    assert history.validation == dict()


def test_log_non_existent_metric_adds_new_dict_entry(history: History) -> None:
    history.log("test_metric", 0, 0)

    assert "test_metric" in history.training
    assert "test_metric" in history.validation

    assert isinstance(history.training["test_metric"], list)
    assert isinstance(history.validation["test_metric"], list)


def test_log_value_increases_all_corresponding_lists_by_one(history: History) -> None:
    history.log("test_metric", 0, 0)

    assert len(history.training["test_metric"]) == 1
    assert len(history.validation["test_metric"]) == 1

    history.log("test_metric", 1, 1)

    assert len(history.training["test_metric"]) == 2
    assert len(history.validation["test_metric"]) == 2


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_saved_csv_structure_matches_expected(history: History, delimiter: str) -> None:
    history.log("A", 111, 121)
    history.log("A", 112, 122)
    history.log("B", 211, 221)
    history.log("B", 212, 222)

    with TemporaryDirectory() as tmp_dir:
        csv_file = Path(f"{tmp_dir}/history.csv")
        history.save_csv(csv_file, csv_delimiter=delimiter)
        written_text = csv_file.read_text()

    assert written_text == (
        f"split{delimiter}A{delimiter}B\n"
        f"training{delimiter}111{delimiter}211\n"
        f"training{delimiter}112{delimiter}212\n"
        f"validation{delimiter}121{delimiter}221\n"
        f"validation{delimiter}122{delimiter}222\n"
    )
