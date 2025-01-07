from copy import deepcopy

import pytest
import torch

from torch_training.classifier.base.model_checkpoint_handler import (
    ModelCheckpointHandler,
)


class DummyModel(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(in_features=1, out_features=1, bias=False)


@pytest.fixture
def model() -> DummyModel:
    model = DummyModel()
    model.weight = torch.nn.Parameter(torch.zeros_like(model.weight))
    return model


@pytest.fixture
def increasing_handler(model: DummyModel) -> ModelCheckpointHandler:
    return ModelCheckpointHandler(
        model=model, initial_metric_value=0, increasing_metric=True
    )


@pytest.fixture
def decreasing_handler(model: DummyModel) -> ModelCheckpointHandler:
    return ModelCheckpointHandler(
        model=model, initial_metric_value=0, increasing_metric=False
    )


@pytest.mark.parametrize("handler", ["increasing_handler", "decreasing_handler"])
def test_initial_checkpoint_is_set(
    handler: str, model: DummyModel, request: pytest.FixtureRequest
) -> None:
    model_handler: ModelCheckpointHandler = request.getfixturevalue(handler)
    initial_state_dict = deepcopy(model.state_dict())
    assert model_handler.best_checkpoint == initial_state_dict


@pytest.mark.parametrize(
    "handler, metric_value",
    [
        ("increasing_handler", 1.0),
        ("decreasing_handler", -1.0),
    ],
)
def test_update_performed_if_metric_improved(
    handler: str, metric_value: float, model: DummyModel, request: pytest.FixtureRequest
) -> None:
    model.weight = torch.nn.Parameter(torch.ones_like(model.weight))

    model_handler: ModelCheckpointHandler = request.getfixturevalue(handler)
    model_handler.update(model, metric_value)

    assert model_handler.best_metric_value == metric_value
    assert model_handler.best_checkpoint == model.state_dict()


@pytest.mark.parametrize(
    "handler, metric_value",
    [
        ("increasing_handler", -1.0),
        ("decreasing_handler", 1.0),
    ],
)
def test_no_update_performed_if_metric_not_improved(
    handler: str, metric_value: float, model: DummyModel, request: pytest.FixtureRequest
) -> None:
    model_handler: ModelCheckpointHandler = request.getfixturevalue(handler)
    old_model_state = model_handler.best_checkpoint
    old_metric_value = model_handler.best_metric_value
    model.weight = torch.nn.Parameter(torch.ones_like(model.weight))

    model_handler.update(model, metric_value)

    assert model_handler.best_metric_value == old_metric_value
    assert model_handler.best_checkpoint == old_model_state


def test_load_best_checkpoint(
    increasing_handler: ModelCheckpointHandler, model: DummyModel
) -> None:
    model.weight = torch.nn.Parameter(torch.ones_like(model.weight))
    increasing_handler.update(model, metric_value=1)

    new_model = DummyModel()
    increasing_handler.load_best_checkpoint(new_model)

    assert model.state_dict() == new_model.state_dict()
