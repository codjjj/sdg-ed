from types import SimpleNamespace

import torch

from sdg_finegrained_features import cls_features


def test_dinov2_pooling_is_exact_cls_token():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    output = SimpleNamespace(last_hidden_state=hidden)
    assert torch.equal(cls_features(output), hidden[:, 0, :])


def test_dinov2_pooling_rejects_non_sequence_output():
    output = SimpleNamespace(last_hidden_state=torch.zeros(2, 3))
    try:
        cls_features(output)
    except ValueError as error:
        assert "hidden-state shape" in str(error)
    else:
        raise AssertionError("non-sequence hidden state was accepted")

