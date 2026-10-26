import pytest
import torch
import torch.nn.functional as F

from regrounding.config import Experiment
from regrounding.experiment import run
from regrounding.model import ConceptNet
from regrounding.train import loss, reground

TINY = Experiment(
    source_train=400,
    calibration=100,
    test=200,
    pool=12,
    target_train=400,
    epochs=3,
    shots=(0, 4),
    seeds=(0,),
    regrounding_steps=20,
)


class TestVerdictLoss:
    def test_it_is_the_cross_entropy_of_the_product_of_the_probabilities(self):
        generator = torch.Generator().manual_seed(0)
        logits = 3 * torch.randn((500, 3), generator=generator)
        concepts = torch.randint(0, 2, (500, 3), generator=generator)
        product = torch.sigmoid(logits.double()).prod(dim=-1)
        expected = F.binary_cross_entropy(product, concepts.bool().all(dim=-1).double())
        assert torch.allclose(loss(logits, concepts, "verdict"), expected, rtol=1e-9)

    @pytest.mark.parametrize(("value", "label"), [(-12.0, 1), (40.0, 0), (100.0, 0)])
    def test_a_confidently_wrong_scene_keeps_its_gradient(self, value, label):
        logits = torch.full((1, 3), value, requires_grad=True)
        loss(logits, torch.full((1, 3), label), "verdict").backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert (logits.grad.abs() > 0.1).all()


class TestModel:
    def test_the_weights_depend_on_the_generator_passed_in_and_not_on_the_global_one(self):
        torch.manual_seed(1)
        first = ConceptNet(torch.Generator().manual_seed(0))
        torch.manual_seed(2)
        again = ConceptNet(torch.Generator().manual_seed(0))
        for a, b in zip(first.parameters(), again.parameters(), strict=True):
            assert torch.equal(a, b)

    @pytest.mark.parametrize(
        ("scope", "changed"),
        [("readout", {"head.3"}), ("head", {"head.1", "head.3"}), ("all", None)],
    )
    def test_re_grounding_changes_only_the_layers_of_its_scope(self, scope, changed):
        model = ConceptNet(torch.Generator().manual_seed(0))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        generator = torch.Generator().manual_seed(1)
        images = torch.rand((4, 3, 32, 32), generator=generator)
        concepts = torch.randint(0, 2, (4, 3), generator=generator)
        reground(model, images, concepts, mode="concepts", steps=3, learning_rate=1e-2, scope=scope)
        moved = {
            k.rsplit(".", 1)[0]
            for k, v in model.state_dict().items()
            if not torch.equal(v, before[k])
        }
        assert moved == (changed or {k.rsplit(".", 1)[0] for k in before})


class TestEndToEnd:
    @pytest.mark.slow
    def test_one_job_yields_a_source_row_and_one_target_row_per_shot_count(self):
        rows = run(TINY, 0, "concepts", torch.device("cpu"))
        assert [(r["trained"], r["shots"]) for r in rows] == [
            ("source", 0),
            ("regrounded", 0),
            ("regrounded", 4),
            ("ceiling", 400),
            ("scratch", 400),
        ]
        assert [r["domain"] for r in rows] == ["source", *["target"] * 4]
        assert all(r["positives"] + r["negatives"] == r["scenes"] for r in rows)

    @pytest.mark.slow
    def test_a_job_is_reproducible_on_the_same_machine(self):
        first = run(TINY, 0, "verdict", torch.device("cpu"))
        again = run(TINY, 0, "verdict", torch.device("cpu"))
        assert first == again
