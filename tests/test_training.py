import pytest
import torch
import torch.nn.functional as F

from regrounding.config import Experiment
from regrounding.experiment import run, run_direct
from regrounding.model import ConceptNet
from regrounding.train import Regrounding, fit, loss, reground

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


class TestDirectLoss:
    def test_it_is_the_cross_entropy_of_the_one_output_against_the_verdict(self):
        logits = torch.tensor([[2.0], [-1.0]])
        concepts = torch.tensor([[1, 1, 1], [1, 0, 1]])
        expected = F.binary_cross_entropy_with_logits(logits[:, 0], torch.tensor([1.0, 0.0]))
        assert torch.equal(loss(logits, concepts, "direct"), expected)


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
        [
            ("first", {"features.0"}),
            ("readout", {"head.3"}),
            ("head", {"head.1", "head.3"}),
            ("all", None),
        ],
    )
    def test_re_grounding_changes_only_the_layers_of_its_scope(self, scope, changed):
        model = ConceptNet(torch.Generator().manual_seed(0))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        generator = torch.Generator().manual_seed(1)
        images = torch.rand((4, 3, 32, 32), generator=generator)
        concepts = torch.randint(0, 2, (4, 3), generator=generator)
        how = Regrounding(scope=scope, steps=3, learning_rate=1e-2)
        reground(model, images, concepts, mode="concepts", how=how)
        moved = {
            k.rsplit(".", 1)[0]
            for k, v in model.state_dict().items()
            if not torch.equal(v, before[k])
        }
        assert moved == (changed or {k.rsplit(".", 1)[0] for k in before})


class TestConvergedFit:
    def _scenes(self):
        generator = torch.Generator().manual_seed(2)
        images = torch.rand((16, 3, 32, 32), generator=generator)
        return images, torch.randint(0, 2, (16, 3), generator=generator)

    def _fit(self, anchor):
        model = ConceptNet(torch.Generator().manual_seed(0))
        how = Regrounding(
            scope="readout", steps=0, learning_rate=0.0, fit="converged", anchor=anchor
        )
        reground(model, *self._scenes(), mode="concepts", how=how)
        return model

    def test_it_changes_the_readout_alone(self):
        before = ConceptNet(torch.Generator().manual_seed(0)).state_dict()
        after = self._fit(0.01).state_dict()
        moved = {k for k, v in after.items() if not torch.equal(v, before[k])}
        assert moved == {"head.3.weight", "head.3.bias"}

    def test_it_ends_where_the_objective_is_flat(self):
        model = self._fit(0.01)
        source = ConceptNet(torch.Generator().manual_seed(0)).head[-1]
        images, concepts = self._scenes()
        with torch.no_grad():
            features = model.head[:-1](model.features(images)).double()
        w = model.head[-1].weight.detach().double().requires_grad_(True)
        b = model.head[-1].bias.detach().double().requires_grad_(True)
        pull = (w - source.weight.double()).pow(2).sum() + (b - source.bias.double()).pow(2).sum()
        (loss(features @ w.T + b, concepts, "concepts") + 0.005 * pull).backward()
        assert w.grad is not None
        assert b.grad is not None
        assert max(w.grad.abs().max(), b.grad.abs().max()) < 1e-5

    def test_a_strong_pull_keeps_the_source_readout(self):
        source = ConceptNet(torch.Generator().manual_seed(0)).state_dict()["head.3.weight"]
        fitted = self._fit(1e6).state_dict()["head.3.weight"]
        assert torch.allclose(fitted, source, atol=1e-5)


class TestFrozenLayers:
    def _reference(self, model, images, concepts, scope):
        """Fine-tune the way re-grounding did before freezing: every layer keeps its gradient."""
        layers = {"readout": model.head[-1], "head": model.head, "all": model}
        optimizer = torch.optim.Adam(layers[scope].parameters(), lr=1e-2, fused=True)
        for _ in range(5):
            optimizer.zero_grad()
            loss(model(images), concepts, "verdict").backward()
            optimizer.step()

    @pytest.mark.parametrize("scope", ["readout", "head", "all"])
    def test_freezing_the_layers_outside_the_scope_changes_no_weight(self, scope):
        frozen = ConceptNet(torch.Generator().manual_seed(0))
        reference = ConceptNet(torch.Generator().manual_seed(0))
        generator = torch.Generator().manual_seed(1)
        images = torch.rand((8, 3, 32, 32), generator=generator)
        concepts = torch.randint(0, 2, (8, 3), generator=generator)
        how = Regrounding(scope=scope, steps=5, learning_rate=1e-2)
        reground(frozen, images, concepts, mode="verdict", how=how)
        self._reference(reference, images, concepts, scope)
        for a, b in zip(frozen.parameters(), reference.parameters(), strict=True):
            assert torch.equal(a, b)
        assert all(p.requires_grad for p in frozen.parameters())


#: The operations torch computes on the CPU through MKL's vector math (ATen's cpu/vml.h),
#: whose kernel MKL picks by CPU vendor.
VECTOR_MATH = {
    f"aten::{op}{suffix}"
    for op in (
        *("acos", "asin", "atan", "cos", "erf", "erfc", "erfinv", "exp"),
        *("log", "log10", "log2", "sin", "sqrt", "tan", "tanh", "trunc"),
    )
    for suffix in ("", "_")
}


def _called(job):
    """Return the name of every operation the dispatcher ran for ``job``, nested ones too."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        job()
    return {event.name for event in profile.events()}


class TestVendorIndependence:
    def _scenes(self):
        generator = torch.Generator().manual_seed(2)
        images = torch.rand((16, 3, 32, 32), generator=generator)
        return images, torch.randint(0, 2, (16, 3), generator=generator)

    def test_the_profiler_sees_an_operation_of_the_vector_math(self):
        assert "aten::sqrt" in _called(lambda: torch.ones(3).sqrt())

    @pytest.mark.parametrize("mode", ["concepts", "verdict", "direct"])
    def test_training_calls_no_operation_of_the_vector_math(self, mode):
        model = ConceptNet(torch.Generator().manual_seed(0))
        images, concepts = self._scenes()

        def job():
            fit(
                model,
                images,
                concepts,
                mode=mode,
                epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                generator=torch.Generator().manual_seed(1),
            )

        assert _called(job) & VECTOR_MATH == set()

    @pytest.mark.parametrize(
        "how",
        [
            Regrounding(scope="all", steps=2, learning_rate=1e-3),
            Regrounding(scope="readout", steps=0, learning_rate=0.0, fit="converged", anchor=0.01),
        ],
    )
    def test_re_grounding_calls_no_operation_of_the_vector_math(self, how):
        model = ConceptNet(torch.Generator().manual_seed(0))
        images, concepts = self._scenes()

        def job():
            reground(model, images, concepts, mode="verdict", how=how)

        assert _called(job) & VECTOR_MATH == set()


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
    def test_the_mixed_mode_starts_from_the_concept_mode_s_source_model(self):
        concepts = run(TINY, 0, "concepts", torch.device("cpu"))
        mixed = run(TINY, 0, "mixed", torch.device("cpu"))
        assert {**mixed[0], "mode": "concepts"} == concepts[0]
        assert mixed[1] == {**concepts[1], "mode": "mixed"}
        assert mixed[2] != {**concepts[2], "mode": "mixed"}

    @pytest.mark.slow
    def test_the_direct_network_yields_the_same_kinds_of_rows_on_the_same_scenes(self):
        rows = run_direct(TINY, 0, torch.device("cpu"))
        concepts = run(TINY, 0, "concepts", torch.device("cpu"))
        assert [(r["trained"], r["shots"]) for r in rows] == [
            (r["trained"], r["shots"]) for r in concepts
        ]
        assert [r["positives"] for r in rows] == [r["positives"] for r in concepts]

    @pytest.mark.slow
    def test_rehearsal_changes_the_regrounded_rows_alone(self):
        plain = run(TINY, 0, "concepts", torch.device("cpu"))
        rehearsed = run(TINY.updated(rehearsal=8), 0, "concepts", torch.device("cpu"))
        kept = [r["trained"] != "regrounded" or r["shots"] == 0 for r in plain]
        assert [a == b for a, b in zip(plain, rehearsed, strict=True)] == kept

    @pytest.mark.slow
    def test_target_calibration_moves_the_target_sets_and_nothing_else(self):
        plain = run(TINY, 0, "concepts", torch.device("cpu"))
        on_target = run(TINY.updated(calibrate_on="target"), 0, "concepts", torch.device("cpu"))
        sets = {"covered", "empty", "full", "decided", "decided_correct"}
        assert on_target[0] == plain[0]
        for a, b in zip(plain[1:], on_target[1:], strict=True):
            assert {k: v for k, v in a.items() if k not in sets} == {
                k: v for k, v in b.items() if k not in sets
            }

    @pytest.mark.slow
    def test_further_draws_add_rows_and_leave_the_first_draw_s_unchanged(self):
        one = run(TINY, 0, "concepts", torch.device("cpu"))
        two = run(TINY.updated(draws=2), 0, "concepts", torch.device("cpu"))
        assert [r for r in two if r["draw"] == 0] == one
        assert [(r["shots"], r["draw"]) for r in two if r["draw"] == 1] == [(0, 1), (4, 1)]

    @pytest.mark.slow
    def test_a_job_is_reproducible_on_the_same_machine(self):
        first = run(TINY, 0, "verdict", torch.device("cpu"))
        again = run(TINY, 0, "verdict", torch.device("cpu"))
        assert first == again
