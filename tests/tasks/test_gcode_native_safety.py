import importlib.util
import json
from pathlib import Path
import time

import pytest


SCRIPTS = Path(__file__).parents[2] / "tasks/engineering/gcode/scripts"
SPEC = importlib.util.spec_from_file_location("gcode_safety", SCRIPTS / "native_safety.py")
safety = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(safety)


@pytest.fixture
def replay(tmp_path):
    app = pytest.importorskip("FreeCAD")
    part = pytest.importorskip("Part")
    stock = part.makeBox(10, 10, 4, app.Vector(-5, -5, -4))
    target = part.makePlane(10, 10, app.Vector(30, 30, -4))
    instance = safety.SafetyReplay(
        target,
        stock,
        {},
        work_dir=tmp_path,
        initial_machine_mm=(0, 0, 6),
        initial_tool=0,
        tool_gauge_offsets_mm={0: (0, 0, 0), 1: (0, 0, 0)},
        linear_tolerance_mm=0.001,
        time_limit_s=60,
    )
    instance.shapes[1] = {"Cutter": part.makeCylinder(1, 8)}
    yield instance
    instance.feeds.close()
    instance.evidence.close()


def append_feed(replay, start, end, error=0):
    row = {"tool": 1, "start_mm": start, "end_mm": end, "chord_error_mm": error}
    replay.feeds.write(json.dumps(row) + "\n")
    return row


def test_swept_bounds_include_chord_error():
    assert safety._swept_bounds(((-1, -2, 0), (1, 2, 4)), (4, 0, 2), (-2, 3, -1), 0.5) == (
        (-3.5, -2.5, -1.5),
        (5.5, 5.5, 6.5),
    )
    assert not safety._separated(((0, 0, 0), (1, 1, 1)), ((1, 1, 1), (2, 2, 2)))


def test_local_bounds_match_global_endpoint_and_enclosure_cuts(replay):
    rows = [
        append_feed(replay, (-2, 0, -3), (-1, 0, -3), 0.01),
        append_feed(replay, (2, 0, -2), (2, 1, -2), 0.05),
        append_feed(replay, (100, 100, -4), (101, 100, -4)),
    ]
    query = ((-3.5, -1.5, -3.5), (3.5, 2.5, 0.5))
    expected_upper = replay.initial_stock.copy()
    expected_lower = replay.initial_stock.copy()
    cutter = replay.shapes[1]["Cutter"]
    for row in rows:
        expected_upper = expected_upper.cut(
            [replay._placed(cutter, row[key]) for key in ("start_mm", "end_mm")]
        )
        expected_lower = expected_lower.cut(
            replay._box(
                safety._swept_bounds(
                    safety._bounds(cutter),
                    row["start_mm"],
                    row["end_mm"],
                    row["chord_error_mm"] + replay.tolerance,
                )
            )
        )
    upper, lower = replay._remaining(query)
    for actual, expected in ((upper, expected_upper), (lower, expected_lower)):
        expected = expected.common(replay._box(query))
        assert actual.cut(expected).Volume < 1e-8
        assert expected.cut(actual).Volume < 1e-8
    assert replay.report["remaining_stock_candidate_feeds"] == 2


def test_history_is_local_incremental_and_does_not_consume_future_feeds(replay):
    for index in range(1000):
        append_feed(replay, (100 + index, 100, -4), (101 + index, 100, -4))
    query = ((-0.5, -0.5, -3.5), (0.5, 0.5, -0.5))
    upper, lower = replay._remaining(query)
    assert upper.Volume == pytest.approx(3)
    assert lower.Volume == pytest.approx(3)
    assert replay.report["remaining_stock_boolean_cuts"] == 0
    append_feed(replay, (0, 0, -4), (0, 0, -4))
    upper, lower = replay._remaining(query)
    assert upper.Volume == lower.Volume == 0
    assert len(replay.feed_index) == 1001
    assert replay.report["remaining_stock_boolean_cuts"] == 1
    replay._remaining(query)
    assert len(replay.feed_index) == 1001
    assert replay.feeds.tell() == replay.applied_feed_offset


@pytest.mark.parametrize("role", ["Cutter", "Shank", "Holder"])
def test_rapid_components_clear_only_after_prior_cut(replay, role):
    shape = replay.part.makeCylinder(0.4, 1)
    start, end = (0, 0, -3), (0, 0, -1)
    replay._stock_check("rapid_stock", role, 1, shape, start, end, 0, {})
    assert replay.report["checks"]["rapid_stock"]["collision"] == 1
    append_feed(replay, (0, 0, -4), (0, 0, -4))
    replay._stock_check("rapid_stock", role, 1, shape, start, end, 0, {})
    assert replay.report["checks"]["rapid_stock"]["clear"] == 1
    assert (
        replay.report["remaining_stock_empty_proofs"]
        + replay.report["remaining_stock_partial_proofs"]
    ) == 1


@pytest.mark.parametrize("role", ["Shank", "Holder"])
def test_noncutting_components_still_hit_unremoved_stock(replay, role):
    append_feed(replay, (0, 0, -4), (0, 0, -4))
    shape = replay.part.makeCylinder(0.4, 1)
    start, end = (3, 0, -3), (3, 0, -1)
    replay._stock_check("noncutting_stock", role, 1, shape, start, end, 0, {})
    assert replay.report["checks"]["noncutting_stock"]["collision"] == 1


def test_current_feed_enclosure_cannot_certify_clear(replay):
    shape = replay.part.makeCylinder(0.4, 1)
    start, end = (0, 0, -3), (0, 0, -1)
    replay._stock_check("noncutting_stock", "Shank", 1, shape, start, end, 0, {}, feed={})
    assert replay.report["checks"]["noncutting_stock"]["unavailable"] == 1
    assert replay.report["checks"]["noncutting_stock"]["clear"] == 0


def test_endpoint_samples_do_not_erase_unsampled_feed_interior(replay):
    append_feed(replay, (-4, 0, -4), (4, 0, -4), 0.05)
    query = ((-0.5, -0.5, -3.5), (0.5, 0.5, -0.5))
    upper, lower = replay._remaining(query)
    assert upper.Volume == pytest.approx(3)
    assert lower.Volume == 0
    shape = replay.part.makeCylinder(0.4, 1)
    result = replay._sweep(shape, (0, 0, -3), (0, 0, -1), upper, lower, 0)
    assert result["status"] == "unavailable"


@pytest.mark.parametrize(
    "start,end", [((-4, 0, -4), (4, 0, -4)), ((-4, -4, -4), (4, 4, -4)), ((0, 0, -6), (0, 0, -2))]
)
def test_verified_cylinder_clears_continuous_feed_interior(replay, start, end):
    append_feed(replay, start, end)
    upper, lower = replay._remaining(((-0.3, -0.3, -3.5), (0.3, 0.3, -0.5)))
    assert upper.Volume == lower.Volume == 0
    assert replay.report["remaining_stock_swept_cylinders"] == 1
    assert replay.report["remaining_stock_boolean_cuts"] == 1


def test_inscribed_sweep_rejects_hollow_cutter_and_uncertain_or_sloped_paths(replay):
    row = {"tool": 1, "start_mm": (-4, 0, -4), "end_mm": (4, 0, -4), "chord_error_mm": 0}
    assert replay._feed_inner_sweep({**row, "chord_error_mm": 0.001}) is None
    assert replay._feed_inner_sweep({**row, "end_mm": (4, 0, -3)}) is None
    replay.shapes[1]["Cutter"] = replay.shapes[1]["Cutter"].cut(replay.part.makeCylinder(0.2, 8))
    assert replay._feed_inner_sweep(row) is None
    append_feed(replay, row["start_mm"], row["end_mm"])
    upper, lower = replay._remaining(((-0.3, -0.3, -3.5), (0.3, 0.3, -0.5)))
    assert upper.Volume > 0
    assert replay.report["remaining_stock_swept_cylinders"] == 0


def test_native_null_difference_is_an_empty_containment_remainder(replay, monkeypatch):
    null = replay.part.Shape()
    assert null.isNull()
    with pytest.raises(replay.part.OCCError, match="NULL shape"):
        null.isValid()
    original = replay.part.makeCylinder

    class Cylinder:
        def cut(self, cutter):
            return null

    monkeypatch.setattr(replay.part, "makeCylinder", lambda *args: Cylinder())
    row = {"tool": 1, "start_mm": (-4, 0, -4), "end_mm": (4, 0, -4), "chord_error_mm": 0}
    sweep = replay._feed_inner_sweep(row)
    assert sweep.isValid()
    assert replay.inscribed_cylinders[1] is not None
    monkeypatch.setattr(replay.part, "makeCylinder", original)
    cut = original(1, 8).cut(original(2, 10, replay.app.Vector(0, 0, -1)))
    assert cut.isNull() or not cut.Solids


def test_failed_upper_cut_preserves_stock_but_failed_lower_cut_stops(replay):
    class FailedCut:
        def __getattr__(self, name):
            return getattr(replay.initial_stock, name)

        def cut(self, cutters):
            raise ValueError("Null shape")

    stock = FailedCut()
    cutter = replay.shapes[1]["Cutter"]
    completed = set()
    retained = replay._subtract(
        stock, cutter, upper_bound=True, on_success=lambda: completed.add("cut")
    )
    assert retained is stock
    assert not completed
    assert retained.Volume == replay.initial_stock.Volume
    assert replay.report["remaining_stock_failed_upper_cuts"] == 1
    assert replay.report["remaining_stock_boolean_cuts"] == 0
    with pytest.raises(ValueError, match="Null shape"):
        replay._subtract(stock, cutter)
    result = replay._sweep(
        cutter, (0, 0, -3), (0, 0, -1), replay.initial_stock, replay.initial_stock, 0
    )
    assert result["status"] == "collision"


@pytest.mark.parametrize("solid", [False, True])
def test_invalid_upper_result_retains_stock_and_invalid_lower_result_stops(replay, solid):
    invalid = replay.part.makeShell(replay.initial_stock.Faces[:-1])
    if solid:
        invalid = replay.part.makeSolid(invalid)
    assert not invalid.isClosed()

    class InvalidCut:
        def __getattr__(self, name):
            return getattr(replay.initial_stock, name)

        def cut(self, cutters):
            return invalid

    stock = InvalidCut()
    cutter = replay.shapes[1]["Cutter"]
    completed = set()
    retained = replay._subtract(
        stock, cutter, upper_bound=True, on_success=lambda: completed.add("cut")
    )
    assert retained is stock
    assert not completed
    assert retained.Volume == replay.initial_stock.Volume
    assert replay.report["remaining_stock_failed_upper_cuts"] == 1
    assert replay.report["remaining_stock_boolean_cuts"] == 1
    with pytest.raises(safety.SafetyUnavailable, match="valid closed native solids"):
        replay._subtract(stock, cutter)
    assert not completed
    for lower, expected in (
        (replay.initial_stock, "collision"),
        (replay.part.Shape(), "unavailable"),
    ):
        result = replay._sweep(cutter, (0, 0, -3), (0, 0, -1), retained.copy(), lower, 0)
        assert result["status"] == expected


@pytest.mark.parametrize("failed_member", [False, True])
def test_invalid_upper_batch_retries_validated_members_without_false_completion(
    replay, failed_member
):
    invalid = replay.part.makeSolid(replay.part.makeShell(replay.initial_stock.Faces[:-1]))
    cutters = [
        replay._placed(replay.shapes[1]["Cutter"], point) for point in ((-2, 0, -4), (2, 0, -4))
    ]

    class RetryBatch:
        def __init__(self, shape):
            self.shape = shape

        def __getattr__(self, name):
            return getattr(self.shape, name)

        def cut(self, cutter):
            if isinstance(cutter, list) or (failed_member and cutter is cutters[1]):
                return invalid
            return RetryBatch(self.shape.cut(cutter))

    completed = set()
    result = replay._subtract(
        RetryBatch(replay.initial_stock),
        cutters,
        upper_bound=True,
        on_success=lambda: completed.add("batch"),
    )
    expected = replay.initial_stock.cut(cutters[:1] if failed_member else cutters)
    assert result.isValid() and result.isClosed()
    assert result.Volume == pytest.approx(expected.Volume)
    assert completed == (set() if failed_member else {"batch"})
    assert replay.report["remaining_stock_failed_upper_cuts"] == (2 if failed_member else 1)
    with pytest.raises(safety.SafetyUnavailable, match="valid closed native solids"):
        replay._subtract(RetryBatch(replay.initial_stock), cutters)


def test_successful_unchanged_and_empty_cuts_confirm_removal(replay):
    completed = set()
    unchanged = replay._subtract(
        replay.initial_stock,
        replay._placed(replay.shapes[1]["Cutter"], (100, 100, 100)),
        upper_bound=True,
        on_success=lambda: completed.add("unchanged"),
    )
    assert unchanged.Volume == pytest.approx(replay.initial_stock.Volume)
    empty = replay._subtract(
        unchanged,
        replay.initial_stock,
        upper_bound=True,
        on_success=lambda: completed.add("empty"),
    )
    assert empty.isNull() or not empty.Solids
    assert completed == {"unchanged", "empty"}


def test_partial_removal_can_certify_sweep_without_emptying_local_box(replay):
    append_feed(replay, (0, 0, -4), (0, 0, -4))
    shape = replay.part.makeCylinder(0.4, 1)
    start, end = (0, 0, -3), (0, 0, -1)
    upper, lower = replay._remaining(
        ((-2, -2, -4), (2, 2, 0)),
        clearance=lambda bound: (
            replay._sweep(shape, start, end, bound, replay.part.Shape(), 0)["status"] == "clear"
        ),
    )
    assert upper.Volume > 0
    assert lower.isNull()
    assert replay.report["remaining_stock_partial_proofs"] == 1
    assert replay._sweep(shape, start, end, upper, lower, 0)["status"] == "clear"


@pytest.mark.parametrize("start,end", [((0, 0, -3), (0, 0, -1)), ((-7, 0, -3), (7, 0, -3))])
def test_clearance_only_rejects_contact_without_deciding_actual_collision(replay, start, end):
    shape = replay.part.makeCylinder(0.4, 1)
    result = replay._sweep(
        shape, start, end, replay.initial_stock, replay.part.Shape(), 0, clearance_only=True
    )
    assert result["status"] == "unavailable"
    assert replay.report["confirmed_collision_count"] == 0
    assert (
        replay._sweep(shape, start, end, replay.initial_stock, replay.initial_stock, 0)["status"]
        == "collision"
    )


def test_clearance_only_still_certifies_entire_sweep_through_removed_stock(replay):
    obstacle = replay.initial_stock.cut(replay._placed(replay.shapes[1]["Cutter"], (0, 0, -4)))
    shape = replay.part.makeCylinder(0.4, 1)
    for clearance_only in (False, True):
        assert (
            replay._sweep(
                shape,
                (0, 0, -3),
                (0, 0, -1),
                obstacle,
                replay.part.Shape(),
                0,
                clearance_only=clearance_only,
            )["status"]
            == "clear"
        )


def test_adjacent_queries_reuse_only_completed_removal(replay):
    append_feed(replay, (-4, 0, -4), (4, 0, -4))
    shape = replay.part.makeCylinder(0.4, 1)
    for coordinate in (0, 0.5):
        replay._stock_check(
            "rapid_stock",
            "Shank",
            1,
            shape,
            (coordinate, 0, -3),
            (coordinate, 0, -1),
            0,
            {},
        )
    assert replay.report["checks"]["rapid_stock"]["clear"] == 2
    assert replay.report["remaining_stock_boolean_cuts"] == 1
    assert replay.report["remaining_stock_cache_hits"] == 1
    replay._stock_check(
        "rapid_stock",
        "Shank",
        1,
        shape,
        (0, 3, -3),
        (0, 3, -1),
        0,
        {},
    )
    assert replay.report["checks"]["rapid_stock"]["collision"] == 1


def test_cached_inner_removal_is_not_repeated_and_new_feeds_still_apply(replay):
    append_feed(replay, (-4, 0, -4), (4, 0, -4))
    query = ((-2, -2, -3.5), (2, 2, -0.5))
    upper, lower = replay._remaining(query, clearance=lambda bound: False)
    cuts = replay.report["remaining_stock_boolean_cuts"]
    repeated_upper, repeated_lower = replay._remaining(query, clearance=lambda bound: False)
    assert repeated_upper.Volume == pytest.approx(upper.Volume)
    assert repeated_lower.Volume == pytest.approx(lower.Volume)
    assert replay.report["remaining_stock_boolean_cuts"] - cuts == 1
    append_feed(replay, (-4, 1.5, -4), (4, 1.5, -4))
    updated_upper, updated_lower = replay._remaining(query, clearance=lambda bound: False)
    assert updated_upper.Volume < upper.Volume
    assert updated_lower.Volume < lower.Volume
    assert replay.report["remaining_stock_boolean_cuts"] - cuts == 3


def test_cached_endpoint_removal_does_not_skip_unapplied_other_endpoint(replay):
    append_feed(replay, (-2, 0, -4), (2, 0, -4), error=0.01)
    first = ((-3, -0.2, -3.5), (0, 0.2, -0.5))
    second = ((1.5, -0.2, -3.5), (2.5, 0.2, -0.5))
    cutter = replay._placed(replay.shapes[1]["Cutter"], (2, 0, -4))
    first_upper, unused_lower = replay._remaining(first, clearance=lambda bound: False)
    assert first_upper.common(cutter).Volume > 0
    second_upper, second_lower = replay._remaining(second, clearance=lambda bound: False)
    assert second_upper.common(cutter).Volume == pytest.approx(0)
    cuts = replay.report["remaining_stock_boolean_cuts"]
    repeated_upper, repeated_lower = replay._remaining(second, clearance=lambda bound: False)
    assert repeated_upper.Volume == pytest.approx(second_upper.Volume)
    assert repeated_lower.Volume == pytest.approx(second_lower.Volume)
    assert replay.report["remaining_stock_boolean_cuts"] - cuts == 1


@pytest.mark.parametrize("failure", ["null_exception", "invalid_result"])
def test_cached_failed_inner_removal_is_retried(replay, failure):
    attempts = []

    class RetryCut:
        def __init__(self, shape):
            self.shape = shape

        def __getattr__(self, name):
            return getattr(self.shape, name)

        def common(self, obstacle):
            return RetryCut(self.shape.common(obstacle))

        def cut(self, cutters):
            attempts.append(True)
            if len(attempts) == 1:
                if failure == "null_exception":
                    raise ValueError("Null shape")
                return replay.part.makeSolid(replay.part.makeShell(self.shape.Faces[:-1]))
            return self.shape.cut(cutters)

    replay.initial_stock = RetryCut(replay.initial_stock)
    append_feed(replay, (-1.5, 0, -4), (1.5, 0, -4))
    query = ((-2, -2, -3.5), (2, 2, -0.5))
    unrefined, unused_lower = replay._remaining(query, clearance=lambda bound: False)
    assert replay.report["remaining_stock_failed_upper_cuts"] == 1
    assert not replay.stock_regions[0]["applied_inner_offsets"]
    assert replay.stock_regions[0]["applied_endpoints"] == {(1, (-1.5, 0, -4)), (1, (1.5, 0, -4))}
    refined, unused_lower = replay._remaining(query, clearance=lambda bound: False)
    assert refined.Volume < unrefined.Volume
    assert replay.stock_regions[0]["applied_inner_offsets"] == {0}
    assert replay.stock_regions[0]["applied_endpoints"] == {(1, (-1.5, 0, -4)), (1, (1.5, 0, -4))}
    cuts = replay.report["remaining_stock_boolean_cuts"]
    repeated, unused_lower = replay._remaining(query, clearance=lambda bound: False)
    assert repeated.Volume == pytest.approx(refined.Volume)
    assert replay.report["remaining_stock_boolean_cuts"] - cuts == 1


def test_gouge_detection_still_checks_original_faces(replay):
    replay.target = replay.part.makePlane(10, 10, replay.app.Vector(-5, -5, -2))
    replay.target_bounds = safety._bounds(replay.target)
    replay.target_faces = [(face, safety._bounds(face)) for face in replay.target.Faces]
    append_feed(replay, (0, 0, -4), (0, 0, -4))
    shape = replay.shapes[1]["Cutter"]
    result = replay._target_sweep(1, shape, (0, 0, 1), (0, 0, -3), 0)
    assert result["status"] == "collision"


def test_incomplete_stream_and_budget_never_certify_safety(replay):
    replay.deadline = time.monotonic() - 1
    with pytest.raises(safety.SafetyUnavailable, match="budget"):
        replay._subtract(replay.initial_stock, replay.shapes[1]["Cutter"])
    replay.deadline = time.monotonic() + 60
    report = replay.consume([])
    assert report["status"] == "unavailable"
    assert report["collision_free"] is None
    assert not report["tool_events_checked"]
