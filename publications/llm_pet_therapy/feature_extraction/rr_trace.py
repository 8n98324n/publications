"""Trace NeuroKit2 peak edits in an arithmetic coordinate grid."""
import copy,hashlib,importlib,math
from pathlib import Path
import numpy as np
RATE=2**20
VERSION='rr_trace'
PARAMS=dict(c1=.13,c2=.17,alpha=5.2,window_width=91,medfilt_order=11,sampling_rate=RATE)
EXPECTED_NK_VERSION='0.2.13'


def _backend():
    nk = importlib.import_module("neurokit2")
    module = importlib.import_module("neurokit2.signal.signal_fixpeaks")
    if nk.__version__ != EXPECTED_NK_VERSION:
        raise RuntimeError(f"Expected NeuroKit2 {EXPECTED_NK_VERSION}, found {nk.__version__}")
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    return nk, module

def _artifact_n(artifacts):
    return sum(len(artifacts[key]) for key in ("extra", "missed", "ectopic", "longshort"))

def _ticks(nodes):
    return np.array([node["tick"] for node in nodes], dtype=np.int64)

def _original_overlap(original, left, right):
    """One-based original interval IDs with positive-duration overlap."""
    return [i + 1 for i, (a, b) in enumerate(zip(original[:-1], original[1:]))
            if int(a) < right and int(b) > left]

def affected_original_mask(original, corrected):
    """Final boundary-pair metric on the same integer grid."""
    pairs = set(zip(map(int, corrected[:-1]), map(int, corrected[1:])))
    return np.array([(int(a), int(b)) not in pairs
                     for a, b in zip(original[:-1], original[1:])], dtype=bool)

def _longest_run(mask, original):
    best_n = best_ticks = current_n = current_ticks = 0
    for changed, left, right in zip(mask, original[:-1], original[1:]):
        if changed:
            current_n += 1
            current_ticks += int(right) - int(left)
            best_n = max(best_n, current_n)
            best_ticks = max(best_ticks, current_ticks)
        else:
            current_n = current_ticks = 0
    return {"max_consecutive_original_intervals": best_n,
            "max_consecutive_affected_duration_s": best_ticks / RATE}

def _initial_nodes(peaks):
    return [dict(boundary_id=f"original:{i}", tick=int(tick), original_boundary_index=i,
                 edit_dependency_original_interval_ids=[], edit_history_ids=[])
            for i, tick in enumerate(peaks)]

def _context(original, nodes, index):
    # Conservative envelope: threshold rolling window, median rolling window,
    # dRR lag and classification look-ahead. This is explicitly an envelope,
    # not a claim to reconstruct the detector's exact dataflow graph.
    radius = PARAMS["window_width"] // 2 + PARAMS["medfilt_order"] // 2 + 4
    lo, hi = max(0, index - radius - 1), min(len(nodes) - 1, index + radius + 1)
    left, right = nodes[lo]["tick"], nodes[hi]["tick"]
    dependencies = set(_original_overlap(original, left, right))
    for node in nodes[lo:hi + 1]:
        dependencies.update(node["edit_dependency_original_interval_ids"])
    return dict(detector_context_kind="conservative_local_envelope_not_exact_dataflow",
                detector_context_start_s=left / RATE, detector_context_end_s=right / RATE,
                detector_context_original_interval_ids=sorted(dependencies))

def apply_iteration(original, nodes, artifacts, iteration, module=None):
    """Trace exactly one private-library correction call, preserving node IDs.

    Semantic extra/missed edits change count; ectopic/longshort are relocation
    operations even though NumPy implements them with delete + concatenate.
    Per-operation equality checks prevent a tracing implementation from silently
    substituting a different correction algorithm.
    """
    if module is None:
        _, module = _backend()
    nodes = copy.deepcopy(nodes)
    start_peaks = _ticks(nodes)
    work = {key: list(map(int, artifacts[key]))
            for key in ("extra", "missed", "ectopic", "longshort")}
    events, steps = [], []
    ever_mask = np.zeros(len(original) - 1, dtype=bool)
    for category in ("extra", "missed", "ectopic", "longshort"):
        indices = work[category]
        if not indices:
            continue
        before_nodes = copy.deepcopy(nodes)
        before = _ticks(nodes)
        if len(set(indices)) != len(indices):
            raise RuntimeError("Duplicate edit indices require explicit identity review")
        if any(i < 0 or i >= len(nodes) for i in indices):
            raise RuntimeError("Out-of-range artifact index")
        if category == "extra":
            valid = set(indices)
            expected = module._correct_extra(indices, before.copy())
        elif category == "missed":
            valid = {i for i in indices if 1 < i < len(nodes)}
            expected = module._correct_missed(indices, before.copy())
        else:
            valid = {i for i in indices if 1 < i < len(nodes) - 1}
            expected = module._correct_misaligned(indices, before.copy())
        replacements, insertions = {}, {}
        for position, index in enumerate(indices):
            node = before_nodes[index]
            event_id = f"iteration:{iteration}/{category}:{position}"
            left_index, right_index = max(0, index - 1), min(len(nodes) - 1, index + 1)
            if category == "missed":
                right_index = index
            left, right = int(before[left_index]), int(before[right_index])
            deps = set(_original_overlap(original, left, right))
            for neighbour in before_nodes[left_index:right_index + 1]:
                deps.update(neighbour["edit_dependency_original_interval_ids"])
            valid_operation = index in valid
            new_tick = None
            if valid_operation and category in ("missed", "ectopic", "longshort"):
                new_tick = int(left + (right - left) / 2)
            actual_edit = valid_operation and (category in ("extra", "missed") or new_tick != node["tick"])
            semantic = ("deletion" if category == "extra" else "insertion" if category == "missed" else "internal_relocation")
            ev = dict(event_id=event_id, iteration=iteration, category=category,
                      index_in_current_operation=index, input_boundary_id=node["boundary_id"],
                      old_tick=node["tick"], new_tick=new_tick,
                      semantic_operation=semantic, operation_accepted_by_backend=valid_operation,
                      actual_edit=actual_edit,
                      no_edit_reason=("backend_boundary_filter" if not valid_operation else "same_coordinate" if not actual_edit else ""),
                      local_support_start_s=left / RATE, local_support_end_s=right / RATE,
                      local_original_interval_ids=sorted(deps),
                      touches_record_endpoint=(index in (0, len(before_nodes) - 1)),
                      **_context(original, before_nodes, index))
            if valid_operation and category == "missed":
                new_id = f"inserted:{iteration}:{position}"
                insertions[index] = dict(boundary_id=new_id, tick=new_tick, original_boundary_index=None,
                                         edit_dependency_original_interval_ids=sorted(deps), edit_history_ids=[event_id])
                ev["output_boundary_id"] = new_id
            elif valid_operation and category not in ("extra", "missed"):
                replacement = copy.deepcopy(node)
                replacement["tick"] = new_tick
                if actual_edit:
                    replacement["edit_dependency_original_interval_ids"] = sorted(deps)
                    replacement["edit_history_ids"] = sorted(set(node["edit_history_ids"] + [event_id]))
                replacements[index] = replacement
                ev["output_boundary_id"] = node["boundary_id"]
            else:
                ev["output_boundary_id"] = None if valid_operation else node["boundary_id"]
            events.append(ev)
        if category == "extra":
            nodes = [node for i, node in enumerate(before_nodes) if i not in valid]
        elif category == "missed":
            nodes = []
            for i, node in enumerate(before_nodes):
                if i in insertions:
                    nodes.append(insertions[i])
                nodes.append(node)
        else:
            # Match np.delete + np.concatenate + stable sort, including ordering
            # of equal coordinates (which will subsequently fail positivity).
            nodes = [node for i, node in enumerate(before_nodes) if i not in valid]
            nodes.extend(replacements[i] for i in indices if i in valid)
            nodes.sort(key=lambda node: node["tick"])
        after = _ticks(nodes)
        if not np.array_equal(after, expected):
            raise AssertionError(f"Trace differs from backend at {iteration}/{category}")
        ever_mask |= affected_original_mask(original, after)
        steps.append(dict(iteration=iteration, category=category,
                          before_boundary_ids=[n["boundary_id"] for n in before_nodes],
                          before_ticks=before.tolist(), after_boundary_ids=[n["boundary_id"] for n in nodes],
                          after_ticks=after.tolist(), matches_backend=True))
        if category == "extra":
            for key in ("missed", "ectopic", "longshort"):
                work[key] = module._update_indices(indices, work[key], -1)
        elif category == "missed":
            # Match the backend exactly: its index update uses the entire input
            # missed list, including items filtered by _correct_missed.
            for key in ("ectopic", "longshort"):
                work[key] = module._update_indices(indices, work[key], 1)
    reference = module._correct_artifacts(copy.deepcopy(artifacts), start_peaks.copy())
    if not np.array_equal(_ticks(nodes), reference):
        raise AssertionError("Traced iteration differs from _correct_artifacts")
    return nodes, events, steps, ever_mask

def trace_correction(rr_seconds):
    """Correct one nonempty finite positive RR sequence; no group data read."""
    rr = np.asarray(rr_seconds, dtype=float)
    if rr.ndim != 1 or len(rr) == 0 or not np.all(np.isfinite(rr)) or np.any(rr <= 0):
        raise ValueError("RR must be a nonempty one-dimensional finite positive sequence")
    nominal = np.r_[0.0, np.cumsum(rr)]
    original = np.rint(nominal * RATE).astype(np.int64)
    if np.any(np.diff(original) <= 0):
        raise ValueError("Arithmetic grid collapses an original interval")
    nk, module = _backend()
    nodes = _initial_nodes(original)
    detections, events, steps, iterations = [], [], [], []
    ever = np.zeros(len(rr), dtype=bool)
    previous_n = None
    for iteration in range(1000):
        peaks = _ticks(nodes)
        artifacts, _ = module._find_artifacts(peaks, **PARAMS)
        n = _artifact_n(artifacts)
        applied = previous_n is None or n < previous_n
        for category, indices in artifacts.items():
            for index in indices:
                detections.append(dict(iteration=iteration, category=category, index_in_iteration=int(index),
                                       boundary_id=nodes[index]["boundary_id"], tick=int(peaks[index]),
                                       correction_iteration_applied=applied,
                                       **_context(original, nodes, index)))
        iterations.append(dict(iteration=iteration, artifact_n=n, input_boundary_n=len(nodes), applied=applied))
        if not applied:
            break
        nodes, ev, st, touched = apply_iteration(original, nodes, artifacts, iteration, module)
        events.extend(ev)
        steps.extend(st)
        ever |= touched
        previous_n = n
    else:
        raise RuntimeError("Correction did not reach published stop rule")
    corrected = _ticks(nodes)
    _, reference = nk.signal_fixpeaks(original.copy(), iterative=True, method="Kubios", show=False, **PARAMS)
    if not np.array_equal(corrected, reference):
        raise AssertionError("Traced correction differs from public signal_fixpeaks")
    if np.any(np.diff(corrected) <= 0):
        raise ValueError("Correction produced nonpositive intervals")
    affected = affected_original_mask(original, corrected)
    actual = [ev for ev in events if ev["actual_edit"]]
    counts = {kind: sum(ev["semantic_operation"] == kind for ev in actual)
              for kind in ("insertion", "deletion", "internal_relocation")}
    endpoints = bool(corrected[0] == original[0] and corrected[-1] == original[-1])
    return dict(core_version=VERSION, neurokit_version=nk.__version__, algorithm_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                parameters=PARAMS.copy(), arithmetic_tick_rate_not_ecg_sampling_rate=RATE,
                input_raw_rr_seconds=rr.tolist(), original_ticks=original.tolist(), corrected_ticks=corrected.tolist(),
                original_rr_duration_s=math.fsum(rr.tolist()), corrected_support_duration_s=(int(corrected[-1]) - int(corrected[0])) / RATE,
                maximum_coordinate_rounding_error_s=float(np.max(np.abs(original / RATE - nominal))),
                original_interval_n=len(rr), corrected_interval_n=len(corrected) - 1,
                net_interval_count_change=len(corrected) - len(original), endpoints_preserved=endpoints,
                span_preserved=bool(corrected[-1] - corrected[0] == original[-1] - original[0]),
                original_intervals_affected_n=int(affected.sum()), correction_fraction=float(affected.mean()),
                final_affected_original_interval_ids=(np.flatnonzero(affected) + 1).tolist(),
                ever_modified_original_interval_ids=(np.flatnonzero(ever) + 1).tolist(),
                ever_edit_dependency_original_interval_ids=sorted(set().union(
                    *(set(ev["local_original_interval_ids"]) for ev in actual))) if actual else [],
                ever_modified_fraction=float(ever.mean()),
                ever_modified_definition="union of original exact coordinate pairs absent after any applied category operation",
                longest_final_affected_run=_longest_run(affected, original),
                edit_counts=counts,
                pure_internal_relocation_history=bool(counts["insertion"] == 0 and counts["deletion"] == 0
                                                     and endpoints and all(not ev["touches_record_endpoint"] for ev in actual)),
                nodes=nodes, events=events, operation_steps=steps, detections=detections, iterations=iterations,
                final_residual_artifact_n=iterations[-1]["artifact_n"], wrapper_matches_public_implementation=True,
                human_review_status="automated_check_only")

def _metrics(nn_ms):
    nn = np.asarray(nn_ms, dtype=float)
    return dict(mean_hr=float(60000 / nn.mean()) if len(nn) else None,
                rmssd=float(np.sqrt(np.mean(np.diff(nn) ** 2))) if len(nn) >= 2 else None,
                sdnn=float(np.std(nn, ddof=1)) if len(nn) >= 2 else None)

def _quality_le(numerator, denominator, percent):
    # Integer comparison: no rounded display percentage and no float boundary.
    return bool(denominator > 0 and 100 * numerator <= percent * denominator)
