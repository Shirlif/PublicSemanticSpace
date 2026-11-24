import numpy as np

from pss.rd2.t1_diagnostics import T1Data, diagnose


def _make_data(signal: float = 1.0, n_objects: int = 8, per_object: int = 5, d: int = 16):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(n_objects, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    rows, ids = [], []
    for i in range(n_objects):
        for _ in range(per_object):
            v = centers[i] + (1.0 - signal) * rng.normal(size=d) * 0.3
            rows.append(v)
            ids.append(f"obj{i}")
    return T1Data({"vision": np.stack(rows)}, np.asarray(ids))


def test_strong_instance_signal_is_detected():
    result = diagnose(_make_data(signal=1.0))
    assert result["fused"]["instance_nn"] > 0.9
    assert result["fused"]["instance_gap"] > 0.2
    assert result["verdict"].startswith("strong")


def test_random_features_are_weak():
    rng = np.random.default_rng(1)
    n = 40
    data = T1Data(
        {"vision": rng.normal(size=(n, 16))},
        np.asarray([f"obj{i % 8}" for i in range(n)]),
    )
    result = diagnose(data)
    assert result["fused"]["instance_nn"] < 0.5
    assert not result["verdict"].startswith("strong")


def test_partial_verdict_when_one_modality_is_weak():
    rng = np.random.default_rng(3)
    n_objects, per_object, d = 8, 5, 16
    centers = rng.normal(size=(n_objects, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    vision, tactile, ids = [], [], []
    for i in range(n_objects):
        for _ in range(per_object):
            vision.append(centers[i] + 0.01 * rng.normal(size=d))
            tactile.append(rng.normal(size=d))
            ids.append(f"obj{i}")
    data = T1Data(
        {"vision": np.stack(vision), "tactile": np.stack(tactile)},
        np.asarray(ids),
    )
    result = diagnose(data)
    assert result["verdict"].startswith("partial")
    assert "tactile" in result["verdict"]


def test_cross_session_reported():
    rng = np.random.default_rng(2)
    centers = rng.normal(size=(4, 8))
    rows, ids, sessions = [], [], []
    for i in range(4):
        for s in range(2):
            rows.append(centers[i] + 0.01 * rng.normal(size=8))
            ids.append(f"obj{i}")
            sessions.append(f"s{s}")
    data = T1Data(
        {"vision": np.stack(rows)},
        np.asarray(ids),
        session_ids=np.asarray(sessions),
    )
    result = diagnose(data)
    assert result["fused"]["cross_session"]["available"]
    assert result["fused"]["cross_session"]["pairs"] > 0
