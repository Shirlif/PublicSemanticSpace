import numpy as np

from pss.observations import make_episode, make_large_scene, make_scene


def test_scene_observations_are_contract_valid():
    scene = make_scene(8, dim=128, seed=0)
    obs = scene.observations()
    assert len(obs) == 8
    for item in obs:
        assert item.gt_id and item.gt_category
        assert item.appearance.shape == (128,)
        assert np.isclose(np.linalg.norm(item.appearance), 1.0, atol=1e-5)
        assert set(item.raw) == {"vision", "tactile", "geometry"}
        for vec in item.raw.values():
            assert vec.shape == (128,)
            assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_episode_occlusion_keeps_at_least_one_entity():
    scene, frames = make_episode(
        n_entities=10, n_frames=6, occlusion_prob=0.7, dim=128, seed=1
    )
    assert len(frames) == 6
    for frame in frames:
        assert len(frame) >= 1
    # Some frames should hide at least one entity.
    assert any(len(frame) < 10 for frame in frames)


def test_large_scene_generates_relations():
    scene = make_large_scene(n_entities=50, n_relations=200, dim=128, seed=2)
    assert len(scene.entities) == 50
    assert len(scene.relations) == 200
    assert all(len(rel) == 3 for rel in scene.relations)
