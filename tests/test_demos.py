from pss.demos import laundry_task_demo, semantic_space_demo, token_vs_vector_reasoning


def test_semantic_space_demo():
    r = semantic_space_demo.run(verbose=False)
    assert r["all_correct"]
    assert r["objects"]["obj0"]["top_concept"] == "苹果"
    assert r["objects"]["obj1"]["top_concept"] == "杯子"
    for oid in ("obj0", "obj1"):
        assert set(r["objects"][oid]["top_attrs"]) == set(r["objects"][oid]["features"])
        assert r["objects"][oid]["concept_ok"]
        assert r["objects"][oid]["attrs_ok"]


def test_token_vs_vector_demo():
    r = token_vs_vector_reasoning.run(verbose=False)
    assert r["answers_match"]
    assert r["first_collision_t"] == 2
    assert r["token_counts"] == sorted(r["token_counts"])
    assert r["token_counts"][0] < r["token_counts"][1] < r["token_counts"][2]
    assert r["vector_ops"] == 6
    assert r["vector_rows"][0]["hit"] is False
    assert r["vector_rows"][1]["hit"] is True
    assert r["vector_rows"][2]["hit"] is False


def test_laundry_demo():
    r = laundry_task_demo.run(verbose=False)
    assert r["goal_reached"]
    assert r["steps"] == 12
    assert all(row["ok"] for row in r["log"])
    assert r["rejection"]["ok"] is False
