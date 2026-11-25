from __future__ import annotations

import time

from pss.isaac.bus import BusPublisher, BusSubscriber


def test_obs_and_gt_buses_are_separate():
    publisher = BusPublisher("tcp://127.0.0.1:*", "tcp://127.0.0.1:*")
    obs_sub = BusSubscriber(publisher.obs_addr)
    gt_sub = BusSubscriber(publisher.gt_addr, kind="pull")
    try:
        time.sleep(0.3)  # PUB/SUB slow join
        publisher.publish_obs({"kind": "obs", "step": 7})
        publisher.publish_gt({"kind": "gt", "step": 7, "instances": []})
        obs_message = obs_sub.recv(1000)
        gt_message = gt_sub.recv(1000)
        assert obs_message == {"kind": "obs", "step": 7}
        assert gt_message["kind"] == "gt"
        assert gt_message["instances"] == []
        # neither socket sees the other channel's message
        assert obs_sub.recv(100) is None
        assert gt_sub.recv(100) is None
    finally:
        obs_sub.close()
        gt_sub.close()
        publisher.close()
