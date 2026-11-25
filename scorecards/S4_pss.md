# S4-PSS: PSS-in-the-loop pick and place (dev)

- spec: `s4_pss_pick_place_v1`
- plan: `pick -> place` (confidence 1.0)
- goal: `['obj_0001', 'in', 'obj_0003']`
- cup track: `obj_0001`  bin track: `obj_0003`
- actions: reach:ok, grasp:ok, place:ok
- **task_success: True**
- cup: [0.108, -0.077, 0.85]  bin: [0.1, -0.05, 0.8]
- steps: 284  skill wall: 0.53 s

Perception is Isaac GT boxes/poses with class labels (oracle sensor); appearance is zero.
PSS selects the cup/bin tracks, asserts the goal fact, runs the symbolic planner,
and the fixed-base G1 skill layer executes. This tests the PSS-to-physical-action
loop, not the visual detector.
