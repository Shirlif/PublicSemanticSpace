# S4-PSS relation: cup left of bowl (dev)

- spec: `s4_pss_relation_v1`
- plan: `pick -> place`
- goal: `['obj_0001', 'in', 'obj_0004']`
- **task_success: True**
- selection: `left_of(obj_0003)` -> `/World/Assets/cup_left`
- actions: reach:ok, grasp:ok, place:ok
- cup: [0.032, -0.076, 0.85]  bin: [0.02, -0.05, 0.8]

Perception is Isaac GT boxes/poses with class labels; no GT ids are used for association.
