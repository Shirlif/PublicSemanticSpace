# S4_mobile 附表（阻塞）

- 状态：**未跑**。
- 原因：Isaac Lab 5.1 官方 Policies 列表中没有 Unitree G1 的 locomotion checkpoint；
  当前 `freeze.json` 的 `locomotion_policy = none`。
- `SkillExecutor.walk_to` 返回 `unsupported`；`base_mobile` 不做任何数字。
- 需要：安装/获取 G1 行走策略（RSL-RL / Isaac Lab checkpoint），冻结版本后再开附表。
