# P1 / W1 notes — 从零到一条 episode

> 开始：2026-11-25
> 依据：`PSS-IsaacSim-实施计划.md` §2、§9（T01–T10）

## 环境

| 项 | 值 |
|---|---|
| PSS Python | 3.13.7（`D:\Python\python.exe`） |
| Isaac Python | 3.11.13（uv venv，`D:\isaac\env`） |
| Isaac Sim | 5.1.0.0（pip / pypi.nvidia.com） |
| Isaac Lab | W1 未安装（W2/W8 需要时再补） |
| GPU | RTX 4070 Laptop 8 GB |
| 驱动 | 616.64 |
| uv | 0.12.2，缓存 `D:\isaac\uvcache` |

没有 conda；用 `uv venv --python 3.11` 建独立环境。安装包：
`isaacsim[all]==5.1.0.0` + `isaacsim[extscache]==5.1.0.0`
（extscache 必须装，否则首次启动会经代理拉扩展并长时间挂起）。

## D1 验收：headless 启动与显存

- 启动命令：`OMNI_KIT_ACCEPT_EULA=YES D:\isaac\env\Scripts\python.exe -m ...`
- 首次启动耗时：TBD
- 空场景显存：TBD
- 最小示例：TBD

## 单条 episode

- 命令：`pss sim run specs/s1_occlusion.yaml --episodes 1 --seed 0 --device dev`
- 物理步数：1200（10 s × 120 Hz）
- 观测：100 行（10 Hz）
- GT：4800 行（1200 物理帧 × 4 实例）
- 耗时：TBD
- bundle：`bundles/dev/s1_occlusion/<hash8>/seed0/`

## 踩到的坑

1. `import pxr` 在裸 Python 下不可用；必须先 `SimulationApp` 启动 Kit。
2. Isaac Sim 首次启动需要接受 EULA；用 `OMNI_KIT_ACCEPT_EULA=YES`。
3. 只装 `isaacsim[all]` 不够：extscache 为空时 Kit 会在线解析扩展，在代理环境下表现为“无输出挂起”。
4. `uv venv` 不装 pip；用 `uv pip` 管理。

## 决策（详见交付时的决策清单）

- 遮挡物用 `scene.occluders`（W1 扩展字段）+ 运动时间采样，而不是 W3 的 fault。
- W1 可见率用解析 ray/box 采样（确定性、零显存），W2 换渲染/PhysX 通路。
- W1 `store_images: false`，先把 bundle 契约跑通；W2 打开相机渲染与检测。
