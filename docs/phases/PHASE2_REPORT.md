# Phase 2 报告：仿真环境与数据管线

> 日期：2026-11-20
> 对应计划：`公共语义空间-落地执行计划-v1.md` → Phase 2
> 状态：**已完成（v1，原始代理资产）**
> 环境：MuJoCo 4.0.1 / Python 3.13.7 / numpy 2.3.2 / pytest 9.1.1

---

## 1. 交付物

| 交付 | 路径 | 状态 |
|---|---|---|
| 场景构建（8 类原始几何体 + 容器 + 遮挡物） | `pss/sim/scene.py` | ✅ |
| 传感器（RGB / 深度 / 分割 / 位姿 / 接触 / 关系） | `pss/sim/sensors.py` | ✅ |
| 动作代理（push / lift / place / wait） | `pss/sim/actions.py` | ✅ |
| 数据生成（类别级 / episode / 遮挡序列） | `pss/sim/generator.py` | ✅ |
| 存储与索引（`.npz` + `index.json` + 数据卡） | `pss/sim/dataset.py` | ✅ |
| 生成 CLI | `pss/sim/scripts/generate_dataset.py` | ✅ |
| Dataset v1（small） | `data/v1/` | ✅ 46 条，5.3 MB |
| Dataset v1（tiny，快速验证） | `data/v1_tiny/` | ✅ 22 条，1.6 MB |
| 测试 | `tests/test_sim.py`、`tests/test_dataset.py` | ✅ 44 passed |

---

## 2. Dataset v1

生成命令：

```bash
python -m pss.sim.scripts.generate_dataset --out data/v1 --scale small --seed 0
```

| 类型 | 数量 | 内容 |
|---|---:|---|
| category | 24 | 8 类 × 3 实例，每样本 3 个随机视角 |
| episode | 16 | 4 物体，4 步动作（初始帧 + 3 个动作后帧） |
| occlusion | 6 | 3 物体，8 帧，遮挡物横扫 |

| Split | 数量 |
|---|---:|
| train | 18 |
| val | 10 |
| test | 12 |
| test_category | 6 |

切分规则：

- 留出类别：`cloth`、`tool` → `test_category`
- 其余类别：实例 0 → train，1 → val，2 → test
- episode / occlusion：按索引 60% / 20% / 20%

### 数据模式

**category**（`.npz`）：

| key | shape | dtype |
|---|---|---|
| rgb | (V, H, W, 3) | uint8 |
| depth | (V, H, W) | float32 |
| segmentation | (V, H, W, 2) | int32（[geom id, object type]） |
| tactile | (V, 8, 8) | float32（材质合成代理） |
| pose | (7,) | float32 |
| meta_json | scalar | category / instance / material / camera / split |

**episode**（`.npz`）：

| key | shape | dtype |
|---|---|---|
| rgb / depth / segmentation | (T, ...) | 同上 |
| tactile | (T, N, 8, 8) | float32（接触导出） |
| poses | (T, N, 7) | float32 |
| meta_json | scalar | categories / instance_ids / actions / relations / camera / geom map |

**occlusion**（`.npz`）：与 episode 相同，另有 `occluder_positions`（T, 3）。

---

## 3. 场景与传感器

| 项 | 设定 |
|---|---|
| 桌面 | 1.2 × 0.9 m 灰色台面，下方有地面兜底 |
| 容器 | 2 个开口盒子（bin0 / bin1），用于 `in` 关系 |
| 物体 | 8 类原始几何体：杯/盘/盒/瓶/布/积木/玩具水果/工具 |
| 相机 | MuJoCo free camera，方位角 ±50°，俯角 -42°～-24°，距离 0.95–1.25 m |
| 光照 | 方向光，位置与强度随机 |
| RGB | 128×128 uint8 |
| 深度 | 128×128 float32（米） |
| 分割 | 128×128×2 int32：geom id + object type（5 = geom） |
| 触觉 | category 为材质参数合成；episode 为接触点投影到 8×8 网格（排除桌面/地面支撑） |
| 关系 | `in`（容器）、`contact`（物体间）、`near`（<0.12 m） |

视觉检查：`data/sample_scene.png`、`data/dataset_montage.png` 已人工确认桌面、容器、物体、遮挡条渲染正常；分割通道能正确区分 table / floor / object。

---

## 4. 验证

```
python -m pytest -q
............................................  [100%]
44 passed
```

新增覆盖：

- 场景构建、RGB/深度/分割形状与 dtype
- 位姿读取、接触导出、触觉图
- push 动作确实改变物体位置
- category / episode / occlusion 三类样本的数组形状
- `.npz` 保存-加载往返（不依赖 pickle）

---

## 5. 设计决策

| # | 决策 | 理由 |
|---|---|---|
| 1 | **原始几何体，不下载外部网格** | 无资产授权/下载步骤，可复现；足够验证 slot/关系/遮挡 |
| 2 | **category 触觉用材质参数合成** | RD2 真实触觉尚未采集；先打通链路，明确标注代理 |
| 3 | **episode 触觉用接触力投影** | 比纯材质合成更接近真实接触，但仍为粗粒度 8×8 |
| 4 | **lift/place 为运动学代理** | 没有抓取控制器；先产出动作-状态对，供世界模型阶段使用 |
| 5 | **每个 episode 固定相机** | 遮挡序列只动遮挡物，便于后续 ID 追踪评测 |
| 6 | **按实例与类别双重留出** | 同时测实例泛化与类别泛化 |
| 7 | **元数据存进 `.npz` 的 JSON 字符串** | 加载不需要 `allow_pickle=True` |

---

## 6. 已知边界（必须在 Phase 3/4 前记住）

| 边界 | 影响 |
|---|---|
| 原始几何体、无真实纹理 | 视觉专家学到的是形状/颜色，不是真实材质；sim2real 差距大 |
| 合成触觉 | 触觉专家只能验证管线，不能证明真实触觉可分性 |
| 无变形、无关节容器 | 布料是刚性薄盒；洗衣机/门等长时序对象未建模 |
| lift/place 非物理抓取 | 世界模型学到的“抓取”是瞬移，不是接触动力学 |
| 单一固定相机 / episode | 多视角覆盖不足；遮挡序列只动遮挡物 |
| 无传感器噪声模型 | 加入噪声/域随机化后才能谈鲁棒性 |
| 数据量小（small） | 只够打通管线；训练专家至少要到 `full` 或更大规模 |

---

## 7. 下一步：Phase 3（感官专家）

Phase 3 的目标：从 Dataset v1 训练/封装三个冻结专家。

| 专家 | v1 方案 | 输出维度 | 备注 |
|---|---|---|---|
| 视觉 | 真实图像预训练骨干（ResNet/DINOv2）+ 物体裁剪 + 轻量 head | 256 | 先做线性探针，不追求精度 |
| 触觉 | 8×8 接触图上的小 MLP / 1D-CNN | 64 | 仅验证管线，合成触觉边界已知 |
| 几何 | 深度 + 分割 → 包围盒尺寸/高度/曲率统计（手工特征） | 32 | 先用可解释特征，避免与学习误差混淆 |

验收：输出形状、确定性、冻结导出；线性探针在留出实例/类别上的结果**如实报告为管线冒烟结果**，不作为能力结论。
