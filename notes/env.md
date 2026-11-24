# Isaac 环境基线（D1）

> 2026-11-25，RTX 4070 Laptop 8 GB，驱动 616.64

## 版本

| 项 | 值 |
|---|---|
| 安装方式 | `uv venv`（Python 3.11.13）+ pip/uv 安装 |
| Isaac Sim | 5.1.0.0（`isaacsim[all]` + `isaacsim[extscache]`） |
| Kit | 107.3（随 5.1） |
| 渲染后端 | **D3D12**（`--/app/vulkan=false`） |
| 扩展缓存 | `isaacsim/extscache` 6.3 GB，433 个扩展 |

## 为什么用 D3D12

在驱动 616.64 + RTX 4070 Laptop 上，Kit 默认 Vulkan 后端在 RTX scene DB 初始化时
稳定崩溃（`rtx.scenedb.plugin.dll`，三次复现）。加 `--/app/vulkan=false` 切到
D3D12 后 11 s 完成启动。该参数写入 `pss/isaac/launch.py`，可用
`PSS_ISAAC_EXTRA_ARGS` 覆盖；两台评测机必须一致，已写入 `freeze.json`。

## 启动与物理

| 指标 | 值 |
|---|---|
| 冷启动（首次，含扩展扫描） | ~20 s |
| 热启动（app ready） | ~8.6 s |
| `SimulationApp` 构造 + app ready | 11.2 s |
| 空场景 120 物理步（1/120 s） | 0.03 s |
| 单物理步 | 0.268 ms |
| 120 步后模拟时间 | 1.0167 s（含 play 的一次内部步进） |
| 单进程显存 | WDDM 下 `nvidia-smi --query-compute-apps` 不报告（N/A） |

> 8 GB 显存下 W1 空场景无压力；G1 + 相机渲染要到 W2/W8 再测并行度上限。

## 网络/安装踩坑

- `pypi.nvidia.com` 单连接被限速到 ~60–90 KB/s；3.4 GB 的 extscache wheel
  用 128 路 HTTP Range 并行下载到 ~5.5 MB/s，再本地安装。
- 只装 `isaacsim[all]` 时 extscache 为空，Kit 会在线拉扩展；在代理环境下表现为
  长时间无输出。`isaacsim[extscache]` 必须装（或从缓存解压）。
- `isaacsim-extscache-kit-sdk` 提供 `omni.sensors.nv.*`；不装会在启动时在线拉取
  ~35 s。已从 uv 缓存补装。
- 运行 Kit 时清空 `HTTP(S)_PROXY`/`ALL_PROXY` 并设 `NO_PROXY=*`，避免代理路由
  导致扩展解析挂起。

## 命令

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
$env:NO_PROXY = "*"
D:\isaac\env\Scripts\python.exe -m pss sim build specs\s1_occlusion.yaml
D:\isaac\env\Scripts\python.exe -m pss sim run specs\s1_occlusion.yaml --episodes 1 --seed 0 --device dev
```
