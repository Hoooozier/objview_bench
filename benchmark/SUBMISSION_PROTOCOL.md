# Submission protocol v1 — Phase A development implementation

这是评测侧的单 episode 协议，不是已上线的提交网站。它不会构建或执行上传代码，也不提供用户注册、任务队列、排行榜或容器沙箱。暂时只在本地/隔离开发环境使用。

## 文件与信任边界

| 文件 | 职责 |
| --- | --- |
| `submission_protocol.py` | 加锁的状态机、匿名 ID、动作校验、幂等、配额、服务端计时 |
| `submission_gateway.py` | 单 session Bearer 鉴权 HTTP；不公开文件、oracle、指标或异常详情 |
| `submission_interaction.py` | 接入现有 runner；私有记录与可释放观察分离；缓存命中也复制观察 |
| `serve_submission_episode.py` | 只启动可信 benchmark；等待 ready、运行 episode、保存私有汇总 |
| `submission_client.py` | 算法端同步 Python SDK；禁用环境代理和自动重定向 |
| `examples/submission/` | Dockerfile、启动脚本、move/stop 示例和待实现的来源清单模板 |

目录必须属于同一个文件系统，且每次 episode 使用新的空目录：

```text
trusted episode root/             # 仅 benchmark 能访问整个目录
  private/                       # 配置、UID、评分、日志、staging、summary.json
  released/                      # 唯一允许算法只读挂载的目录
    step_000/{rgb.png,depth.npz,mask.png,frame_meta.json}
```

只把 `released/` 子目录以只读方式挂载给算法，不能挂载其父目录。图片先在 `private/staging/` 写完，开始服务端计时后原子重命名到 `released/`，再公布步骤；即使算法监视共享目录，也不能提前获得免费推理时间。若到达视图上限，最后用于评分的观察保留私有，不再开启下一轮推理。不同文件系统无法原子重命名，启动时拒绝。

隐藏数据、render/eval cache、真实 UID、GT、日志、summary 均不挂载给算法。不提供任意观察查询；算法只能获得初始视点与已接受动作对应的观察。匿名化不等于无法从图像识别物体，更不等于阻止长期刷榜，见后续控制措施。

## HTTP 与动作

所有请求（包括未知路径）都要求 `Authorization: Bearer <episode-token>`。token 由可信控制器生成、每 episode 独立，不能使用示例固定值。HTTP 仅用于私有 worker 网络，不要在公网暴露此端口。

| 方法/路径 | 内容 |
| --- | --- |
| `GET /v1/interface` | 协议、动作、配额和计时口径 |
| `GET /v1/episode` | 随机 episode ID、相机参数、坐标系、约束名称、视图预算 |
| `POST /v1/ready` | body 为 `{}`；算法完成模型加载后调用，重复调用不重置计时 |
| `GET /v1/status` | starting / ready / running / done；仅安全状态码 |
| `GET /v1/steps/next?after=-1&wait=10` | 下一轮姿态和相对文件名；无更新返回 204，结束返回 done |
| `POST /v1/feasibility/check` | `{"poses": [[7个数], ...]}`；只返回可行性布尔值 |
| `POST /v1/actions` | move 或 stop；不接受用户耗时、指标、任意附加字段 |

move 示例：

```json
{"episode_id":"服务端随机ID","step_index":0,"request_id":"唯一请求ID","action":"move","pose":[3,0,0,0,0,0,0]}
```

stop 用 `"stop_reason":"plan_end"` 替换 `pose`；还支持 candidate_exhausted、next_view_visited、next_view_infeasible。每步骤最多接受一个动作；网络失败时重发**相同 request_id 和完整 payload**，不会重复推进或重新计时。相同 request_id 不同 payload 返回冲突。SDK `make_action` 生成 payload，调用方保留它以便重试。

JSON 最多 64 KiB、读取最长 5 秒；拒绝重复键、NaN/Infinity、额外字段与非有限姿态。长轮询最多 10 秒；应用同时处理最多 16 个已鉴权请求。可行性请求最多 512 个姿态、每步骤累计 4096 个。无任意渲染、下载、oracle、原始 metric 或 reset 接口。畸形请求不推进步骤且计时继续；不可行 move 终止 episode；默认 120 秒无有效动作则超时。

## 观察与坐标约定

pose 为 `[camera_x,camera_y,camera_z,lookat_x,lookat_y,lookat_z,roll_rad]`，旋转单位为弧度；物体中心原点、归一化到单位球。相机位置和朝向点不能重合。world Z-up；相机 x-right、y-down、z-forward；以 frame metadata 的 camera_to_world/world_to_camera 为准。

RGB 为 uint8 PNG；depth.npz 的 `depth` 为 float32 相机 Z 深度，单位为归一化物体单位，0 为无效；mask.png 为 0/255。frame_meta.json 只有相机内外参和固定 source 标签，不含缓存命中信息、物体名称或路径。目前不导出 PCD；算法可以使用 RGBD 重建点云。

## 时间与结果口径

- 使用 benchmark 的 `perf_counter_ns`，不读取算法上报耗时。
- 每轮从观察原子发布前开始，到动作通过身份/结构校验、进入可行性校验前结束；包含文件读取、推理、SDK、IPC、可行性查询和重试等待。
- 不包含两轮之间的 benchmark 渲染、融合和 metric 计算；包括首步和 stop 决策；相同动作重试不重复累加。
- `official_timing.total_decision_wall_time_ns` 与成功 episode 的 evaluator `total_algorithm_runtime_sec` 对应；超时记录封顶等待时间，失败 episode 不作为有效成绩。
- `model_ready_time_ns` 目前是**协议对象创建到 ready 的时长**，可能包含 benchmark 初始化；不是算法纯模型加载时间，也不是容器冷启动时间。后续控制器单独测量容器启动/就绪时间。
- 旧文件交互的 `algorithm_runtime_sec` 是算法自报，不能混入官方提交榜。新 CLI 禁止 oracle family/算法 shell 命令；统一使用视图 cap，不使用旧算法 family 驱动的 enough-info 提前终止。正式赛道的终止策略需版本化。
- `private/summary.json` 仅供可信调度器读取；`status=failed`（包括 INVALID_ACTION）不能发布为成功成绩。服务退出码非 0 或 summary 缺失也必须判失败。Python watchdog 不能硬中断 GPU/native 调用；控制器必须另设进程/容器硬超时。

## 本地开发运行

HTTP 层测试环境为 Python 3.10；`requirements-submission.txt` 是本机通过测试的兼容版本快照，不是漏洞审计后的生产锁文件。原 benchmark 文档的 Python 3.9 环境尚未与新层联合验收。完整执行还需要原有渲染、PyTorch3D/Open3D、数据与 GPU 环境。

```bash
# repository root; 安装依赖前建议创建独立虚拟环境
pip install -r benchmark/requirements-submission.txt
python benchmark/serve_submission_episode.py --help
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmark/tests -v
```

以下命令中的数据路径均需替换为维护者本机真实路径。令牌通过环境注入（不要写进日志/命令行/仓库）：

```bash
export OBJVIEW_EPISODE_TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
python benchmark/serve_submission_episode.py \
  --uid PRIVATE_UID --obj-path /trusted/object.obj \
  --gt-pointcloud /trusted/gt.pcd --cache-index-json /trusted/cache_index.json \
  --feasibility-json benchmark/configs/feasibility/whole.json \
  --private-dir /tmp/objview-dev-unique/private \
  --observation-root /tmp/objview-dev-unique/released \
  --host 127.0.0.1 --port 8081 --device cpu
```

另一终端/容器注入相同 token、`BENCHMARK_URL=http://127.0.0.1:8081`、`OBSERVATION_ROOT=/tmp/objview-dev-unique/released` 后，仅可在**可信示例开发调试**中执行：

```bash
PYTHONPATH=benchmark python benchmark/examples/submission/algorithm.py
# Docker 示例构建命令；此阶段未在本机实际构建验证
docker build -f benchmark/examples/submission/Dockerfile -t objview-submission-example .
```

不能直接在宿主机或 benchmark 容器执行研究者代码。示例 Dockerfile 仅演示启动约定，浮动 base tag 和依赖尚需构建器解析为固定 digest、生成可信来源清单并扫描。

## 上线前必须完成的控制措施

1. **隔离执行**：独立构建机与运行 worker；用户代码无宿主 Docker socket、无 GT/私有挂载；独立只读根目录、临时空间限额、非 root、cap-drop、no-new-privileges、seccomp、CPU/内存/PID/GPU 限制。普通 Docker 不是恶意代码的完整安全边界；公开注册前需要 VM/microVM 或等效强化隔离。
2. **网络**：算法仅可访问对应 benchmark 网关，禁止外网、宿主、云 metadata 与其他任务；不能把“同一个 Docker network”当作已实现网络白名单。benchmark 端口不发布公网，禁止 IPv6/DNS 等旁路。
3. **反复提取与刷榜**：受邀身份/团队级频率与总预算、源代码/镜像/权重不可变摘要、重复提交检测、每任务新环境、失败也记账；正式评测人工批准、只回聚合结果，算法日志/文件/终止原因不直接回传给研究者。当前进程内计数只约束单 episode，不能跨运行限制提交次数。
4. **发布与审计**：原始结果私有、聚合阈值与延迟发布、异常提交审计、数据/指标/硬件/策略版本；不要拆分或重新分配已发布的 528 个测试对象，400/128 提案暂不执行。
5. **网站**：后续才接邀请登录、上传对象存储、审核与任务队列。github.io 继续静态展示，部署地址确定后增加导航，不嵌入评测控制权限。
