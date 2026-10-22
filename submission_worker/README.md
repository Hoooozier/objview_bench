# Phase B：可信 Docker smoke 执行器（开发版）

此模块是可信 worker 的本地生命周期控制器，不是上传服务。**只允许管理员自己构建、检查过的仓库 smoke 镜像**。`--trusted-smoke` 和镜像 label 是防误用措施，不是能抵抗恶意用户的信任认证；Web/API 不能直接把用户参数交给这里。

当前用户已选择暂缓 Docker 安装。可继续运行所有 CPU/协议测试和 Docker 命令契约测试；双容器实测、安全隔离验收保持“未验证”，不会用模拟结果代替。

## 已实现

- 状态：PREPARING → STARTING → RUNNING → SUCCEEDED/FAILED，报告原子写入新建的私有目录。
- 对已有本地镜像做 inspect，验证角色 label、拒绝隐式 VOLUME，把 tag 一次解析为不可变 `sha256:` image ID；运行阶段 `--pull never`。不构建/下载/执行上传的 Dockerfile。
- 创建每任务独立内部网络和 named volume；一个固定可信 init 容器创建目录并交给 UID/GID 65532，随后启动 benchmark 与 algorithm 两个独立非 root 容器。
- benchmark 挂整个 `/episode` 卷，算法**只读挂载 `released` 子目录**，不挂父目录、GT、日志、Docker socket、主机目录。private staging 与 released 在同一卷，可继续使用计时开始后的原子观察发布。
- 两个运行容器只读根文件系统、cap-drop ALL、no-new-privileges、CPU=1、内存/含 swap 总额=512 MiB、PID=128、打开文件=1024、`/tmp` 64 MiB tmpfs、禁用 Docker logs；默认 Docker seccomp，不发布宿主端口。
- 整个准备/启动/运行过程共享 `--timeout-sec` 预算。单条 Docker 命令最长 10 秒且不超剩余预算。失败后定向强制删除本次容器，再删除网络/卷；清理有独立每命令 5 秒预算，所以最终退出可晚于任务截止时间。
- 记录创建意图后才调用 Docker，处理“服务端创建成功但 CLI 返回失败”的情况。删除前与删除后核对资源及 job label；外来资源不删除；清理无法确认时报告失败和明确资源名。无全局 prune，无自动失败重试，不删除镜像。
- 只拷贝可信 benchmark 的 summary，验证 job ID、smoke 标签、完成状态、动作数量对应的视图数和计时结构；报告只保留白名单计时字段。任务 token 文件与原始合成 summary 随后删除。报告不含分数、token、原始算法输出或观察。

## 本机测试（不需要 Docker）

从仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 -m unittest discover -s benchmark/tests -v
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 benchmark/smoke_submission.py
python3 -m submission_worker.controller --help
```

新增 `test_submission_container.py` 用真实子进程启动**与镜像相同的 benchmark 入口**，再运行真实 dummy 算法，通过 localhost HTTP 完成观察、move、stop、评分调用和私有汇总保存。此测试不检查算法质量/科学指标数值，也不验证 Docker 隔离。

`test_worker_*.py` 使用有状态 Docker 替身验证命令契约、错误分类、deadline、镜像引用、报告字段和清理行为。这些测试不能证明 Docker 实际接受了所有选项或落实了内核限制。

## Docker 安装与运行条件

宿主机是 Ubuntu 22.04，无 Docker；`sudo -n true` 显示需要密码。不要把密码交给代理。用户已选择以后再安装，因此本轮没有更改系统包、用户组或服务。

后续优先使用 rootless Docker。其前置依赖包括 `newuidmap/newgidmap` 和对应 subordinate UID/GID 映射；cgroup 资源限制需要 systemd/cgroup v2。参见 [Docker rootless 安装](https://docs.docker.com/engine/security/rootless/)和[资源限制说明](https://docs.docker.com/engine/security/rootless/tips/#limiting-resources)。控制器拒绝缺少内存/PID 控制或不支持的 rootless cgroup 配置；Docker 输出 WARNING 时也失败，不假装限制生效。真实 CPU 限制仍需专门实测。

用户在宿主机安装前置包后，可继续让代理完成 Docker 安装：

```bash
sudo apt-get update
sudo apt-get install -y uidmap dbus-user-session slirp4netns
```

若采用 system-wide Docker，则按 [Docker 官方 Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/)操作；不要为方便而开放 TCP daemon 或使用 privileged Docker-in-Docker。控制器只允许显式本地 Unix socket，忽略环境中的 `DOCKER_HOST/CONTEXT`，不会误操作远程 Docker。

## Docker 可用后的双容器验收（当前未执行）

需支持 `--mount volume-subpath` 的 Docker Engine；算法挂载的子目录必须先存在。参见 [Docker volume subdirectory](https://docs.docker.com/engine/storage/volumes/#mount-a-volume-subdirectory)。若版本或权限不支持，直接失败；不能降级成把整个卷挂给算法。

```bash
# 仓库根目录；这里的 Dockerfile 仅构建可信 CPU/dummy 开发镜像
docker build -f benchmark/containers/Dockerfile.smoke -t objview-benchmark-smoke:dev .
docker build -f benchmark/examples/submission/Dockerfile -t objview-algorithm-smoke:dev .

# 先创建唯一父目录，job 子目录必须不存在
worker_run_dir=$(mktemp -d /tmp/objview-worker.XXXXXX)
python3 -m submission_worker.controller \
  --trusted-smoke \
  --benchmark-image objview-benchmark-smoke:dev \
  --algorithm-image objview-algorithm-smoke:dev \
  --docker-host unix:///run/user/1000/docker.sock \
  --output-dir "$worker_run_dir/job" --timeout-sec 30
```

把 socket 路径改成实际用户 ID 的 rootless socket；system-wide daemon 是 `unix:///var/run/docker.sock`，但需要相应授权。不要擅自 chmod 666 socket。

退出 0、`report.json` 为 SUCCEEDED、`cleanup.complete=true` 才表示这次双容器 smoke 通过。必须保存 resolved image IDs、Docker/cgroup 信息、状态/耗时/清理证据；标签固定为 `mode=smoke_test`、`official_result=false`、`leaderboard_eligible=false`，不可发布为正式成绩。

根目录 `.dockerignore` 只允许两个 smoke Dockerfile 所需源码进入上下文，排除测试数据/cache/Git/本机报告。基础镜像 tag、pip 依赖快照尚非生产供应链锁定；正式构建器需要另行固定 digest、扫描和生成 provenance。

## 时间口径与安全限制

- `algorithm_start_call_ns`：控制器调用 `docker start` 的耗时；不是模型加载耗时。
- `algorithm_container_wall_time_ns`：从调用 start 前到观察到算法退出；包括 CLI/轮询延迟和 benchmark 等待，不是 Planning Runtime。
- `total_job_wall_time_ns`：包括准备、运行与清理。
- `benchmark_timing`：原协议在 benchmark 本地测得的每轮 decision time；不相信算法上报。
- 模型就绪的独立控制器测量尚未实现；原协议 ready 时间仍包含 benchmark 准备，不能混充容器启动时间。
- 内部 Docker network **不等于**完整“只能访问指定 gateway”的防火墙，Docker 内置 DNS 仍可能可用；没有 egress/DNS/宿主访问 canary 验收。没有卷硬磁盘配额、GPU 限制验证或 VM/microVM 边界。
- 容器创建前的镜像 label 可以被伪造；本地 operator 必须知道这些镜像是可信仓库构建产物。不得接收外部镜像、隐藏集或真实提交。
- SIGKILL、宿主断电或 daemon 无响应不能保证 finally 清理成功；报告的 `org.objview.job` 标签可用于管理员查找遗留资源，不能使用无筛选的 prune。持久化恢复清理/reaper 属于后续工作。

本阶段 worker 源码暂存 benchmark 仓库的 `submission_worker/`，只作独立 CLI；后续接入独立 `objview_submit` 控制面时通过任务契约迁移/复用，Web 进程始终不得持有 Docker socket。
