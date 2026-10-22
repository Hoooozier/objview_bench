# ObjView-Bench 算法提交与隐藏评测平台设计

- 日期：2026-10-21
- 状态：设计基线已确认；2026-10-22 补充本机链路验收要求
- 适用阶段：受邀研究者试运行；架构兼容后续公网注册
- 关联仓库：`objview_bench`、`objview.github.io`，以及后续独立建立的 `objview_submit`

## 1. 背景与目标

ObjView-Bench 已经具备单 episode 执行、观察发布、算法动作提交、可达性检查、渲染、指标计算和多 episode 汇总能力。平台建设的目标不是重写 benchmark，而是在现有评测引擎外增加安全的提交、构建、调度、隐藏评测和排行榜能力。

平台需要满足以下目标：

1. `objview.github.io` 继续作为静态论文主页，只增加到算法提交网站的导航链接。
2. 受邀用户提交固定版本的源码、Dockerfile 和启动脚本，由平台构建不可变镜像。
3. 用户算法容器与可信 benchmark 容器分离，通过版本化协议完成逐步观察与动作交互。
4. 隐藏几何、隐藏集标识、逐对象指标和评分内部状态不向算法或用户泄漏。
5. 官方耗时全部由 benchmark/controller 本地测量，不采用用户上报时间。
6. 开发阶段支持人工审核和单个专用 worker；未来可迁移到多 worker、Kubernetes 和更强沙箱。
7. 每项正式结果能够由镜像 digest、benchmark commit、数据版本、缓存版本和硬件配置复现。

## 2. 非目标

首个可用版本不包含：

- 公网匿名注册和自动开放评测。
- 用户自定义 Docker 运行参数、特权模式或主机目录挂载。
- 训练任务托管。
- 用户在隐藏集上下载逐对象结果、原始日志或运行产物。
- 多云调度、自动扩缩容和跨区域容灾。
- 实时对战式排行榜。

## 3. 方案选择

采用“控制面与执行面分离”的架构。

开发阶段使用：

- 独立 Web/API 控制面；
- 独立 rootless BuildKit 构建 worker；
- 一台或数台专用 GPU evaluation worker；
- 每次正式 episode 创建一套临时 benchmark/algorithm 容器环境。

公网阶段保持提交格式、API 和数据库语义不变，只将 evaluation worker 的底层执行器替换为 Kubernetes Job，并采用 gVisor、Kata Containers 或一次性虚拟机提供更强隔离。

不采用 Web 服务器直接挂载 Docker socket 并启动用户镜像的方案。

## 4. 仓库与部署边界

### 4.1 `objview.github.io`

职责保持为静态展示：

- 论文介绍、图表、视频、BibTeX；
- Benchmark 和 Dataset 文档入口；
- 新增 `Submit Algorithm` 链接，跳转到独立域名；
- 不承载登录、提交、构建或评测逻辑。

提交网站域名作为部署配置，推荐使用 `submit.objview-bench.org`。

### 4.2 `objview_bench`

继续承担可信评测引擎职责：

- episode 状态机；
- observation 发布；
- feasibility 查询；
- render/eval cache；
- SC、NSC、Chamfer Distance、Path Cost、Views、Map Stabilization 等指标；
- 本地可信计时；
- episode summary 和 method aggregation。

需要新增或调整：

- HTTP interaction gateway；
- 对外协议 schema；
- 匿名 case/episode 标识；
- observation 匿名化发布；
- benchmark 本地耗时字段；
- 容器化 benchmark runner；
- 面向外部镜像的启动方式，不再要求写入静态 `METHODS` 注册表。

### 4.3 `objview_submit`

新建独立项目，负责：

- 用户、邀请和认证；
- 方法与 submission 管理；
- 构建、preflight、hidden evaluation 任务编排；
- 管理员审核；
- 结果入库和排行榜；
- worker agent；
- 对象存储和私有镜像仓库集成。

建议技术栈为 React/Next.js、FastAPI、PostgreSQL、Redis、Celery 或 RQ、MinIO/S3 和私有 OCI Registry。

## 5. 系统组件

### 5.1 Web 前端

用户页面：

- 登录与邀请状态；
- 方法列表；
- 新建 submission；
- 构建/preflight/正式评测状态；
- Public Dev 详细结果；
- Hidden Leaderboard 聚合结果；
- 协议和 Dockerfile 文档；
- 排行榜。

管理员页面：

- 邀请管理；
- submission 审核；
- 构建和评测任务控制；
- 完整日志查看；
- infrastructure failure 重试；
- Final Audit 批准；
- 排行榜发布和撤回；
- 审计日志。

### 5.2 API 服务

API 服务只操作数据库、对象存储、Registry 元数据和任务队列。它不持有隐藏数据、不挂载 Docker socket，也不执行用户命令。

### 5.3 Build Worker

Build Worker 主动领取构建任务，在隔离环境中执行 rootless BuildKit：

- 仅检出指定 Git commit 或解压受限大小的构建上下文；
- 不获得数据库、对象存储或隐藏数据凭据；
- 禁止 `security.insecure`、`network.host` 和任意 device entitlement；
- 通过受控代理或允许的 package registry 下载依赖；
- 施加构建时间、CPU、内存、磁盘和网络流量限制；
- 生成 image digest、构建日志、SBOM 和 provenance；
- 推送至私有 Registry；
- 构建完成后销毁工作目录和 builder 状态。

### 5.4 Evaluation Worker

Evaluation Worker 保存或只读挂载隐藏数据及缓存，运行可信 controller。每个正式 episode 创建：

- 一个可信 benchmark 容器；
- 一个不可信 algorithm 容器；
- 一个独占私有网络；
- 一个只包含已释放 observation 的临时 volume；
- 独立 CPU、内存、PID、磁盘和 GPU 配额。

正式 episode 结束后销毁容器、网络和临时 volume。镜像层可以缓存，但用户进程、可写层和 episode 状态不能复用。

## 6. 用户提交契约

用户提交内容：

1. Git 仓库 URL 和不可变 commit SHA，或受限大小的源码压缩包；
2. Dockerfile 路径；
3. 容器内启动脚本 `/opt/objview/run.sh`；
4. `submission.yaml`；
5. 方法名称、描述、作者和相关论文信息。

`submission.yaml` 基线结构：

```yaml
schema_version: objview-submission/v1
method:
  name: MyPlanner
  execution_mode: iterative
source:
  repository: https://github.com/example/my-planner
  commit: 0123456789abcdef
build:
  dockerfile: Dockerfile
runtime:
  entrypoint: /opt/objview/run.sh
  accelerator: gpu
  max_memory_gb: 16
protocol:
  version: objview-interaction/v1
```

平台决定实际运行参数，忽略镜像声明中可能扩大权限的设置。用户不得指定 host mount、privileged、host network、额外 capabilities 或 Docker socket。

启动时平台注入：

- `OBJVIEW_BENCHMARK_URL`；
- `OBJVIEW_RUN_TOKEN`；
- `OBJVIEW_PROTOCOL_VERSION`；
- `OBJVIEW_OBSERVATION_ROOT=/objview/observations`；
- `OBJVIEW_RUN_ID`，其值是不可反查数据对象的一次性随机标识。

启动脚本必须启动一个前台算法进程；进程退出即认为算法终止。模型加载完成后，算法调用 gateway 的 ready 接口。启动时间单独测量，不计入 planning runtime。

## 7. `objview-interaction/v1` 协议

### 7.1 传输方式

- 控制消息：私有网络上的 HTTP/JSON；
- RGB、depth、mask、frame metadata 和 point cloud：每个 episode 独占的临时 volume；
- benchmark 对 observation volume 可写；
- algorithm 对 observation volume 只读；
- algorithm 不拥有向共享 volume 写入动作的权限，动作只通过 HTTP 提交。

所有 HTTP 请求使用一次性 Bearer token，只在当前 evaluation unit 内有效。

### 7.2 接口

```text
GET  /v1/interface
POST /v1/ready
GET  /v1/episode
GET  /v1/steps/next?after=<step_index>
POST /v1/feasibility/check
POST /v1/actions
GET  /v1/status
```

`GET /v1/interface` 返回协议版本、pose 格式、支持的动作和资源限制。

`POST /v1/ready` 表示模型已加载并准备接收 episode。

`GET /v1/episode` 只返回任务类型、相机模型、归一化语义、预算语义和 view-space constraint，不返回真实 UID、GT 信息或指标。

`GET /v1/steps/next` 使用限时 long polling，返回当前 step、pose 和匿名 observation manifest。manifest 中只能使用以下形式的相对路径：

```text
step_000/rgb.png
step_000/depth.npz
step_000/mask.png
step_000/frame_meta.json
step_000/points.pcd
```

manifest 不得引用全局 render cache、对象 UID 或宿主机绝对路径。

`POST /v1/feasibility/check` 支持批量 pose，单请求最多 512 个 pose，单 step 累计最多检查 4096 个 pose。该限制作为 track protocol 的一部分固定并公开。

`POST /v1/actions` 每个 step 只接受一个最终动作：

```json
{
  "episode_id": "opaque-episode-id",
  "step_index": 3,
  "request_id": "client-generated-id",
  "action": "move",
  "pose": [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
}
```

或：

```json
{
  "episode_id": "opaque-episode-id",
  "step_index": 3,
  "request_id": "client-generated-id",
  "action": "stop",
  "stop_reason": "plan_end"
}
```

相同 `request_id` 和相同 payload 重试时返回相同结果；同一 `request_id` 携带不同 payload 返回冲突。旧 step、未来 step、第二个动作、非法 pose 和不可达 pose 均返回稳定的错误码。

普通 submission 不暴露 oracle 方法。用户提供的 `algorithm_runtime_sec` 字段被拒绝或忽略，不进入官方指标。

### 7.3 时间限制

基线限制：

- image build：45 分钟；
- algorithm startup/ready：10 分钟；
- 单 step action：120 秒；
- graceful shutdown：10 秒；
- stdout/stderr：每个 episode 最多保留 10 MiB，超出后截断但不中止算法；
- output 文件：hidden evaluation 不接受用户自定义输出文件。

正式 suite 按 episode 分片调度，不依赖一个超长、不可恢复的单体进程。

## 8. 可信计时

官方 planning time 只由 benchmark/controller 使用单调时钟测量。

每一步：

1. benchmark 完成 observation 文件写入和同步；
2. benchmark 发布 step，并记录 `t0 = time.perf_counter_ns()`；
3. algorithm 读取 observation、完成 inference 并提交 action；
4. gateway 完整接收请求并完成 schema、episode 和 step 校验；
5. 记录 `t1 = time.perf_counter_ns()`；
6. `decision_wall_time_ns = t1 - t0`。

官方指标包含：

- `container_startup_time_ns`；
- `model_ready_time_ns`；
- `decision_wall_time_ns` 每步值；
- `total_decision_wall_time_ns`；
- `benchmark_render_time_ns`；
- `benchmark_evaluation_time_ns`；
- `total_job_wall_time_ns`。

排行榜 Planning Runtime 使用 `total_decision_wall_time_ns`。渲染和 evaluator 时间不计入算法 planning time。用户上报耗时不显示为官方结果。

正式结果必须绑定固定 hardware profile。正式任务不在同一张算法 GPU 上并发运行多个 submission。记录 GPU 型号、驱动、CUDA、CPU、内存、benchmark image digest 和节点标识。若 online render 也使用 GPU，应为 benchmark 分配独立 GPU，或保证 render 与算法 inference 串行且不造成算法显存回收；具体硬件 profile 在部署时固定并公开。

## 9. 隐藏数据保护与反刷分

### 9.1 数据层级

采用三层数据：

1. Public Dev：公开数据，允许详细逐对象结果、完整用户日志和调试；
2. Hidden Leaderboard：日常正式排行榜，只返回有限精度的聚合指标；
3. Final Audit：不参与日常反馈，只在版本冻结或管理员最终审核时运行。

现有 528 个 Main-Hidden-Test 对象在首次平台发布前按 planning difficulty 等字段进行确定性分层划分：400 个用于 Hidden Leaderboard，128 个用于 Final Audit。划分 manifest 加密/受限保存并版本化，不进入用户可访问仓库。

### 9.2 匿名化

当前 `runner_episode.py` 会在 episode config 中发布真实 `uid`，必须移除。所有对外标识改为每次正式 submission 都重新生成的一次性随机 ID。不同 submission 不能通过 ID 对齐同一个隐藏对象。

cache hit 时也必须把已释放 observation 物化到匿名临时目录，不能在 manifest 中返回包含 UID 的全局 cache 路径。

### 9.3 结果最小化

Hidden Leaderboard 向用户返回：

- 总体成功/失败；
- 聚合后的 NSC、SC、Path Cost、Views 和官方 Planning Runtime；
- 精度固定为三位小数；
- 有限的终止原因分布；
- submission ID、image digest、benchmark/data/hardware profile 版本。

不返回：

- 单对象、单 start view、单 constraint 或单 step 指标；
- 隐藏 observation；
- 真实 UID；
- 原始 stdout/stderr；
- 用户生成的文件；
- evaluator 异常栈和绝对路径。

隐藏运行的原始日志只对管理员开放。用户仅获得枚举错误码，例如 `ALGORITHM_TIMEOUT_AT_STEP`、`INVALID_ACTION`、`OUT_OF_MEMORY` 和 `PROCESS_EXITED`。这是为了防止恶意算法把 observation 编码到日志或产物中带出。

### 9.4 提交限额

受邀阶段：

- 账户必须由管理员邀请并绑定 GitHub 身份；
- 每个 image digest 只能完成一次 Hidden Leaderboard 正式评测；
- 每个 method lineage 每 7 天最多一次 Hidden Leaderboard 评测；
- Final Audit 仅由管理员批准，通常在方法版本冻结后执行一次；
- infrastructure failure 不消耗限额，算法失败消耗限额；
- 管理员覆盖限额必须写入审计日志。

结果可以定时批量发布，避免提交后立即获得高频反馈。公网阶段在账户之外增加组织、仓库和 method lineage 联合限流，并保留一批从未用于公开排行榜的审计对象。

### 9.5 运行期网络

Algorithm Container 默认无互联网、无 DNS，只能通过私有网络访问当前 gateway。用户不能向外部服务器发送 observation，也不能访问其他 evaluation unit。

## 10. 容器和主机安全基线

Algorithm Container 必须：

- 以非 root UID/GID 运行；
- root filesystem 只读；
- `no-new-privileges`；
- drop 全部 Linux capabilities；
- 禁止 privileged、host network、host PID、host IPC；
- 不挂载 Docker/containerd socket；
- 不挂载用户 home、`/mnt`、`/dev`、X11 或宿主机源码目录；
- `/tmp` 和必要缓存使用有限大小 tmpfs；
- observation volume 只读；
- 设置 CPU、内存、PID、打开文件数、进程数、磁盘、日志和 GPU 限额；
- 使用默认拒绝的 egress policy；
- 任务结束后销毁可写层、volume、token 和网络。

现有 `algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh` 的 home、`/mnt`、`/dev`、X11 和全部 GPU 挂载不能用于外部 submission。

开发期即使用户受邀，也要求 Build/Evaluation Worker 与 Web/API/DB 分离。公网前增加强化 runtime 或一次性 VM，并完成第三方依赖和宿主机补丁策略。

## 11. 评测生命周期

### 11.1 状态机

```text
DRAFT
→ SUBMITTED
→ ADMIN_REVIEW
→ BUILDING
→ PREFLIGHT
→ READY_FOR_EVALUATION
→ RUNNING_HIDDEN
→ RESULT_REVIEW
→ PUBLISHED
```

失败状态：

- `BUILD_FAILED`；
- `PROTOCOL_FAILED`；
- `ALGORITHM_TIMEOUT`；
- `ALGORITHM_CRASHED`；
- `INVALID_ACTION`；
- `INFRASTRUCTURE_FAILED`；
- `REJECTED`。

只有确认属于平台的 `INFRASTRUCTURE_FAILED` 自动重试。算法崩溃、超时、非法动作和 OOM 不自动重试，防止利用重试获得额外隐藏反馈。

### 11.2 Preflight

正式隐藏评测前依次执行：

1. manifest 和 Dockerfile 基础检查；
2. 隔离构建；
3. 镜像大小、用户、entrypoint 和依赖检查；
4. 启动与 ready 检查；
5. dummy 协议测试；
6. 非法 step、重复动作、超时和 feasibility 批量查询测试；
7. Public Dev 小规模端到端评测；
8. 资源上限和网络隔离测试；
9. 管理员批准。

Preflight 允许向用户返回详细日志，因为使用的是公开数据。

### 11.3 正式 episode

正式 hidden episode 使用全新的 algorithm 容器和可写层。不能在多个 hidden episode 之间复用用户进程，从机制上阻止跨 episode 保留状态、在线更新模型或累计隐藏对象信息。镜像层和只读模型权重可以由 worker 缓存。

该决定增加模型重复加载成本，但优先保证公平性、可复现性和反提取能力。启动耗时独立统计，不进入 Planning Runtime。

## 12. 数据模型

核心表：

- `users`：身份、GitHub subject、状态；
- `invitations`：邀请、过期时间、使用记录；
- `methods`：方法名称、owner、lineage、描述；
- `submissions`：源码 commit、manifest、状态；
- `builds`：image digest、日志、SBOM、builder 版本；
- `evaluations`：数据版本、protocol、hardware profile、聚合状态；
- `episodes`：内部 episode 状态和可信结果，仅管理员可读；
- `leaderboard_entries`：可公开聚合结果；
- `artifacts`：对象存储 key、哈希、保留策略和可见级别；
- `audit_events`：管理员操作、限额覆盖、重试和发布记录。

数据库中区分 `public`、`submitter`、`admin` 三种结果可见级别。API 不直接序列化内部 episode 模型，避免误返回隐藏字段。

## 13. 结果与可复现性

每个正式结果保存：

- source commit；
- Dockerfile 和 manifest 哈希；
- image digest；
- benchmark Git commit 和 image digest；
- protocol version；
- hidden split version；
- render/eval cache version；
- feasibility config version；
- hardware profile；
- driver/CUDA/runtime 版本；
- controller 版本；
- 聚合脚本版本；
- 可信 summary 哈希。

排行榜条目从不可变 summary 生成。重新聚合产生新版本，不覆盖原始结果。

## 14. 错误处理

错误分为三类：

1. 用户错误：构建失败、协议错误、非法动作、超时、OOM、进程退出；
2. 平台错误：worker 丢失、Registry/Object Storage 不可用、benchmark 崩溃、节点故障；
3. 审核错误：安全策略拒绝、manifest 不一致、方法身份冲突。

平台错误允许在新的临时环境中重试，并保留前一次内部记录。用户错误不自动重试。对用户返回稳定错误码，不返回隐藏路径、对象 ID、内部堆栈或逐 episode 定位信息。

## 15. 测试策略

### 15.0 本机开发验收边界（2026-10-22 确认）

本机没有运行多数 ObjView-Bench 算法的完整环境。项目后期的开发验收以**整个业务链路跑通**为目标，不以真实算法、重建质量、指标数值正确性、论文成绩复现或性能排名为通过条件；这些不再作为本机开发的阻塞项。

- 使用可信 CPU dummy 算法、合成 RGBD 观察，以及必要的测试专用指标计算替身；不安装或执行研究者真实算法，不依赖真实测试集、权重或 GPU。
- 保留真实的 SDK、HTTP、协议状态机、runner、评分调用与汇总落盘；在后续阶段逐步覆盖网站提交、构建/调度、状态展示、审核/发布和清理。不能用模拟状态直接宣称尚未实现的环节已经通过。
- “不验证算法结果正确”不等于允许链路错误：接口格式、动作序列、鉴权、匿名信息边界、服务端耗时、幂等、错误分类、超时、持久化与清理仍需测试；评分调用必须产出可序列化结果，但不与科学指标真值比较。
- 开发报告固定标识 `mode=smoke_test`、`official_result=false`、`leaderboard_eligible=false`、`algorithm_correctness_checked=false`。所有合成结果必须保留此来源，存放在独立测试空间；未来发布端必须拒绝 smoke 或来源缺失的结果进入正式榜单。
- 无 Docker 时允许以本地可信进程验证业务通信；这种通过不代表已验证容器隔离、网络隔离或资源限制。缺少 Docker/GPU 不阻塞本机业务开发，但真实安全隔离仍是接收不可信代码/正式上线前的门槛。
- 一键入口输出本次实际覆盖项和未覆盖项；测试失败、未执行或跳过必需用例均不能标记为通过。不要把此前的成功报告复用于本次失败运行。

### 15.1 协议测试

复用并扩展现有 dummy algorithms，覆盖：

- 正常 move/stop；
- 重复 request；
- stale/future step；
- 非法 action 和 pose；
- 不可达 pose；
- action timeout；
- startup timeout；
- 进程提前退出和 stop 后不退出；
- feasibility 配额。

### 15.2 安全测试

建立恶意 canary 镜像，验证其不能：

- 访问互联网或 DNS；
- 访问 Docker socket；
- 读取宿主机、其他任务或隐藏数据目录；
- 提权或新增 capabilities；
- 通过日志和 artifact 向用户导出隐藏 observation；
- 超出 CPU、内存、PID、磁盘和时间限制；
- 访问非当前 gateway。

### 15.3 计时测试

使用可控 sleep 和固定推理 dummy，验证：

- 用户上报时间不影响官方指标；
- render/evaluator 时间不混入 decision time；
- timeout 边界稳定；
- 重试不会重复累计时间；
- 相同 hardware profile 的重复运行波动在预设容差内。

### 15.4 端到端测试

最终覆盖 GitHub 登录、邀请、提交、构建、preflight、管理员批准、hidden evaluation、结果审核、发布和静态主页跳转。遵循 15.0 的本机验收边界：链路中的算法可以是 dummy，评分可以使用测试替身；当前仅对已经实现的链路报告通过，外部 OAuth/镜像构建/容器等替身必须在报告中列出，不等同于真实部署验收。

## 16. 验收标准

验收分为两层：**本机开发交付**按 15.0 验证已实现的业务链路，不检查算法质量或科学指标数值；**接收不可信代码的受邀试运行上线**仍须满足以下安全与运行条件。本机 smoke 测试不能替代上线条件：

1. 静态主页能够跳转到独立提交网站；
2. 只有受邀 GitHub 用户可以提交；
3. Web/API 主机不执行 Docker build/run，也不挂载 Docker socket；
4. 用户提交能够生成不可变 image digest；
5. 官方 SDK 和至少一个 Python/C++ dummy submission 通过 preflight；
6. algorithm 看不到真实 UID、GT、全局 cache 路径或隐藏指标；
7. hidden 运行无公网连接，且资源上限有效；
8. 官方耗时由 benchmark 本地测量；
9. hidden 用户响应不包含逐对象数据、原始日志或用户 artifact；
10. infrastructure failure 与 algorithm failure 能正确区分；
11. 每个 leaderboard 结果具备完整的版本和硬件证据；
12. 管理员能够审核、重试平台故障、发布和撤回成绩；
13. 任务结束后容器、网络、token 和临时 observation 被清理；
14. 全部协议、安全、计时和端到端测试通过。

## 17. 分阶段实施顺序

### 阶段 A：Benchmark 协议与可信计时

- 固化 `objview-interaction/v1` schema；
- 移除 UID 和 cache path 泄漏；
- 增加 HTTP gateway；
- 增加本地可信计时；
- 完成 SDK、dummy submission 和 contract tests。
- 增加一键 CPU/dummy 链路验收及明确不可上榜的 smoke 报告。

### 阶段 B：本地双容器 Evaluation Unit

- benchmark image；
- algorithm image contract；
- 私有网络和匿名 observation volume；
- controller；
- 资源、超时、清理和安全 canary 测试。
- 无 Docker 的开发机可先验证可信进程和调度契约；双容器及 canary 的真实验证单独标记，不能由 mock 推断安全隔离通过。

2026-10-22 开发落实范围：先实现仅接受可信仓库 smoke 镜像的本地 worker，固定资源/挂载参数、镜像 ID、任务期限、状态和定向清理；不开放上传镜像执行。用户已选择暂缓 Docker 安装，先以实际入口的双进程测试及 Docker 命令契约测试继续开发。真实双容器验收、独立 model-ready 计时、磁盘配额、网络 canary 和恢复清理仍为后续任务。具体运行与未验收项见 `submission_worker/README.md`。

### 阶段 C：受邀提交控制面

- `objview_submit` Web/API；
- GitHub OAuth 和邀请；
- PostgreSQL、Redis、MinIO/S3、Registry；
- submission/build/preflight/admin 工作流。

2026-10-22 C1 开发增量：先落地独立 `objview_submit` API 包，使用本机 SQLite 完成邀请身份、会话撤销、源码元数据登记、幂等、持久化入口额度和管理员审核。用户已要求暂缓 Docker 安装；这一增量不依赖 Docker，也不运行用户代码。开发凭证由管理员私下交付，**不冒充 GitHub OAuth**；只有显式 `--development-only` 才可启动 CLI，固定绑定 localhost。它是开发适配器，不替代生产 PostgreSQL/OAuth/对象存储/隔离构建器。

C1 状态为 `SUBMITTED -> ADMIN_REVIEW -> APPROVED_FOR_PREFLIGHT | REJECTED`；`APPROVED_FOR_PREFLIGHT` 是尚无 worker 消费的等待状态，不能报告 BUILDING/RUNNING 或生成成绩。仅接收 `objview-interaction/v1`、不可变 SHA、GitHub HTTPS 仓库和相对 Dockerfile 路径等元数据；无下载、上传、构建、hidden、分数或用户时间接口。额度为每账户滚动 24 小时 5 次新提交、3 个未拒绝提交、10 个方法；它们只限制开发入口，不替代 §9 的团队/image/method-lineage 正式配额。具体契约和未实现能力见 `objview_submit/README.md`，实施计划见 `docs/superpowers/plans/2026-10-22-phase-c-intake.md`。

### 阶段 D：Hidden Leaderboard

- 数据分层与版本化；
- 限额和 method lineage；
- 聚合结果、审核和发布；
- GitHub Pages 导航链接。

### 阶段 E：公网强化

- Kubernetes Job；
- gVisor/Kata/一次性 VM 评估和落地；
- 多 worker 调度；
- 组织级限流和滥用检测；
- 安全审计和应急响应。

## 18. 关键取舍

- 选择每个 hidden episode 重启用户容器，以计算成本换取可验证的状态隔离。
- 选择控制消息走 HTTP、观察数据走只读临时 volume，避免大规模 RGB-D 经全局消息队列传输。
- 选择邀请制和管理员审核作为首版边界，但不把“受邀”当成唯一安全措施。
- 选择聚合结果和限额降低 hidden test 被反复探测的风险，同时保留 Final Audit 作为独立验证集。
- 选择 benchmark 本地 wall-clock 作为官方耗时，以避免用户伪造运行时间。
- 选择独立 `objview_submit` 项目，避免把动态平台逻辑混入 GitHub Pages 或评测引擎。
