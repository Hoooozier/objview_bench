# 提交评测平台开发进度

更新：2026-10-22。设计基线：[设计方案](specs/2026-10-21-objview-submission-platform-design.md)。

## 已实现：Phase A，单 episode 的可信协议边界

工作树 `/home/huhao/Documents/codes/objview_bench-platform`，分支 `feat/submission-protocol`。主仓库与静态站未被合并/发布，测试集划分未改动。

- 匿名 episode ID、只允许公开相机信息的配置、Bearer 鉴权 HTTP 与 Python SDK。
- ready → 当前观察 → move/stop → runner → 私有 evaluator summary；不开放 oracle、GT、缓存索引、原始分数与文件下载路由。
- 服务端单调时钟计时、动作幂等、视图预算、120 秒决策期限、可行性查询额度、有限 JSON 体积/读取时限。
- 缓存命中也生成独立匿名 RGBD 文件；文件在私有目录暂存，计时开始后原子释放。私有记录不再与算法共享；文件式动作/RPC 在提交模式中被忽略。
- 服务端 CLI、启动等待/失败状态、私有 summary、示例 Dockerfile/启动脚本/来源清单模板。
- 详细运行与边界说明：[SUBMISSION_PROTOCOL.md](../../benchmark/SUBMISSION_PROTOCOL.md)。

## 验证证据

命令（仓库根目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 -m unittest discover -s benchmark/tests -v
python3 benchmark/serve_submission_episode.py --help
git diff --check
```

35 项测试通过，覆盖纯协议、真实 FastAPI ASGI 请求、真实本地 TCP HTTP、SDK、现有 EpisodeRunner 与 EvaluateAPI、生成的 PNG/NPZ 文件、计时汇总、超时与非法动作。CLI help、11 个 Python 文件编译、启动脚本 `sh -n` 和 diff 空白检查通过；编译缓存输出到临时目录，没有污染仓库。集成测试使用微型合成渲染数据；仅将 evaluator 的 PyTorch3D 距离核替换为 NumPy 参考计算，并统一无渲染依赖时的 fallback CameraPose 类型。不能由此推断真实渲染、GPU 数值或完整数据集评测已通过。

本机沙箱中 TestClient 曾卡在异步线程唤醒；已用最小 FastAPI 程序确认并获准在沙箱外重跑。本次完整测试只访问本地临时文件和 127.0.0.1 临时端口，不访问外部服务。

独立只读审查发现并用先失败后通过的回归修复：

1. 巨大整数姿态先进入 `math.isfinite` 导致 OverflowError：先进行数值上界检查，action/feasibility HTTP 返回 422，session 仍可继续。
2. 视图 cap=1 时旧 runner 仍会接受一个 move：初始观察已耗尽预算时直接结束，无额外渲染与决策计时。

另增加了文件监视提前推理的回归：计时必须早于观察目录的原子发布。

审查者复查修复后，独立运行上述两个直接回归通过；在 Phase A 已声明范围内未发现剩余阻断问题。审查不是生产安全认证，也没有独立复跑完整 HTTP 测试。

## 当前没有完成/不能声称的能力

- 本机没有 Docker 命令；2026-10-22 用户明确选择暂缓安装。示例镜像未构建，双容器网络、只读挂载、GPU 隔离、资源限制和逃逸防护未实测；这不阻止本地控制器契约与双进程验收。
- 未启动真实数据集/GPU 渲染评测；根据新的本机验收要求，这不再阻塞开发交付，不要求在本机验证算法效果或科学指标数值。实际部署安全仍需专用 worker 验收。
- 尚没有源码上传网站、GitHub OAuth、构建器、调度队列、全局反复提取防护或榜单发布。C1 已增加本地邀请凭证、SQLite 持久化提交/入口额度和审核 API（见文末）；不是正式隐藏评测配额。
- 单 episode 匿名与查询限额只减少直接泄漏，不保证阻止反复提取测试数据或基于公开对象识别刷分。
- 当前 ready 计时包含部分 benchmark 准备时间；容器启动/模型就绪耗时应由下一阶段控制器独立测量。
- HTTP server 的 watchdog 是协作式退出，不是 native/GPU 调用的硬资源边界。

## 下一阶段实施顺序

1. **Phase B：可信执行控制器**。本地 trusted-smoke 控制器已实现（见下）；待 Docker 可用后验收镜像构建与双容器真实链路，补全隔离 canary、磁盘限额、独立 readiness 计时和崩溃恢复清理。当前不运行研究者代码。
2. **Phase C：受邀用户提交站后端**。本地 C1 邀请/元数据/审核 API 已实现（见文末）；接下来做页面和持久化队列，再接 GitHub 登录、团队级额度、上传对象存储、源码/权重哈希。Web 不挂载 Docker socket。正式配额必须持久化并覆盖失败尝试。
3. **Phase D：前端与发布**。提交/状态页面、只显示安全错误与聚合结果、管理员审核；确定真实域名后在 github.io 增加链接，不迁移静态站。
4. **Phase E：公开注册前安全验收**。强化 VM/microVM 边界、网络白名单/旁路测试、跨作业访问与数据导出测试、审计、依赖和镜像扫描、限流与恢复演练。

继续开发不需要重新讨论上述已确认架构。真正部署时才需要维护者提供专用 worker 环境、GPU 配置、域名和 GitHub OAuth 凭据；这些不能写入仓库。

## 2026-10-22 验收标准补充：只验链路，不验算法质量

已修改设计 §15.0、§15.4、§16–17，将本机开发验收与接收不可信代码的上线验收分开。新增 `benchmark/smoke_submission.py`，只需 HTTP 层、NumPy、Pillow，运行固定的协议/runner/service 用例；其中新增真实 dummy 子进程 move/stop 链路与私有汇总落盘、清理测试。

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 benchmark/smoke_submission.py
```

首次实跑：36 项链路/协议测试通过，报告 `mode=smoke_test`，明确不可上榜；没有测试算法结果是否正确。另有报告器回归检查失败、跳过、空套件、中断和旧报告覆盖保护。新的验收入口不导出合成分数，不启动真实算法，不改变正式 benchmark 的评分实现。网站/构建/控制器链路将在相应阶段实现后加入验收，当前报告明确列为未覆盖。

本次最终验证：全套 **42 项通过**（36 项协议/链路 + 6 项报告器回归）；一键入口单独复跑 **36 项通过**，并检查 `sys.modules` 确认未加载 Torch、PyTorch3D、Open3D。CLI help、修改的 Python 文件编译及 `git diff --check` 通过。

审查后将汇总保存提取为服务端与链路用例共用的 `persist_episode_result`，避免仅测试测试代码自己的写文件逻辑；报告加入实际执行的测试 ID。最新本机报告位于 `/tmp/objview-smoke-report-qacckto5/report.json`（临时文件，系统清理后可用一键命令重建）。

## 2026-10-22 Phase B：开发版 worker 与镜像入口

代码：[submission_worker](../../submission_worker/README.md)。计划：[phase-b](plans/2026-10-22-phase-b.md)。

已实现：

- 独立 CLI 控制器，明确要求 `--trusted-smoke`；已有本地镜像解析为 SHA256 ID，不自动拉镜像，不构建或运行上传的源码。
- 创建任务专属内部 network / volume、固定可信 init 容器，以及独立非 root benchmark/algorithm 容器；算法只读挂载 released 子目录，private staging 保持私有。
- CPU/内存/PID/tmpfs/日志等固定 Docker 参数、只读根目录、cap-drop、no-new-privileges、命令与总时限、任务标签核对及定向清理。不允许任意 Docker flags 或远程 daemon。
- PREPARING/STARTING/RUNNING/成功失败状态、算法崩溃/OOM/benchmark 失败/总超时分类、私有报告、清理失败不报成功。没有自动重试。
- 合成 benchmark 容器入口、两份可信 Dockerfile 和源码白名单 `.dockerignore`。使用真实进程验证与镜像相同的 benchmark 入口 + dummy 算法完成完整 episode；仍不验证算法效果。

验证：

- 全套 **60 项测试通过**：原有用例、容器入口双进程链路、17 项 worker 命令/状态测试。
- 一键 smoke 验收扩展为 **54 项通过**，报告包含 `docker_command_boundary` 替身标记；报告 `/tmp/objview-smoke-report-8518ac5i/report.json`。
- 实际调用 worker CLI 验证缺少 Docker 时退出 1，状态 `FAILED/INFRASTRUCTURE_FAILED`、`runtime_code=DOCKER_UNAVAILABLE`、未创建容器资源；失败报告 `/tmp/objview-worker-check.I68ujh/job/report.json`。
- 独立只读审查发现准备阶段 Docker 调用未遵守总期限，已用失败回归修复并复核；同时补齐 benchmark 提前崩溃检测和报告嵌套计时白名单。修改的 Python 文件编译、CLI help 和 diff 检查通过。

安装情况：宿主 Ubuntu 22.04、无 Docker，sudo 需要交互密码；告知后用户选择“先继续开发，稍后再安装 Docker”。未修改系统包、sudo 配置、用户组或服务。实际 Docker 构建/执行保持**未验收**；不是整个 Phase B 或生产安全已完成。rootless 前置包与后续自动安装步骤已写入 worker README。

后续可以先做 Phase C 的邀请用户、提交元数据与持久化任务状态；Web/API 不挂 Docker socket。真实 Docker 验收与更强隔离措施在安装/专用 worker 就绪后补做，正式上传执行仍不开启。

## 2026-10-22 Phase C1：本地受邀提交与人工审核 API

代码与启动说明：[objview_submit/README.md](../../objview_submit/README.md)。实施计划：[phase-c-intake](plans/2026-10-22-phase-c-intake.md)。

已实现：

- 独立 `objview_submit` Python 包，不导入 benchmark/worker；CLI 要求 `--development-only`、固定 localhost、管理员凭证从环境读取，不挂 Docker socket，不拉仓库/构建/执行源码。
- 一次性、限时邀请，管理员预先指定身份标签；兑换 24 小时会话。SQLite 仅存 token 摘要，支持邀请撤销、退出和禁用账户。相同标签续邀保留账户与配额。这是开发凭证流程，不是 GitHub OAuth。
- 用户方法与源码元数据、完整 SHA/协议版本/路径白名单、用户所有权校验；拒绝用户指定 owner、状态、指标和耗时。
- 每账户持久化入口额度、原子并发检查、请求幂等；管理员基于 version 的状态转换和同事务审计。`SUBMITTED -> ADMIN_REVIEW -> APPROVED_FOR_PREFLIGHT | REJECTED`，批准仅是等待状态，无实际执行和榜单资格。
- JSON 大小/读取时限、兑换临时限流、禁用文档/访问日志、无 CORS、no-store 与错误脱敏。私有状态目录 0700、DB 0600、未知 schema 拒绝启动。

本次最终验收：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 -m unittest discover -s benchmark/tests -v
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 benchmark/smoke_submission.py
PYTHONDONTWRITEBYTECODE=1 python3 -m objview_submit --help
git diff --check
```

- 全套 **84 项测试通过**（7.495 秒）：原有 60 项 + 24 项 intake 测试。
- intake 含实际 TCP HTTP 服务进程、真实 SQLite、停止并重启服务后凭证与提交继续有效、管理员继续审核；不是只检查 health 200。覆盖并发邀请兑换、提交额度与审核竞争、跨用户/跨角色拒绝、重复 key、凭证过期/撤销、数据库错误脱敏等。
- 原有 benchmark/worker smoke 独立复跑 **54 项通过**（3.578 秒），报告 `/tmp/objview-smoke-report-2n5sfcui/report.json`；报告仍不包含 intake，不声称网站→构建→评测已全线贯通。
- 独立只读审查未发现当前本地开发范围内的 Critical/Important 问题；审查者独立复跑 14 项真实 SQLite 测试通过，HTTP/进程测试由主执行者实跑。非阻断项：提交/审计列表只有最近 100 条，无分页，旧待审核项可能被新历史挤出；正式试用前补齐分页。
- 复核协议字段时修正为设计与 gateway 一致的 `objview-interaction/v1`，先复现旧短名称校验不兼容，再通过回归。CLI help、语法和 diff 检查通过；`.state` 已在包级 `.gitignore` 排除。

未做/下一步：前端提交与审核页面、分页、持久化工作队列与可信 smoke 对接；真实 OAuth、源码文件接收/对象存储、独立构建器、团队与 image/method-lineage 配额仍未实现。没有安装 Docker，没有启动长期服务，没有改 github.io 或数据集，不验证科学指标正确性。不允许把本地模式暴露公网或接收不可信算法执行。
