# ObjView 提交评测平台：跨机器 / Codex 接续文档

更新：2026-10-22。本文件是**接续入口**，与代码一起提交；无需原机器、旧对话、`/tmp` 报告或 Codex 私有记忆。完整设计与历史计划均保存在本仓库，见第 3 节。

## 1. 给下一台机器上的 Codex

可直接把下面这段发给新会话：

> 请先完整阅读仓库根目录 `AGENTS.md` 和 `HANDOFF.md`，再阅读后者链接的设计基线、开发进度及当前任务相关文档。检查当前分支与工作区，安装/核对最小开发依赖并复跑基线测试。继续已确认的 ObjView 算法提交评测平台，不要重新从零设计。优先按第 7 节推进列表分页、提交/审核页面，再接持久化可信 smoke 队列。开发机只验链路，不要求真实算法或科学指标正确；禁止把 smoke 当官方成绩、让 Web 执行上传代码或直接访问 Docker。报告每阶段实际验证结果并更新进度。不要合并 main 或公开部署；新环境的凭证/域名/worker 资源缺失时先完成不依赖它们的工作。

`AGENTS.md` 提供仓库级自动发现提示；本文件提供完整背景。机器上的技能插件、代理 ID、聊天会话 ID 和绝对路径不是继续开发的前提。

## 2. 代码位置、版本与迁移方法

### 2.1 仓库与分支

- 算法/平台仓库：`https://github.com/Hoooozier/objview_bench.git`
- **开发分支：`feat/submission-protocol`**，不要只克隆默认 `main` 后误以为功能缺失。
- 本轮用户授权：提交交接文档、推送该开发分支，**不合并 main**。
- 业务代码检查点：`7080a94`。交接文档会有后续提交；请以 `git log -1` 和远端分支实际 HEAD 为准，不硬编码文档自身的 commit SHA。
- 原机器的开发目录为 `/home/huhao/Documents/codes/objview_bench-platform`，它是 linked worktree；主仓库目录 `/home/huhao/Documents/codes/objview_bench`。**不要只复制开发目录当作完整 Git 仓库**，其 `.git` 指向另一目录。新机器使用 clone 或下面的 bundle。
- 静态站另在 `https://github.com/Hoooozier/objview.github.io.git`，此次未修改；后续确定真实提交站域名后只加导航链接。

关键提交，按时间从早到晚：

| 提交 | 内容 |
| --- | --- |
| `f29fcef` | 已确认平台设计基线 |
| `587d3c7` | 单 episode HTTP 协议、隔离数据发布、本地可信计时 |
| `a2acfd9` | CPU-only dummy 链路验收与不可上榜 smoke 报告 |
| `3e214d7` | 可信 Docker smoke 生命周期控制器与容器入口 |
| `7080a94` | 本地邀请、提交元数据、持久化配额、人工审核 API |

### 2.2 推荐：从已推送的分支克隆

在一个不含同名目录的位置执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch feat/submission-protocol --single-branch \
  https://github.com/Hoooozier/objview_bench.git objview_bench-platform
cd objview_bench-platform
git status --short --branch
git log -6 --oneline
```

跳过 LFS 下载是有意的：开发链路不用原算法权重。仓库历史含两个 LFS 指针（PoinTr `ckpt-best.pth`、MASCVP `best_f1.pt`）；正式算法运行时另行验证 Git LFS 对象、权重与数据授权/可用性，不要把指针文件当实际权重。Git LFS 不在最小 CPU 验收依赖中。

### 2.3 离线备选：Git bundle

本轮交付同时生成 `objview-platform.bundle`，放在仓库外的唯一交付目录，确切路径在本次对话最终答复中。它携带开发分支可达的完整 Git 历史，无先决提交；可复制这一个文件到另一台机器。交付目录的 `SHA256SUMS` 用于传输校验。

```bash
# 在复制来的交付目录内
sha256sum -c SHA256SUMS
git bundle list-heads objview-platform.bundle
GIT_LFS_SKIP_SMUDGE=1 git clone --branch feat/submission-protocol \
  objview-platform.bundle objview_bench-platform
cd objview_bench-platform
# clone bundle 后 origin 指向 bundle 文件，按需改成正常远端
git remote set-url origin https://github.com/Hoooozier/objview_bench.git
git status --short --branch
```

bundle 包括全部已跟踪源码、设计与计划；**不包括** LFS 对象实体、工作区未跟踪数据、模型缓存、原论文目录、独立静态站仓库、虚拟环境、Docker 镜像、运行时数据库、秘密或 `/tmp` 报告。本地开发不需要这些。正式评测的数据/权重迁移另行安排，不把它们装入公开代码交付包。

## 3. 设计方案与已有文档（均随 Git 保存）

| 文档 | 作用 / 阅读顺序 |
| --- | --- |
| [设计基线](docs/superpowers/specs/2026-10-21-objview-submission-platform-design.md) | 完整架构、安全、隐藏集、计时、状态、数据模型、阶段与验收标准；主要依据 |
| [开发进度](docs/superpowers/progress.md) | 逐阶段实际完成项、测试证据、修复与明确未完成项 |
| [Phase A 计划](docs/superpowers/plans/2026-10-22-phase-a.md) | HTTP 协议与 benchmark 接入的历史实现计划 |
| [Phase B 计划](docs/superpowers/plans/2026-10-22-phase-b.md) | 可信 smoke worker；真实 Docker 验收仍未勾选 |
| [Phase C1 计划](docs/superpowers/plans/2026-10-22-phase-c-intake.md) | 本地 intake 身份、元数据、审核增量，已完成 |
| [提交协议](benchmark/SUBMISSION_PROTOCOL.md) | SDK、gateway、trusted runner、计时与 CPU 验收命令 |
| [Worker 说明](submission_worker/README.md) | Docker 命令契约、报告、资源清理、安装与真实容器验收缺口 |
| [提交 API 说明](objview_submit/README.md) | 本地启动、完整 API 示例、身份/配额/状态与安全限制 |

设计文档中的生产目标不是完成声明：例如 PostgreSQL、GitHub OAuth、正式 hidden 队列尚不存在。判断“目前实现了什么”应结合进度、代码与测试；历史测试数量和 `/tmp` 路径只是当时证据，不是新环境保证。

## 4. 已确认需求，不要在迁移后丢失

1. `github.io` 继续静态展示，仅导航至独立提交站；不把动态后端塞入 GitHub Pages。
2. 最终用户提交 Dockerfile、启动脚本、代码/权重来源。算法容器向独立可信 benchmark 请求初始/后续观察，输出视点动作，benchmark 本地计算指标。
3. 尽量抑制反复提取隐藏数据和刷分：限制暴露、无原始 hidden 日志/逐对象分数、团队/方法谱系/镜像配额、独立 Final Audit、人工审核。不存在“仅靠 Docker 或邀请制就绝对防泄漏”的承诺。
4. 不信任用户返回的耗时/成绩。可信 benchmark 本地单调时钟计时，平台决定实际运行参数。
5. 初期邀请制，后期公开注册。用户不熟悉安全细节，需要系统默认拒绝不安全操作；Web 不持有 Docker socket、不执行用户代码、不访问隐藏数据。
6. **本机/开发阶段只要求端到端业务链路通，不要求真实算法效果或科学指标正确。** 可以用 CPU、合成 RGBD、可信 dummy；报告必须 smoke-only、不可上榜。
7. 不要求因缺少 Torch/PyTorch3D/Open3D/GPU 而停止开发。真实隔离、安全与正式数据评分属于后续部署验收，不能把 mock 当验证通过。
8. 用户先允许按需安装 Docker，随后明确选择“先继续开发，稍后再安装 Docker”。旧机器 sudo 需要交互密码；未安装任何系统包、未改用户组/服务。新机器重新检查环境，不继承旧机器 sudo/socket/UID 假设，不向用户索取密码。
9. 不修改论文实验、现有算法语义或实际 528 对象数据集划分。设计中更细的 hidden/public 分层仍需真实发布前确认，不按文档示例擅自重新划数据。

## 5. 已完成源码地图与边界

### A：单 episode 协议与可信计时

- `benchmark/submission_protocol.py`：会话状态机、匿名 episode、字段白名单、有限7D pose、动作幂等、步数/可行性查询/决策时间限额、本地时间汇总。
- `benchmark/submission_gateway.py` / `submission_client.py`：私有 Bearer HTTP gateway、大小/读取时限、SDK；不提供 oracle、GT、分数或文件下载接口。
- `benchmark/submission_interaction.py`：与旧 runner 接口适配；观察先在私有目录暂存，**计时开始后**原子发布到只读共享目录；不接受旧文件式 action/RPC。
- `benchmark/runner_episode.py`、`serve_submission_episode.py`：真实 runner/evaluator 接入、启动等待、终止/超时状态、私有结果写入、可信计时；cap=1 不多走一步。
- `benchmark/examples/submission/`：仓库可信 dummy、Dockerfile、启动脚本。旧 `submission-manifest.example.json` 是构建说明草稿，不可直接 POST 到 C1 intake；标准 intake 协议名称为 `objview-interaction/v1`，标准启动路径 `/opt/objview/run.sh`。
- 当前 protocol ready 时间含部分 benchmark 准备，不能当独立模型加载耗时；这是后续须修复的测量口径，不应隐瞒。

### B：CPU 验收与可信 smoke worker

- `benchmark/smoke_fixtures.py`、`smoke_container_service.py`：2×2 合成 RGBD / 简单点云，保留真实 runner / EvaluateAPI 调用；测试专用 NumPy 距离核和 fallback camera type，不装重型算法依赖。
- `benchmark/smoke_submission.py`：固定验收套件、JSON 报告、失败/跳过/空套件不冒充成功；不可覆盖旧报告，记录执行的 test IDs。
- `benchmark/containers/Dockerfile.smoke` 和根 `.dockerignore`：可信 CPU smoke 镜像入口与最小源码构建上下文。
- `submission_worker/docker_runtime.py`：固定 argv 的 Docker 适配器、显式本地 Unix socket、无 shell、环境隔离、版本/资源前置检查、只读挂载参数、已有镜像解析为 digest、精确资源查找。
- `submission_worker/controller.py`：必须 `--trusted-smoke`；准备/启动/运行/失败生命周期、总 deadline、异常分类、可信汇总白名单、按任务标签清理并核验。
- 已用真实子进程跑通与镜像相同的 benchmark 入口 + dummy；Docker 调用边界由有状态替身测。**没有真实构建/运行容器；controller 也不支持研究者上传镜像。**
- 镜像 label/`--trusted-smoke` 仅防误用，不是恶意镜像认证；internal network 不是严格 egress 防火墙。SIGKILL/断电恢复清理、磁盘硬配额、DNS/宿主旁路 canary、GPU 隔离尚未实现或验收。

### C1：独立本地 intake / 审核 API

- `objview_submit/store.py`：真实 SQLite，私有目录/DB，事务合并身份/额度/插入/审核/审计；schema version=1，未知版本拒绝，无迁移系统。
- `validation.py`：严格六字段元数据、HTTPS GitHub 仓库、完整小写40位 SHA、相对 Dockerfile 路径、固定启动脚本和协议。
- `app.py` / `__main__.py`：一次性邀请、24小时会话、撤销/禁用、所有权隔离、方法/提交列表、幂等、管理员 expected-version 审核；只监听 localhost，显式开发模式。
- token 仅存摘要，管理员 token 从环境读；身份是管理员指定 label，**不是 GitHub OAuth**。SQLite 是本地开发适配器，不是生产 PostgreSQL 替代承诺。
- 每账户滚动24小时5次新提交、最多3个非 REJECTED 提交、最多10个方法；同一 label 续邀不重置身份/额度。它们不是正式的团队/镜像/方法谱系 hidden 配额。

```text
SUBMITTED -> ADMIN_REVIEW -> APPROVED_FOR_PREFLIGHT
                         -> REJECTED
```

`APPROVED_FOR_PREFLIGHT` 只是等待状态，**没有 worker/队列消费它**，不表示 BUILDING/RUNNING；没有正式结果或榜单资格。当前网站 intake 与 benchmark smoke 各自可测试，但尚未连成“网站提交→构建→评测”的一条链路。

## 6. 新机器最小环境与验证

基线环境为 Linux、Python 3.10。新环境建议先用 3.10；其他 Python 版本未经此检查点验收。本轮无需 Node、Docker、GPU、数据集、模型权重或重型渲染库。

```bash
# 仓库根目录；虚拟环境放到仓库外，父目录自行选择已存在的位置
python3 -m venv ../objview-platform-venv
. ../objview-platform-venv/bin/activate
python -m pip install -r benchmark/requirements-smoke.txt -r objview_submit/requirements.txt
PYTHONDONTWRITEBYTECODE=1 timeout 90s python -m unittest discover -s benchmark/tests -v
PYTHONDONTWRITEBYTECODE=1 timeout 90s python benchmark/smoke_submission.py
PYTHONDONTWRITEBYTECODE=1 python -m objview_submit --help
PYTHONDONTWRITEBYTECODE=1 python -m submission_worker.controller --help
git diff --check
```

依赖是本机兼容快照，并非经过安全审计的生产 lockfile。正式部署前需重新扫描/锁定。不要顺手安装整个原论文算法环境来跑上述验收。

交接时预期：

- **完整 unittest 84 项通过**：含24项 intake、真实 SQLite 并发/持久化、实际 HTTP 服务进程重启、原有 protocol/runner/worker/report tests。
- **benchmark smoke 54 项通过**：与完整套件有重叠，不能相加；不包含 intake。输出临时报告应 `status=passed`、`official_result=false`、`leaderboard_eligible=false`、`algorithm_correctness_checked=false`。
- 原机器曾因沙箱限制卡住 HTTP/异步线程测试，获准在沙箱外仅跑 localhost/临时文件后通过。新机器如有类似问题先定位沙箱权限，不删测试或无限放宽应用权限；必要时请求产品允许的测试执行权限。
- 少于预期数量、跳过关键测试或只返回 health 200 不算链路验证通过。测试数因后续新增而变化时更新这里和进度。

本地启动 API 的环境变量、安全说明和完整交互示例见 [API README](objview_submit/README.md)。不把真实管理员 token/邀请/session/数据库放入本文件。新机器生成新开发凭证即可；当前没有需要迁移的长期运行服务或正式用户数据库。

Docker 可用后的构建/双容器命令见 [Worker README](submission_worker/README.md)。其旧示例中 UID=1000 的 socket 必须改成新机器实际值；不要照抄绝对路径，更不要自动连接不明 daemon。真实容器验收通过前保持“未验证”。

## 7. 后续任务：推荐按这个顺序推进

以下是待执行 backlog 与验收边界，不代表已经实现。既定总体架构不用重新确认；新接口细节在对应小计划中设计并验证，别一次把生产系统全部铺开。

### P0：分页与本地研究者/管理员页面（无需 Docker）

- [ ] 为提交与审计列表加稳定、有限的游标分页（必要时加审核状态筛选）。当前只返回最近100条，旧待审任务可能被新拒绝记录挤掉；这是独立审查记录的非阻断缺口。
- [ ] 回归：一个旧待审核记录位于100条新记录之后时仍能到达；分页不能越权、漏/重复项；输入限制和安全错误不退化。
- [ ] 添加邀请兑换、方法/提交表单、我的提交详情/状态、管理员开始审核/批准/拒绝页面；批准页显式“等待预检，尚未运行”。
- [ ] 页面提交当前支持的元数据，不伪装成已上传源码；用户凭证不放 URL，不把管理员 token 打包进前端；为动态文本、CSP、会话存储/退出机制做实际设计与测试，不临时开启宽泛 CORS 来迁就页面。
- [ ] 浏览器或等效真实客户端串起邀请→提交→审核→刷新/重启后状态。保持只显示安全信息，无 hidden 原始日志/分数。GitHub Pages 仍不改域名/跳转地址占位。

### P1：持久化队列与可信开发链路（默认仍非正式）

- [ ] 建立 API 与独立 worker 的作业契约；持久化 claim/lease/heartbeat/timeout、幂等结果提交、失败状态、进程崩溃后的恢复/回收。
- [ ] Web 只写记录/队列，worker 单独拉任务；避免在 FastAPI 请求里直接调用 subprocess/Docker。worker 凭证、结果归属、重复或过期回传必须校验。
- [ ] 初期只派发管理员预配置的可信 dummy/smoke，不把 intake 的任意仓库/命令透传给 worker。无 Docker 时继续真实可信进程链路或契约测试，明确其与真实容器的差异。
- [ ] 增加真正的“页面/API→持久化任务→可信 benchmark/dummy→私有汇总→安全状态”验收报告；失败不得上榜，不接受用户伪造耗时，不让旧 worker 覆盖新作业。

### P2：Docker 真实隔离与 worker 生命周期补全

- [ ] 新机器核对 Docker、rootless 前置条件、cgroup v2/systemd、UID/GID、volume-subpath 支持；如需安装按用户确认和权限流程执行。
- [ ] 构建两个仓库可信 smoke 镜像，执行真实双容器流程并验证资源清理；保存镜像 digest、Docker/cgroup 版本、资源限制证据。
- [ ] 网络/DNS/宿主/跨任务访问 canary、隐藏目录不可见、只读根/观察、CPU/内存/PID/磁盘限额、异常与总超时、kill/restart/reaper。
- [ ] 单独测量容器启动/模型就绪，不把 benchmark 准备或 docker CLI 耗时当模型时间。更强 VM/microVM 边界留作接收不可信代码的上线门槛。

### P3：实际受邀研究者服务（需要部署配置）

- [ ] 真正 GitHub OAuth / 邀请身份绑定，安全会话与 CSRF/登录流程；团队/组织关联和账号生命周期。缺 OAuth 凭据时不伪造真实登录成功。
- [ ] PostgreSQL 迁移、对象存储、受限源码包/仓库快照、内容哈希/权重来源与不可变构建记录；所有网络抓取/压缩包解压都在明确限制下处理。
- [ ] 独立 Build Worker，不接触 hidden 数据；构建出口/时间/资源限额、依赖/镜像扫描、SBOM/provenance、私有 Registry、可信 image digest、协议 preflight。
- [ ] 生产持久化队列、权限隔离、备份恢复、分页与审计、全局限流、部署 TLS；不能把 development-only 服务直接反代上线。

### P4：hidden 榜单与公开注册门槛

- [ ] 固化数据分层/版本、不可变 benchmark 配置与硬件证据；独立 Final Audit；确认真实数据与访问授权。
- [ ] 正式团队/method-lineage/image digest 配额，区分算法失败消耗与平台故障退款，所有人工 override 可审计；限制重试造成的数据探测。
- [ ] 只返回聚合安全结果；成绩审核/发布/撤回、版本信息、公开榜单；最终域名确定后给 github.io 增加导航。
- [ ] 在公开注册前做隔离、提取/刷分、网络旁路、日志/制品导出、跨用户访问与恢复/滥用测试。不得声称完全消除测试集提取风险。

需要用户/环境补充但不阻止 P0 的信息：最终域名/TLS、GitHub OAuth 应用凭据、专用 builder/worker/GPU 配置、数据存储/授权、公开策略与运维负责人。秘密走配置/密钥管理，不写 Git。

## 8. 实施与交接纪律

- 修改前检查 branch/status 和相关计划，保留已有变动；迁移后的普通 clone 不强制再建一个 worktree，先判断环境与用户意愿。
- 用小步骤和真实边界测试推进；优先已有 `unittest`、FastAPI/httpx、SQLite 的验证体系。修复用回归测试证明，不用删除测试凑绿。
- 更新设计里的开发差异与 `progress.md`，明确“实现/模拟验证/真实验证/待完成”。不要只把新任务状态写在聊天里。
- 每个检查点提交代码与文档；推送/合并/部署分别尊重用户授权。此次授权仅推开发分支，不合并 main。
- 交付包和临时测试目录不提交回仓库。新开发数据库/密钥不随离线包传播；需要迁移正式状态时另做停服一致性备份与受控传输。
- 交接文档不是对未知新机器的保证：到新机器后先复跑第 6 节，再开始新增功能。
