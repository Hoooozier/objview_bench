# ObjView Submit：本地开发版提交 API

网站首页公开展示论文基线结果，无需 token；只有进入提交流程才要求提交邀请 token，个人记录和管理操作仍需认证。

本阶段已提供独立控制面：管理员邀请 → 研究者兑换会话 → 登记方法 → 提交源码元数据 → 人工审核 → 持久化查询。**不执行提交的算法，不拉取仓库，不构建镜像，不产生评测成绩。** GitHub Pages 不受影响。

代码暂存在 benchmark 的独立开发分支，便于一次提交审阅；`objview_submit` 是独立 Python 包，不导入 benchmark 或 worker，可单独复制包并安装自己的依赖运行。正式服务拆仓不改变这个边界。

## 启动

Python 3.10；已有依赖时无需重装。确需安装时建议在虚拟环境中执行：

```bash
python3 -m pip install -r objview_submit/requirements.txt
```

依赖版本是本机兼容性快照，不是经过安全扫描的生产锁文件。

在仓库根目录运行：

```bash
export PYTHONDONTWRITEBYTECODE=1
export OBJVIEW_SUBMIT_ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf 'Admin token: %s\nHome: http://127.0.0.1:8090/ui\nAdmin: http://127.0.0.1:8090/ui/admin\n' "$OBJVIEW_SUBMIT_ADMIN_TOKEN"
python3 -m objview_submit --development-only --state-dir objview_submit/.state --port 8090
```

仅监听 `127.0.0.1`，不接受 `--host`。管理员 token 不从命令行参数接收、不写入数据库。需要重启后继续使用同一管理凭证时，在受保护的环境配置中保存它；重新生成环境变量即轮换管理凭证，不会撤销已有研究者会话。不要把凭证写入 Git、聊天记录或截图。

状态目录自动以 0700 创建，父目录必须已存在；已存在但不属于当前用户或允许其他用户访问的目录会拒绝启动，不会自动更改权限。数据库文件 0600，目录/数据库本身不接受符号链接。推荐以上被 `.gitignore` 排除的 `.state` 路径；自定义路径请放在仓库外。数据库含身份标签和源码引用，应视为敏感数据；SQLite 未加密。

健康检查：`GET http://127.0.0.1:8090/healthz`，返回 `mode=development_intake`、`execution_enabled=false`。此接口只检查服务存活，不代表数据库、worker 或 Docker 可用。

## 本地使用示例

下面的 Python 客户端在另一个已安全设置同一 `OBJVIEW_SUBMIT_ADMIN_TOKEN` 的终端执行。示例只在内存中使用返回的研究者 token，不打印它；实际邀请由管理员通过私下渠道交付，不放到 URL 中。

```python
import os
import httpx

base = 'http://127.0.0.1:8090'
admin = {'Authorization': 'Bearer ' + os.environ['OBJVIEW_SUBMIT_ADMIN_TOKEN']}
with httpx.Client(base_url=base, trust_env=False, timeout=10) as client:
    def post(path, payload, headers=None):
        response = client.post(path, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    invitation = post('/v1/admin/invitations',
                      {'label': 'researcher-demo', 'ttl_seconds': 3600}, admin)
    session = post('/v1/invitations/redeem', {'token': invitation['token']})
    researcher = {'Authorization': 'Bearer ' + session['access_token']}
    method = post('/v1/methods', {'name': 'Dummy pipeline method'}, researcher)
    submitted = post('/v1/submissions', {
        'method_id': method['id'],
        'repository': 'https://github.com/example/algorithm',
        'commit_sha': 'a' * 40,
        'dockerfile': 'Dockerfile',
        'entrypoint': '/opt/objview/run.sh',
        'protocol_version': 'objview-interaction/v1',
    }, {**researcher, 'Idempotency-Key': 'demo-submission-1'})
    review = '/v1/admin/submissions/' + submitted['id'] + '/review'
    post(review, {'decision': 'start', 'expected_version': 0}, admin)
    approved = post(review, {'decision': 'approve', 'expected_version': 1}, admin)
    print(approved['id'], approved['status'])
```

示例仓库和 SHA 只是格式有效的元数据，**未校验它们是否存在或包含启动脚本**。这不是构建/协议预检通过的证据。重复执行示例会创建新方法；如果复用同一 Idempotency-Key 却改了 method_id，会正确返回 409，应查询已有提交或为新提交使用新 key。

## API 契约

除健康检查和邀请兑换外，必须传 `Authorization: Bearer <token>`。所有 POST 使用 `Content-Type: application/json`，不接受额外字段；撤销、禁用与退出传 `{}`。研究者不能调用管理员路由，管理凭证不能伪装成研究者提交。

| 权限 | 方法与路径 | 请求/含义 |
| --- | --- | --- |
| 管理员 | `POST /v1/admin/invitations` | `{label, ttl_seconds}`，TTL 1–604800 秒，返回一次性 token |
| 无需会话 | `POST /v1/invitations/redeem` | `{token}`，返回 24 小时会话；身份由邀请绑定，不接收客户端身份/角色 |
| 研究者 | `GET /v1/me` | 当前身份；label 是管理员标签，非验证过的 GitHub 账号 |
| 研究者 | `POST /v1/session/logout` | 撤销当前会话 |
| 管理员 | `POST /v1/admin/invitations/{id}/revoke` | 撤销尚未兑换的邀请；已兑换会话需禁用用户 |
| 管理员 | `POST /v1/admin/users/{id}/disable` | 禁用用户并撤销所有会话；开发版不提供重新启用 |
| 研究者 | `POST /v1/methods` | `{name}`，服务器生成 method_id，不能认领别人的方法 |
| 研究者 | `GET /v1/methods` | 自己的方法 |
| 研究者 | `POST /v1/submissions` | 上述六字段元数据；必需 `Idempotency-Key` 请求头 |
| 研究者 | `GET /v1/submissions`、`GET /v1/submissions/{id}` | 只可看自己的提交，跨账户详情返回 404 |
| 研究者 | `GET /v1/submissions/{id}/evaluation` | 只返回安全的可信 smoke 队列状态和允许的聚合计时；未批准提交返回 404 |
| 管理员 | `GET /v1/admin/submissions` | 提交审核列表；支持 `limit`（1–100）、`cursor` 和状态筛选 |
| 管理员 | `POST /v1/admin/submissions/{id}/review` | `{decision, expected_version}`；decision 为 start/approve/reject |
| 管理员 | `POST /v1/admin/evaluation-batches/release` | `{}`，原子发布当前收集批次，返回 ID 和 job_count |
| 管理员 | `GET /v1/admin/evaluation-batches` | 最近批次，limit 默认 50、最大 100；RELEASED 只表示已发布，不代表完成 |
| 管理员 | `GET /v1/admin/audit` | 固定事件码、操作者、目标 ID 和服务端时间；支持 `limit`（1–100）与 `cursor`；不含请求体/凭证/源码日志 |

提交列表使用按 `created_at,id` 倒序的稳定游标，审计列表使用递减事件 ID；响应在还有下一页时带 `next_cursor`。游标绑定到账户和筛选条件，仍须通过原有权限检查，不能越权读取。默认每页 50 条，单页最多 100 条。

## 本地页面

开发服务提供无 JavaScript 的 loopback 页面：`GET /` 跳转到公开 `/ui` 论文结果首页；浏览与筛选无需 token，已登录也不会自动跳走。`/ui/submit` 解释提交材料并提供邀请 token 验证入口，`/ui/researcher` 创建方法、提交元数据和查看状态，`/ui/admin` 登录管理员并开始/批准/拒绝审核。页面只接受当前 API 支持的元数据，不上传源码；所有固定页面文案为英文，页面展示按期收集通知，批准后显示“Collecting — awaiting batch validation”。管理员另有“Release collected batch”按钮和最近批次记录。研究者和管理员凭证放在 `HttpOnly`、`SameSite=Strict` 的短期 cookie 中，不放入 URL；`POST /ui/logout` 会撤销研究者会话并清除 cookie。动态标签、方法名和 manifest 字段均 HTML 转义，页面沿用 no-store、CSP 和 no-referrer 响应头。这里的“公开浏览”指页面无需认证；服务仍只监听 loopback，不构成公网部署或 OAuth。

源码当前只接受 `https://github.com/owner/repository` 与小写 40 位十六进制 commit SHA；不接受分支名、带凭据/查询串的 URL、绝对/穿越 Dockerfile 路径、自定义命令或协议版本。body 最大 16 KiB、读取 5 秒，重复 JSON 键、NaN、Infinity 会被拒绝。平台不接受用户提供的指标、耗时、owner_id 或执行状态。

此接口是设计 §6 的最小 intake 表单，不是完整 `submission.yaml` 解析器。`benchmark/examples/submission/submission-manifest.example.json` 仍是旧 smoke 构建说明草稿，不能直接 POST 到此接口；后续可信构建器负责转换完整清单、读取真实文件并计算摘要。

## 论文结果展示与更新

首页展示 9 种基线算法，数据来自用户提供的 `pannips/main_post_rebuttal.tex`。该修订稿的三张仿真完整表与 `main.tex` 相同，另包含真实 RGB-D 回放结果。运行时只读取独立的 `objview_submit/data/paper_results.json`，不挂载或公开论文目录，不查询提交数据库/worker 结果。

- 仿真实验：76 条“算法 × 视空间 × 停止协议”结果；支持 whole/quarter、K=5/10/30、Automatic（MS@0.02 或原生计划结束）、补充 MS@0.01/0.03。
- 主表：Views、NSC@0.02、SC@0.02、Path Cost、论文 PR 秒区间；展开查看 NSC/SC 的 0.01/0.03 阈值与 CD。
- 按 NSC 排序，或按论文口径先保留 NSC≥0.95 再按路径排序；不生成跨协议总榜，不给未报告设置补零，已舍入同值不细分名次。
- 真实 RGB-D：14 条、8 种适用算法、25 个配置，单独展示 Views、CRC@1cm/2cm 和路径；排除不适配固定回放视点的 BENBV-Net。CRC 不是仿真 SC/NSC，NBV-Net 早停的低路径不能当效率优势。
- 论文 PR 是算法自报时间，保留原始区间；与新平台 benchmark 单调时钟的可信计时分开。论文已有实验不混入 smoke 或用户待审核结果。
- 首页提供匿名 JSON 下载 `/ui/paper-results.json`，含来源文件名、SHA256 和表标签；不包含私有提交、用户身份、凭证或上传源码。

在仓库根目录显式更新/核对数据（只解析白名单表，不运行 LaTeX）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m objview_submit.import_paper_results
PYTHONDONTWRITEBYTECODE=1 python3 -m objview_submit.import_paper_results --check
```

原论文只在导入时需要；正常启动和完整测试使用独立数据快照，不依赖 `pannips` 或浏览器安装。修改源码或重新导入后重启服务即可看到更新。样式位于 `site.css`，由固定 `/ui/assets/site.css` 路由提供；CSP 仅额外允许同源样式，不开放脚本、外部 CDN 或通用文件目录。

## 状态、配额与幂等

```text
SUBMITTED --start--> ADMIN_REVIEW --approve--> APPROVED_FOR_PREFLIGHT
                                \--reject--> REJECTED
```

`APPROVED_FOR_PREFLIGHT` 是开发版等待状态，先进入收集池，不创建可执行作业；不会执行提交者上传的代码，也不是 BUILDING、RUNNING、已评测或可上榜。所有提交响应带 `official_result=false`、`leaderboard_eligible=false`。

管理员到统一验证时间后，在页面发布批次，或调用 `POST /v1/admin/evaluation-batches/release`（JSON `{}`）。发布事务为当前全部已批准且未分批的提交分别创建固定的 `trusted_smoke_v1` 作业，记录相同 batch ID；之后批准的提交留待下批。无可发布提交时返回 `NO_ELIGIBLE_SUBMISSIONS/409`，不会创建空批次或重复作业。当前手动发布，没有自动月度调度；每月一次只是可选周期。各任务独立运行，批次不引入算法间交互，也不要求同时占用硬件。

独立 worker 通过 SQLite 领取短租约、续租并提交一次性 allowlist 结果；租约过期会重新排队，旧 worker 的回传会被拒绝，终态回传可安全幂等重试。作业执行器只收到作业 ID/提交 ID/固定类型，不会收到仓库、Dockerfile、启动命令或任意用户输入。当前队列仍是开发版契约，只有可信 dummy/smoke 执行器可接入，失败状态不产生正式成绩。

- 每个账户滚动 24 小时最多 5 次新提交；被拒绝的提交仍计数。
- 每个账户最多 3 个非 REJECTED 提交，包含已批准等待预检的记录；开发版批准后不会自动释放名额。
- 每账户最多 10 个方法；最多同时存在 100 个未过期/未使用/未撤销邀请。
- 相同账户和 key 的重试返回已有提交的**当前状态**，不增加计数；换 payload 返回 `IDEMPOTENCY_CONFLICT/409`。key 是 1–64 位字母、数字、`-`、`_`。
- 管理员操作必须携带读到的 version。并发审批只有一个成功，其余 `STALE_VERSION/409`，应重新读取后由管理员判断，不能盲目自动重试。
- SQLite 事务把额度检查、插入/状态变更和审计写入放在一起。重新启动服务不清空配额/状态。
- 同一 label 的新邀请为已有账号续发会话，不重置账号/提交额度；label 区分大小写，管理员需维护稳定映射。多个 label 不是经过核实的不同自然人或团队。

正式方案中的团队配额、方法谱系关联、同一 image digest 的 hidden 次数限制，以及故障配额退还**还未实现**。不能把当前入口限额当成防刷分已完成；隐藏评测入口仍然关闭。

## 安全边界与运维限制

- 邀请和会话只保存 SHA256 token 摘要，原始 token 只在创建/兑换响应出现。邀请 token 是持有者凭证，不验证收件人；仅用于本地开发，不能替代 GitHub OAuth。
- CLI 固定 loopback、禁用代理头信任和访问日志，无 CORS、无 Swagger/OpenAPI；所有响应 no-store，错误不返回 SQL、文件路径、请求值或原始异常。
- 每进程最多 16 个应用活动请求；兑换每进程每分钟最多 60 次。该临时限流全局共享、重启重置，可能被耗尽，**不是公网/多实例限流方案**。
- 控制面无 Docker/socket/数据集依赖，不接受源码压缩包、不发出仓库请求；恶意 Dockerfile 在本阶段只被作为一个相对路径记录。
- 不支持公开访问/反向代理部署。`--development-only` 是明确授权开关，不是安全沙箱；不要通过端口转发、隧道或改源码暴露到互联网。
- 只有本地 additive schema 升级：旧版未分批 QUEUED 作业保留 ID/历史，等待显式发布；已有运行/终态结果保留。暂无通用数据库迁移/备份/审计防篡改、数据加密、会话设备管理、账号恢复或双管理员身份。SQLite 仅供单机小规模开发；schema 不认识时拒绝启动，不自动降级。
- 备份时停止服务再复制整个私有状态目录到同样受保护的位置；不要只拷贝正在使用的数据库文件。停止/启动不撤销研究者会话。数据目录删除会丢失记录与配额，绝不能用于绕过额度。

## 验证

仓库根目录：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmark/tests -p 'test_intake_*.py' -v
PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 -m unittest discover -s benchmark/tests -v
```

测试使用临时数据库，覆盖一次性邀请并发兑换、凭证散列/过期/撤销、所有权隔离、配额并发、幂等、审核竞争、SQL/内部异常脱敏、真实服务进程启动和重启后查询。进程测试只关闭自身启动的服务，不占用长期固定端口。

原有 `benchmark/smoke_submission.py` 仍单独验证 benchmark/worker 链路，不包含本 API，也不声称网站→构建→评测已贯通。按用户要求，不验证算法科学指标的正确性。

下一步：将已发布批次接到独立可信 worker 进程；在提供 OAuth 配置、隔离构建环境和真实 Docker 验收之前，保持上传执行与 hidden 关闭。


## English guide and downloadable API example (2026-10-26)

All fixed UI copy, including errors, forms and admin pages, is English. User-provided names and labels remain unchanged. Public browsing still requires no token.

- `/ui/guide`: producer–consumer workflow, endpoint contract, observation formats and enforced limits.
- `/ui/submit`: Dockerfile example and submission invitation entry.
- `/ui/examples/{name}`: only `Dockerfile`, `run.sh`, `algorithm.py` and `submission_client.py` can be downloaded. These are public package assets; no arbitrary path or benchmark data is exposed.

Save the four example files into one directory. The Dockerfile uses the requested internal base, installs `httpx`, copies the client/algorithm/script to `/opt/objview`, and runs as a non-root user. Match the base architecture to the worker and configure an internal Python package mirror or offline dependencies. The current ARM64 host cannot execute the installed AMD64 base; this example has not been Docker-built here.

The example consumes only the current published observation through the private episode API and submits move/stop actions. The worker no longer mounts observation or benchmark volumes into the algorithm. Tests run the canonical example as a real process and verify that downloadable copies match it. This does not enable execution of arbitrary submissions.

The algorithm does not need to implement every endpoint. Its required flow is `GET /v1/episode`, `POST /v1/ready`, `GET /v1/steps/next`, `GET /v1/observations/{step_index}/{asset}` and `POST /v1/actions`. Interface discovery, status polling and feasibility checks are optional; the benchmark validates submitted actions regardless.

Gateway limits: 20 requests/second, burst 40, total `128 + 64 × max_steps`, 8 concurrent requests with at most one long poll and two downloads, 8 MiB/file, 10 downloads and 32 MiB/step, 512 MiB/episode. Slots remain held through response delivery. Invalid and unauthenticated requests consume budgets too. See [protocol](../benchmark/SUBMISSION_PROTOCOL.md) for the full contract and isolation limitations.

Intake HTML forms now share the API's 16 KiB/5-second read limits. Invitation API redemption, browser redemption and admin login share a 60-attempt/minute limiter; this is an in-process development guard, not a production distributed perimeter.
