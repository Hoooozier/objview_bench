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

- 本机没有 Docker 命令；示例镜像未构建，双容器网络、只读挂载、GPU 隔离、资源限制和逃逸防护未实测。
- 未启动真实数据集/GPU 渲染评测；既有完整依赖栈需要专用 worker 验收。
- 没有上传网站、GitHub 邀请登录、数据库/持久化提交配额、构建器、调度队列、全局反复提取防护或榜单发布。
- 单 episode 匿名与查询限额只减少直接泄漏，不保证阻止反复提取测试数据或基于公开对象识别刷分。
- 当前 ready 计时包含部分 benchmark 准备时间；容器启动/模型就绪耗时应由下一阶段控制器独立测量。
- HTTP server 的 watchdog 是协作式退出，不是 native/GPU 调用的硬资源边界。

## 下一阶段实施顺序

1. **Phase B：可信执行控制器**。先实现可测试的 job 状态机、不可变镜像摘要/数据版本输入契约、独立 build/run 路径、命令白名单与失败清理；在有 Docker 的专用 worker 验收双容器真实链路。容器执行默认关闭，未完成隔离前不运行研究者代码。
2. **Phase C：受邀用户提交站后端**。GitHub 登录与邀请名单、团队级额度、上传对象存储、源码/权重哈希、人工审核、任务状态；Web 不挂载 Docker socket。正式配额必须持久化并覆盖失败尝试。
3. **Phase D：前端与发布**。提交/状态页面、只显示安全错误与聚合结果、管理员审核；确定真实域名后在 github.io 增加链接，不迁移静态站。
4. **Phase E：公开注册前安全验收**。强化 VM/microVM 边界、网络白名单/旁路测试、跨作业访问与数据导出测试、审计、依赖和镜像扫描、限流与恢复演练。

继续开发不需要重新讨论上述已确认架构。真正部署时才需要维护者提供专用 worker 环境、GPU 配置、域名和 GitHub OAuth 凭据；这些不能写入仓库。
