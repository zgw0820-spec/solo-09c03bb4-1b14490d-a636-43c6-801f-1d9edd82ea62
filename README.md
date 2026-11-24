# 产线缺陷分数推理 · 单帧缺口补算

FastAPI 推理网页：内置固定权重的 NumPy 递推模型和两路样例。按相机 + 帧号提交
8 维特征向量，每路允许暂缺一帧；补交原帧后从缺口前的状态按帧序补算，补算追上
本路最新已出分帧后，才把修正状态交给后续推理，已发分数不改。

## 启动

```bash
docker compose up --build
```

打开 <http://localhost:8000> 。网页里可以：

- 「试跑：边补帧边进新帧」——一键跑完整演示（A 路缺第 4 帧、B 路缺第 8 帧，
  补算 A 的同时 B 和 A 自己都继续进新帧）；
- 每个相机卡片里单步「送入下一帧 / 标记缺失 / 补交原帧 / 补算故障演练」；
- 「手动提交一帧」表单按相机和帧号提交任意 8 维特征向量；
- 「操作记录」里看到每次出分、暂估、补算范围和修正量。

不用 Docker 也可以本地跑（Python ≥ 3.11）：

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## 规则怎么落实的

| 需求 | 实现 |
| --- | --- |
| 每路允许暂缺一帧 | 每相机维护 `last_frame`；下一帧跳过一个帧号即打开缺口，缺口未补时不允许再缺第二帧（`409`） |
| 用零向量代入、受影响分数标暂估 | 缺口帧以零向量推进，`estimated=true`；缺口未补期间后续帧从临时状态继续出分，同样标暂估 |
| 按相机和帧号提交特征 | `POST /api/cameras/{camera}/frames`，也可 `POST /api/frames` 平铺提交 |
| 补交原帧后从缺口前状态按帧序补算 | 开缺口时保存 `state_before`，补交时从它开始逐帧重放缺口帧到最新已出分帧 |
| 两路仍照常接帧、出分 | 每相机独立锁，补算在锁外做模型运算；其它相机和本相机新帧都不被阻塞 |
| 追上本路最新已出分帧才切换状态 | 重放目标取当前 `last_output_frame`，每帧检查是否又长了；追平后才一次性提交修正状态 |
| 已发分数不改 | `score` 字段只写一次；修正值写入 `revised_score/delta/revision`，界面两列并列 |
| 补算失败按原状态出分、可重试 | 重放期间不提交任何状态；失败返回 `422 {"retryable": true, "state_preserved": true}`，缺口和临时状态原样保留 |
| 固定权重 NumPy 递推模型 | `app/model.py` 里的权重字面量，`GET /api/model` 给出权重指纹，运行时不训练 |

模型：`h_t = tanh(W x_t + U h_{t-1} + b)`，`s_t = sigmoid(v·h_t + bias)`，
8 维输入、6 维隐状态、分数 0–1。样例数据在 `app/data/samples.json`，两路各 18 帧，
A 路第 6/7 帧、B 路第 5/10 帧带明显缺陷特征，方便观察分数跳变。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health`、`/api/model`、`/api/samples` | 健康检查、模型信息、样例数据 |
| GET | `/api/status` | 两路汇总（缺口、暂估数、补算数、状态指纹） |
| GET | `/api/cameras/{camera}/state` | 状态、分数、修正值、事件 |
| POST | `/api/cameras/{camera}/frames` | `{frame_no, features[8]}` 提交一帧 |
| POST | `/api/cameras/{camera}/missing` | `{frame_no}` 声明该帧缺失 |
| POST | `/api/cameras/{camera}/fill` | `{frame_no, features[8], pace_ms?, simulate_failure?}` 补交并补算 |
| POST | `/api/reset` | `{confirm: true}` 清空（补算中的任务会被取消且不提交） |

`pace_ms` 只在演示时放慢补算便于观察；`simulate_failure` 是演示用故障注入，
在重放第一帧后失败，用来验证“失败仍按原状态出分、可重试”。

## 验证

```bash
python -m pip install -r requirements-dev.txt
python -m pytest                 # 20 个单元/接口测试
python scripts/check_interleaved.py   # 交付核对：边补帧边进新帧的端到端场景
```

`scripts/check_interleaved.py` 自己拉起一个本地 uvicorn，逐条打印 PASS/FAIL：

1. 缺口期间照常出分、标暂估；
2. 补算期间另一路和本路新帧都继续出分；
3. 补算目标从 #8 长到 #9，追平后才提交；
4. 已发分数逐帧不变，修正值与“无缺口重算”逐位一致；
5. 失败重试：状态指纹、缺口、已发分数都不变，之后仍能出分，重试追平。

## 边界与未完成

- 状态只存在进程内存里，重启即丢；没有持久化、没有多副本共享状态。
- 相机只支持两路（0/1），是配置常量而不是数据库配置。
- 每路同一时间只允许一个缺口；缺口窗口内再出现第二个零向量帧时，补算会明确
  拒绝（`422`）而不是猜，需先人工处理数据。
- `simulate_failure` 是演示开关，生产部署前应移除或加权限。
- 没有鉴权、限流和审计日志落盘；事件只在内存里保留最近 500 条。
- 没有做真实产线帧率/吞吐压测，只有并发行为测试。
