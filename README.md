# LLM Router

一个面向 llama.cpp 与 KVMem 的本地推理网关，对外提供统一的 OpenAI 兼容接口，并按请求的模型自动启停和切换引擎。

## 功能

- **统一入口**：`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 等 OpenAI 兼容接口，客户端只需更换 `base_url`。
- **多引擎**：同时管理 llama.cpp（`llama-server` 路由模式）与 KVMem（`serve.server --engine strata`）两类后端。
- **按需加载**：同一时间只保留一个引擎和一个模型；请求哪个模型就启动哪个，空闲时可手动卸载，显存随用随取。
- **请求队列**：切换模型期间到达的请求会排队等待，不会直接失败；停止引擎前会等待进行中的生成结束。
- **控制台**（`/`）：查看服务阶段、活动模型、请求队列、运行时长；加载/卸载模型、停止引擎、重新扫描模型目录；编辑每个模型的采样参数；配置服务端 API Key。
- **新增模型**：在控制台填写模型类型（llama.cpp / KVMem）、GGUF 路径、可选的视觉 mmproj 路径和采样参数。提交后自动完成：路径校验、GGUF 头解析（架构、上下文上限、视觉编码器）、KV 精度选择、写入引擎配置、启动引擎、文本与图片冒烟测试；任一步失败会回滚已写入的配置。
- **删除模型**：只删除引擎配置，不移动或删除模型文件；如果删除的是当前活动模型，会先停止引擎。
- **默认策略**：上下文默认取 GGUF 中的上限；KV 精度优先使用 k8v8（`cache-type-k/v = q8_0` 或 `--kv q8_0`），引擎不支持时回退其默认值；视觉 mmproj 默认走 CPU；其余采样参数以引擎参考模型（llama.cpp 为 LFM3-Omni-2B，KVMem 为 Ornith）为默认。
- **聊天 UI**（`/ui`）：内置多模型聊天界面，支持文本、图片、音频和视频输入，API Key 仅保存在浏览器本地。
- **鉴权**：可选服务端 API Key，保护 `/v1` 与聊天 UI 的 `/props`；控制台与 `/models` 不鉴权，适合可信内网使用。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/router/status` | 网关与引擎状态 |
| GET | `/v1/models` | 模型列表 |
| GET | `/router/models/manage` | 引擎默认值与新增表单选项 |
| POST | `/router/models/add` | 异步新增模型，返回任务 ID |
| GET | `/router/models/job?id=` | 查询新增任务进度 |
| POST | `/router/models/delete` | 删除模型配置 |
| POST | `/router/load` / `/router/unload` | 加载 / 卸载模型 |
| POST | `/router/stop` | 停止引擎 |
| POST | `/router/reload` | 重新扫描模型目录 |
| GET/POST | `/router/params` | 读取 / 保存采样参数 |
| GET/POST | `/router/auth` | 读取 / 设置服务端 API Key |

## 运行

```bash
./START-router.sh
```

默认监听 `0.0.0.0:8080`，控制台为 `/`，聊天 UI 为 `/ui`。网关通过 `config.json` 指定引擎命令与模型目录，通过 `models.ini`（llama.cpp）和 KVMem 的 `strata-*.json` 描述模型。

## 许可

[MIT](LICENSE)
