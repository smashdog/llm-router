# LLM Router

一个面向 llama.cpp、KVMem 与 NInfer 的本地推理网关，对外提供统一的 OpenAI 兼容接口，并按请求的模型自动启停和切换引擎。

## 功能

- **统一入口**：`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 等 OpenAI 兼容接口，客户端只需更换 `base_url`。
- **多引擎**：同时管理 llama.cpp（`llama-server` 路由模式）、KVMem（`serve.server --engine strata`，polystrata）与 NInfer（`ninfer-serve` 单模型模式）三类后端。
- **按需加载**：同一时间只保留一个引擎和一个模型；请求哪个模型就启动哪个，空闲时可手动卸载，显存随用随取。
- **请求队列**：切换模型期间到达的请求会排队等待，不会直接失败；停止引擎前会等待进行中的生成结束。
- **控制台**（`/`）：只展示引擎状态与模型列表；加载/卸载模型、停止引擎、重新扫描模型目录；编辑每个模型的采样参数。右上角“管理”进入管理页，左侧菜单提供“引擎路径”和“API 密钥管理”。
- **引擎路径**：管理页为 llama.cpp、KVMem 与 NInfer 三个固定槽位分别配置可执行文件、工作目录、模型目录和预设/模板文件。保存时先校验路径并轻量探测二进制（llama.cpp 与 NInfer 执行 `--version`，KVMem 导入 `serve.server`），再备份旧配置并原子写入 `config.json`；缺失的槽位可直接创建，运行中的引擎修改路径后会标记“需重启”，可一键重启生效。NInfer 槽位还会在二进制旁的 `lib` 目录存在时自动写入 `LD_LIBRARY_PATH`。
- **无引擎启动**：启动时若没有配置任何引擎，控制台不显示状态区与模型区，只保留引导卡片，点击即可进入管理页创建引擎；创建后无需重启网关，模型列表会自动出现。
- **模型参数**：llama.cpp 模型可修改上下文长度、K 量化精度和 V 量化精度（写入 `models.ini` 预设），保存后自动重启引擎生效；KVMem 模型固定使用模型上下文与 q8 量化，不显示这些参数；NInfer 的上下文与 KV 参数由模型模板决定，只开放采样参数，且 `repetition_penalty` 只能为 1、不支持 `repeat_last_n`。
- **新增模型**：在控制台填写模型类型（llama.cpp / KVMem / NInfer）、模型路径、可选的视觉 mmproj 路径、视觉运行位置（CPU / GPU）和采样参数；llama.cpp 还可设置上下文长度与 K/V 量化精度。提交后自动完成：路径校验、GGUF 头解析（架构、上下文上限、视觉编码器）或 `.ninfer` 检查、KV 精度选择、写入引擎配置、启动引擎、文本与图片冒烟测试；任一步失败会回滚已写入的配置。NInfer 采用单模型模式：每新增一个模型就复制模板生成一份 `ninfer-<name>.json`，切换 NInfer 模型时网关会重启该引擎。
- **删除模型**：只删除引擎配置，不移动或删除模型文件；如果删除的是当前活动模型，会先停止引擎。
- **默认策略**：上下文默认取 GGUF 中的上限（可改小，不能超过上限）；llama.cpp 的 K/V 量化默认均为 `q8_0`（k8v8），可分别选择 `f16`、`bf16`、`q4_0`、`q4_1`、`q5_0`、`q5_1`、`iq4_nl`、`f32` 等引擎支持的类型；KVMem 固定 `--kv q8_0`；NInfer 的上下文、KVMem 参数和 KV 量化全部继承模板。视觉 mmproj 默认走 CPU，可切换 GPU（NInfer 暂不支持视觉）；其余采样参数以引擎参考模型（llama.cpp 为 LFM3-Omni-2B，KVMem 为 Ornith，NInfer 为模板）为默认。
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
| GET/POST | `/router/engines/paths` | 读取 / 保存引擎启动路径；POST 传 `create` 可创建缺失的固定引擎槽位 |
| POST | `/router/engines/restart` | 按当前 `config.json` 重启指定引擎 |
| GET/POST | `/router/params` | 读取 / 保存采样参数；llama.cpp 可同时保存上下文长度与 K/V 量化精度 |
| GET/POST | `/router/auth` | 读取 / 设置服务端 API Key |

## 安装与运行

要求 Python 3.10+，唯一第三方依赖是 `requests`。Debian/Ubuntu 的系统 Python 受 PEP 668 保护，推荐用项目内虚拟环境（`START-router.sh` 会自动优先使用 `.venv`）：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./START-router.sh
```

若系统已经能 `import requests`，也可以直接运行 `./START-router.sh`。需要指定其它解释器时设置 `ROUTER_PYTHON`。

首次启动会在当前目录自动生成最小 `config.json`（不含引擎），控制台显示“未配置引擎”引导，进入管理页填写引擎路径即可，无需重启网关。默认监听 `0.0.0.0:8080`，控制台为 `/`，聊天 UI 为 `/ui`；临时换端口可用 `./START-router.sh --host 127.0.0.1 --port 8081`（不写入配置文件）。`./START-router.sh --check` 只扫描并打印模型列表后退出。

启动脚本会检查 Python 版本与 `requests`。网关通过 `config.json` 指定引擎命令与模型目录，通过 `models.ini`（llama.cpp）、KVMem 的 `strata-*.json` 和 NInfer 的 `ninfer-*.json` 描述模型。

NInfer 需要一份可用模板（管理页的“模板配置”），新增模型时从模板复制参数，并把模型 ID、`.ninfer` 路径和 `--model-id` 写入新的 `ninfer-<name>.json`。模板里的 `env`（如 `CUDA_VISIBLE_DEVICES`、`LD_LIBRARY_PATH`）会随每份模型配置一起继承。

## 许可

[MIT](LICENSE)
