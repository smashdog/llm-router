# LLM Router

一个面向 llama.cpp、Strata、KVMem、NInfer 与 KVMem + llama.cpp 的本地推理网关，对外提供统一的 OpenAI 兼容接口，并按请求的模型自动启停和切换引擎。

## 功能

- **统一入口**：`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings` 等 OpenAI 兼容接口，客户端只需更换 `base_url`。
- **多引擎**：同时管理 llama.cpp（`llama-server` 路由模式）、Strata（独立 `serve.server --engine strata` 单模型服务）、KVMem（polystrata）、NInfer（`ninfer-serve` 单模型模式）与 KVMem + llama.cpp（独立的 `llama-kvmem-server` 单模型模式）五类后端。
- **按需加载**：同一时间只保留一个引擎和一个模型；请求哪个模型就启动哪个，空闲时可手动卸载，显存随用随取。
- **请求队列**：切换模型期间到达的请求会排队等待，不会直接失败；停止引擎前会等待进行中的生成结束。
- **控制台**（`/`）：只展示引擎状态与模型列表；加载/卸载模型、停止引擎、重新扫描模型目录；编辑每个模型的采样参数与视觉运行位置（CPU / GPU）。右上角“下载”按钮以红色角标显示进行中的下载数量（无下载时不显示），点击浮层列出正在下载的模型，点任一项查看进度并可停止；“管理”进入管理页，左侧菜单提供“引擎”“API 密钥管理”“代理”和“模型下载”。
- **启动环境**：`START-router.sh` 默认在项目目录创建或复用 `.venv`，按 `requirements.txt` 的内容哈希安装依赖；随后检查 `hf`（Hugging Face CLI）与 `modelscope`，缺失时尝试安装到 venv，安装失败会打印明确错误但不阻止网关启动（下载页会标记工具不可用）。设置 `ROUTER_PYTHON` 时改用指定解释器，只检查不安装。
- **引擎**：管理页按一级分类展示引擎槽位：`llama.cpp` 分类下为原生 llama.cpp；`Strata` 分类下为独立 Strata 服务；`kvmem` 分类下依次为 llama.cpp（KVMem 补丁版）、polystrata 与 ninfer。每个槽位分别配置可执行文件、工作目录、模型目录和预设/模板文件。保存时先校验路径并轻量探测二进制（llama.cpp 与 NInfer 执行 `--version`，Strata 与 KVMem 导入 `serve.server`，KVMem + llama.cpp 执行 `--kvmem-capabilities` 并要求返回 llamacpp 后端），再备份旧配置并原子写入 `config.json`；缺失的槽位可直接创建，未配置时输入框保持空白，仅以灰色占位提示约定路径（可点“填入建议路径”后修改），不会把建议值当成已保存配置；创建时若模型目录不存在会自动创建，llama.cpp 与 KVMem 还会生成空的 `models.ini` / `models.json`，Strata 与 NInfer 的模板文件必须已存在，KVMem + llama.cpp 会自动生成 `kvmem-llama.template.json`；运行中的引擎修改路径后会标记“需重启”，可一键重启生效。NInfer 与 KVMem + llama.cpp 槽位还会在二进制旁的 `lib` 目录存在时自动写入 `LD_LIBRARY_PATH`。新增模型时先选一级分类，再在右侧子引擎列表中选择具体引擎（默认选中第一项）。
- **引擎更新**：管理页“引擎”面板顶部提供“检查更新”和“使用代理”（未配置代理时隐藏且强制直连）。检查时在各引擎工作目录执行 `git fetch`，与上游比较提交数并列出落后提交；若工作目录是子模块（detached HEAD），自动改用父仓库。发现新版本后卡片出现“安装更新”：先备份当前二进制为 `<binary>.bak-<时间戳>`，再校验工作区干净后执行 `git merge --ff-only`，并根据二进制旁的 `CMakeCache.txt` 推断构建目录，用 `cmake --build` 以 Release 重新编译目标；任一步失败立即停止，编译或探测失败时恢复二进制备份并回滚源码到更新前提交。KVMem 为纯 Python 引擎，只更新源码；运行中的引擎需重启后才会使用新二进制。
- **无引擎启动**：启动时若没有配置任何引擎，控制台不显示状态区与模型区，只保留引导卡片，点击即可进入管理页创建引擎；创建后无需重启网关，模型列表会自动出现。
- **模型名称**：新增或编辑模型时可填写不限长度的“模型名称”（显示名；新增对话框在两条路径下方使用单行输入框，编辑时使用可换行的多行框，换行会合并为空格），控制台卡片标题与聊天 UI 的模型选择器优先显示它，完整模型 ID 仍保留在卡片下方和 API 中。名称保存在网关根目录的 `model-meta.json`（可用 `model_meta_file` 改路径），按模型 ID 覆盖各引擎从 GGUF / 配置读到的默认名称；留空保存即恢复默认名称，名称变更不需要重启引擎。显示名同时注册为模型别名，可直接用于 `/v1` 请求的 `model` 字段。
- **模型参数**：增加/编辑模型时分为“基本参数”和“更多参数”两页；KVMem + llama.cpp 模型另有“KVMem 引擎”页，按“预算与保留、检索与选择、内存与存储、运行与并发、设备与多卡、推测解码（MTP）、思考、视觉”分组列出 53 项 `llama-kvmem-server` 进程级参数（预算 B 仍在基本参数页）；Strata 的进程级参数并入“更多参数”页，按“模型与包、上下文与 KV、推测解码（MTP / 查表）、专家缓存与内存、预填充与并发、提示与会话缓存、运行与显存、视觉”分组，写入模型自己的 `strata-*.json`（修改 `args`，清除 `user_args` 中同名开关并刷新 `args_written`），保存后自动重启引擎。基本参数保持原有内容：llama.cpp 可修改上下文长度、K/V 量化精度（写入 `models.ini` 预设，保存后自动重启引擎），KVMem 的上下文在新增时确定（留空读取 GGUF）、之后不能在这里修改且 KV 精度固定 q8_0，NInfer 的上下文与 KV 参数由模型模板决定，KVMem + llama.cpp 的上下文与 KV 精度在新增模型时写入、之后不能在这里修改，KVMem 预算可随时改；有视觉编码器的模型还可切换视觉运行位置（CPU / GPU）：llama.cpp 写入 `mmproj-offload = off/on`，KVMem、KVMem + llama.cpp 与 Strata 写入配置文件的 `vision.gpu`，保存后自动重启引擎；由目录扫描发现、没有可编辑配置的模型只读，NInfer 不支持视觉。五个引擎都开放温度、`top_p`、`top_k`、`min_p`、重复/存在/频率惩罚（NInfer 的 `repetition_penalty` 只能为 1，NInfer 与 KVMem + llama.cpp 不支持 `repeat_last_n`；Strata 的 `top_k` 限制 0–64，`repeat_last_n` 作为 `penalty_last_n` 且取值 ≥ 1）。更多参数列出引擎支持的高级采样与生成字段：llama.cpp 含 `seed`、`n_predict`、`typical_p`、动态温度、XTC、DRY、Mirostat、`n_keep`/`min_keep`/`n_probs`、`ignore_eos`、`t_max_predict_ms`、`samplers` 等；KVMem、Strata、NInfer 与 KVMem + llama.cpp 提供 `seed`。留空的字段不写入模型配置、请求时也不注入，由引擎使用默认值；请求中显式给出的同名字段优先。`stream`、`stop`、`grammar`、`json_schema`、`logit_bias`、`lora` 等请求级字段不在此列。
- **新增模型**：在控制台填写模型类型（llama.cpp / Strata / KVMem / NInfer / KVMem + llama.cpp）、模型名称（显示名，选填）、模型路径、可选的视觉 mmproj 路径、视觉运行位置（CPU / GPU）和采样参数；llama.cpp、KVMem 与 KVMem + llama.cpp 还可设置上下文长度（留空读取 GGUF，GGUF 缺少该字段时必须手动填写，否则停止新增），llama.cpp 与 KVMem + llama.cpp 还可设置 K/V 量化精度，KVMem + llama.cpp 还可设置 KVMem 预算以及“KVMem 引擎”页的进程级参数（模型路径可以是单个 `.gguf`，也可以是目录——目录会挑选最大的 GGUF 与匹配的 F16 mmproj）。提交后自动完成：路径校验、GGUF 头解析（架构、上下文上限、视觉编码器）或 `.ninfer` 检查、KV 精度选择、参数校验（预算/预留落在上下文内、block ≤ 预留、SSD 额度与目录成对出现、图片 token 下界不大于上界、枚举与数值范围）、写入引擎配置、启动引擎、文本与图片冒烟测试；任一步失败会回滚已写入的配置。Strata、NInfer 与 KVMem + llama.cpp 都采用单模型模式：每新增一个模型就复制模板生成一份 `strata-<name>.json` / `ninfer-<name>.json` / `kvmem-llama-<name>.json`，切换模型时网关会重启对应引擎。Strata 新增模型时把模板的 `--native` 换成所选 GGUF（模板的 `--ple-gguf` 若原本指向同一文件也会一起替换），未显式给出上下文时按新 GGUF 上限收紧 `--max-context`，视觉 mmproj 写入配置的 `vision` 段；引擎槽位的 `template` 必须指向一份可用的 `strata-*.json`。
- **删除模型**：只删除引擎配置，不移动或删除模型文件；如果删除的是当前活动模型，会先停止引擎。
- **默认策略**：上下文默认读取 GGUF 中的上限（可改小，不能超过上限）；GGUF 缺少 `context_length` 时必须手动填写，留空会停止新增并提示原因；llama.cpp 的 K/V 量化默认均为 `q8_0`（k8v8），可分别选择 `f16`、`bf16`、`q4_0`、`q4_1`、`q5_0`、`q5_1`、`iq4_nl`、`f32` 等引擎支持的类型；KVMem 固定 `--kv q8_0`；NInfer 的上下文、KVMem 参数和 KV 量化全部继承模板；KVMem + llama.cpp 默认 `--kvmem`、预算 32768 tokens、block 32、生成保留 1024、KV 精度 `q8_0`（K/V 相同时写 `--kv-dtype`，不同且均为 `q8_0`/`q5_0`/`q4_0` 时分别写 `-ctk`/`-ctv`），预算与生成保留必须落在上下文内；显式给出的预算/预留放不下时直接报错，沿用模板的默认值则按上下文自动收缩，block 不大于生成预留。KVMem 引擎页未填的字段不写入模型配置，由模板或引擎默认决定，保存后自动重启引擎。视觉 mmproj 默认走 CPU，可切换 GPU（NInfer 暂不支持视觉）；其余采样参数以引擎参考模型（llama.cpp 为 LFM3-Omni-2B，KVMem 为 Ornith，NInfer 与 KVMem + llama.cpp 为模板）为默认。
- **代理**：管理页“代理”支持 HTTP 与 SOCKS5，填写类型、主机、端口和可选的用户名/密码；点“测试连接”会先做 TCP 探测，再经代理并行访问 `hf-mirror.com` 与 `modelscope.cn` 验证连通。面板下方直接列出 Hugging Face、HF-Mirror、ModelScope 三个下载源，勾选表示该来源的搜索与下载走代理，未勾选则直连；代理未启用时不可勾选。密码只写入 `config.json`（设置密码后文件权限收紧为 600），读取接口从不回传密码；禁用代理会同时关闭所有下载源的单源代理开关。HTTP 代理使用 `http://`，SOCKS5 使用 `socks5h://`（域名在代理解析）。
- **模型下载**：管理页“模型下载”顶部设置下载目录，保存进 `config.json`，下次打开自动带出。搜索框下方勾选要查询的来源（默认三个全选），只并行调用勾选来源对应的命令行：Hugging Face（`hf` 直连）、HF-Mirror（`HF_ENDPOINT=https://hf-mirror.com` 的 `hf`）与 ModelScope（`modelscope`），聚合为类似 Hugging Face 的卡片，展示仓库、pipeline、库、参数量、更新时间、下载量、点赞、标签等命令行可获得的字段；单个源失败不影响其它源。来源是否走代理在“代理”页分别设置，Hugging Face、HF-Mirror 与 ModelScope 各自独立开关（共用同一代理地址）。点“选择文件”查看仓库文件列表并勾选，下载保存为 `<下载目录>/<仓库ID>/<文件>`；计划阶段标出本地已存在的文件，勾选已存在文件会拒绝开始；支持同时下载多个模型；开始后任务统一显示在控制台右上角的“下载”按钮中，红色角标标注进行中数量，浮层列出所有正在下载的模型，点击任一任务可查看状态、脱敏日志并停止（终止整个进程组），下载页不再重复显示任务进度。进度弹窗每 1.5 秒刷新，按目标目录中已完成的文件与 `.cache/huggingface/download/*.incomplete` 分片统计已下载字节并显示进度条；CLI 日志只在每个文件完成时写一行，单看日志会显得长时间不动。Hugging Face 直连源会清除进程继承的代理变量，避免误走系统代理。
- **聊天 UI**（`/ui`）：内置 llama.cpp 上游 full-ui 构建（当前对应 `9e2a04a439ed`，`webui/llama-ui/ui-build.json` 记录源码入口哈希），保留网关专用的 API Key 说明，使用相对资源路径，可随 `ui_path` 部署到任意子路径。后端引擎统一以 `--no-ui` 启动，网关在 `/props` 中固定返回 `ui: true`，因此加载后的模型不会从模型选择器消失；支持文本、图片、音频和视频输入，API Key 仅保存在浏览器本地。
- **鉴权**：可选服务端 API Key，保护 `/v1` 与聊天 UI 的 `/props`；控制台与 `/models` 不鉴权，适合可信内网使用。

## 内置聊天 UI

`webui/llama-ui` 是 llama.cpp `tools/ui` 的完整（full-ui）静态构建，源码版本记录在 `ui-build.json` 的 `llama_commit` 与 `entry_files` 中。本地修改有两处：`src/lib/constants/settings.constants.ts` 里 API Key 的帮助文字改为网关说明（仅保存在浏览器、服务端密钥在控制台 `/` 修改）；发送消息前等待正在处理（文件读取、音频转码）的附件完成，避免附件落在发送之后、跟到下一条消息。构建输出保持相对资源路径，不写死 `base href`，因此 `ui_path` 可以改成任意子路径。

更新步骤：在 llama.cpp 的 `tools/ui` 执行 `npm ci && npm run build`，把 `dist/` 复制到 `webui/llama-ui`，重新应用上述本地修改，并用新源码的入口哈希更新 `ui-build.json`。网关会在 `/props` 中固定返回 `ui: true`，不要依赖引擎的 `--no-ui` 值。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/router/status` | 网关与引擎状态 |
| GET | `/v1/models` | 模型列表 |
| GET | `/router/models/manage` | 引擎默认值、基本/更多参数 schema 与新增表单选项 |
| POST | `/router/models/add` | 异步新增模型，返回任务 ID |
| GET | `/router/models/job?id=` | 查询新增任务进度 |
| POST | `/router/models/delete` | 删除模型配置 |
| POST | `/router/load` / `/router/unload` | 加载 / 卸载模型 |
| POST | `/router/stop` | 停止引擎 |
| POST | `/router/reload` | 重新扫描模型目录 |
| GET/POST | `/router/engines/paths` | 读取 / 保存引擎启动路径；POST 传 `create` 可创建缺失的固定引擎槽位 |
| POST | `/router/engines/restart` | 按当前 `config.json` 重启指定引擎 |
| POST | `/router/engines/update/check` | 用 git 检查各引擎工作目录的上游版本；`{"use_proxy": bool}` 控制是否走已配置代理 |
| POST | `/router/engines/update/install` | 快进到上游、备份二进制并重新编译；`{"engine": "llama", "use_proxy": bool}`，失败自动恢复备份并回滚源码 |
| GET/POST | `/router/params` | 读取 / 保存模型参数（基本 + 更多）；llama.cpp 可同时保存上下文长度与 K/V 量化精度，有视觉的模型可传 `vision_device: "cpu" \| "gpu"` |
| GET/POST | `/router/auth` | 读取 / 设置服务端 API Key |
| GET/POST | `/router/proxy` | 读取 / 保存下载代理（密码不回传） |
| POST | `/router/proxy/test` | 验证代理连通性 |
| GET | `/router/downloads` | 下载目录、各源、CLI 工具与任务状态 |
| POST | `/router/downloads/settings` | 保存下载目录与各源代理开关 |
| GET | `/router/downloads/search?q=&limit=&sources=` | 按指定来源并行搜索并聚合结果（省略 sources 表示全部） |
| POST | `/router/downloads/plan` | 列出仓库文件与本地已存在情况 |
| POST | `/router/downloads/start` | 启动下载任务，返回任务 ID |
| GET | `/router/downloads/job?id=` | 查询下载进度（含 `downloaded_bytes`）与脱敏日志尾部 |
| POST | `/router/downloads/cancel` | 取消正在运行的下载 |

## 安装与运行

要求 Python 3.10+。`START-router.sh` 首次运行会自动创建 `.venv`，并按 `requirements.txt` 的内容哈希安装依赖（`requests`、`PySocks`、`socksio`；后两者分别让 requests 与 hf CLI 支持 SOCKS5 代理），依赖变化时自动重装：

```bash
./START-router.sh
```

随后脚本检查 `hf` 与 `modelscope` 两个命令行工具，缺失时在 venv 内安装 `huggingface_hub` 和 `modelscope`；两者只服务于“模型下载”页，安装失败会打印 `llm-router: ERROR: ...` 但网关仍会启动，下载页会把对应源标记为不可用。Debian/Ubuntu 若缺少 venv 组件，按报错提示安装 `python3-venv`。需要指定其它解释器时设置 `ROUTER_PYTHON`，此时不创建 venv、不自动安装。

首次启动会在当前目录自动生成最小 `config.json`（不含引擎），控制台显示“未配置引擎”引导，进入管理页的“引擎”页填写启动路径即可，无需重启网关。默认监听 `0.0.0.0:8080`，控制台为 `/`，聊天 UI 为 `/ui`；临时换端口可用 `./START-router.sh --host 127.0.0.1 --port 8081`（不写入配置文件）。`./START-router.sh --check` 只扫描并打印模型列表后退出。

启动脚本会检查 Python 版本、venv 依赖与下载工具。网关通过 `config.json` 指定引擎命令、模型目录、下载代理与下载目录，通过 `models.ini`（llama.cpp）、KVMem 与 Strata 的 `strata-*.json`、NInfer 的 `ninfer-*.json` 与 KVMem + llama.cpp 的 `kvmem-llama-*.json` 描述模型。

Strata 是独立的单模型服务，网关以 `python -m serve.server --engine strata --config <strata-*.json>` 启动它，没有 `--switch-on-request`，切换模型即重启进程。模板里的 `exe`、`args`、`cwd`、`tokenizer`、`vision` 与 `lib_dirs` 由网关读取并随每份模型配置继承；`host`/`port` 由网关命令覆盖。模型配置的 `args` 是生效参数，`user_args` 会在启动时覆盖 `args`，因此保存参数时网关会同时改写 `args`、从 `user_args` 清除被修改的开关并刷新 `args_written`。卸载模型会停止进程（SIGTERM，走优雅退出）。

NInfer 需要一份可用模板（管理页的“模板配置”），新增模型时从模板复制参数，并把模型 ID、`.ninfer` 路径和 `--model-id` 写入新的 `ninfer-<name>.json`。模板里的 `env`（如 `CUDA_VISIBLE_DEVICES`、`LD_LIBRARY_PATH`）会随每份模型配置一起继承。

KVMem + llama.cpp 对应独立的 `llama-kvmem-server`（KVMem 补丁版 llama.cpp），同样需要一份模板（管理页的“模板配置”，缺失时使用内置默认）。新增模型时从模板复制其余参数，并把模型路径、别名、视觉 mmproj、上下文、KV 精度、`--kvmem-budget`、block 与生成保留写入新的 `kvmem-llama-<name>.json`；模板里的 `env` 同样随模型配置继承。“KVMem 引擎”页的进程级参数（检索方法、sink/recent、query 长度、LRFU、Host/SSD 预算、并发、MTP、思考、图片/视频限制等）也写入这份 JSON 的 `args`，留空表示沿用模板或引擎默认，保存时按字段清除对应开关；`-m`、`-a`、`--host`、`--port`、`--mmproj`、`--mmproj-offload`、`-c`、`-ctk`/`-ctv`/`--kv-dtype` 与 `--kvmem` 由网关管理，不能通过该页覆盖。该引擎没有卸载接口，卸载模型或切换模型都会直接停止进程（SIGTERM），下次加载时重新启动。

## 许可

[MIT](LICENSE)
