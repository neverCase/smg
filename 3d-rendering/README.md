# XHIVE-PRO · Orbital Console

纯前端 3D 推理星图。产品品牌为 **XHIVE-PRO**，中央网关节点为 **ai-gateway**。使用本地 Three.js ES modules，不依赖 CDN、构建步骤或框架。

## 启动

Windows 推荐直接双击 `start.cmd`，或在 PowerShell 中执行 `./start.cmd`。脚本会自动查找 Node.js；未单独安装时会尝试使用当前用户的 Codex 内置 Node.js，因此不要求普通终端已经配置好 `node` 的 PATH。保持启动窗口打开，按 `Ctrl+C` 停止服务。

如果提示找不到 Node.js，请从 https://nodejs.org/ 安装 Node.js 22 或更新版本，然后重新打开终端。Codex 内置运行环境的路径可能随版本变化，独立部署时建议使用正式安装的 Node.js。

在本目录执行（Node.js 22 或更新版本）：

```powershell
node server.mjs
```

浏览器打开 http://localhost:5173 。也可以用任意静态 HTTP 服务器托管本目录。不要使用 `file://` 直接打开 HTML，因为浏览器会限制 ES module 加载。

自定义端口：

```powershell
$env:PORT = '5174'
node server.mjs
```

`server.mjs` 只提供静态文件，不代理或修改后端 API。页面默认直接请求 `http://10.8.0.27:31188/workers`，每 3 秒刷新。设置中可以修改地址与刷新间隔，并手动切换到多模型演示模式。

## 交互

- 拖拽空白区域旋转相机；滚轮缩放；触摸屏支持单指旋转、双指缩放。
- 左侧模型组可以展开 / 收起；点击模型定位星系，点击 worker 定位星球并打开右侧详情。
- 每个模型星系绕 ai-gateway 公转，worker 在自己的星系内旋转；中心到所有 worker 均有光束。
- 左侧模型列表、右侧观测站、底部趋势图可以分别折叠；边缘的小标签用于重新展开。折叠状态保存在本机浏览器。
- 顶部“沉浸模式”或 `F` 隐藏周边面板；`Esc` 退出。点击 worker 会退出沉浸并打开详情。
- `/` 打开模型搜索；底部工具条支持重置相机、缩放、暂停公转、显示 / 隐藏节点标签。双击场景回到全景。
- 暂停公转只影响动效，不会暂停数据采集。浏览器后台标签暂停动画与轮询，返回时立即刷新。

## 数据和视觉口径

| 表现 | 来源 |
| --- | --- |
| 模型星系 | `workers[].models[].id`，缺失时回退到 `model_id` |
| 星球 | worker ID，缺失时使用 URL；支持 worker 同时服务多个模型 |
| 网关在途数 | `load` |
| 引擎执行 / 排队 | 所有 `engine_load.loads[]` rank 的 `num_running_reqs` / `num_waiting_reqs` 之和 |
| KV 光环 | rank 的 `token_usage` 平均值 |
| 星球和光束颜色 | **同一星系**的 `load` 最小值到最大值，冰蓝 → 琥珀 → 珊瑚红 |
| 灰色星球 | 节点不可用、stalled，或者在途数据未知 |
| 底部趋势图 | 当前页面会话实际采集的最近 120 个在途数样本 |

相对颜色不等于过载判定，也不比较不同模型间的负载。相同负载或单个 worker 统一冰蓝；不可用节点不参与热度范围。流动光点是状态示意，不是逐请求追踪。KV 占用不是 GPU 利用率。

未提供的指标显示“—”或“未提供”，不填造累计成功 / 失败数。引擎快照与网关计数可能不同步；页面刷新频率也不改变 ai-gateway 的引擎采样频率。跨 rank 数据不完整时显示部分数据标记。

连接失败时显示错误并保留同一来源最后一次成功的数据；不会自动切换到演示数据。更换接口或切换真实 / 演示来源会清空旧趋势和选中状态。演示模式始终显示明显的 DEMO 标识。

## 浏览器与部署

需要支持 WebGL 2 的现代浏览器及硬件加速。不能创建 WebGL 时，列表和数据界面仍可使用，并显示说明。

API 必须允许页面来源的 CORS。当前默认地址为 HTTP，建议本地开发也使用 HTTP；从 HTTPS 站点请求 HTTP API 可能被浏览器阻止。部分浏览器会询问本地网络访问权限。所有设置仅保存在浏览器 localStorage，不发送给第三方。

## 校验

```powershell
node --test --test-isolation=none tests/data.test.mjs
```

覆盖网关与引擎口径区分、缺失值与零值、多模型去重、组内相对颜色、多 rank 聚合和接口校验。

## 文件

- `index.html` / `styles.css`：布局、面板与响应式样式。
- `src/main.js`：轮询、连接状态、面板交互与趋势。
- `src/scene.js`：Three.js 场景、相机、星系、公转、光束、节点选择。
- `src/data.js`：API 数据转换、分组、相对负载与显式演示数据。
- `vendor/`：Three.js **0.180.0** 及 MIT 许可证。源自 npm 官方 `three` 包，下载时已校验包的 SHA-512 integrity：`sha512-o+qycAMZrh+TsE01GqWUxUIKR1AL0S8pq7zDkYOQw8GqfX8b8VoCKYUoHbhiX5j+7hr8XsuHDVU6+gkQJQKg9w==`。
