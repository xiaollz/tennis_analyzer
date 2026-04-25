# Baseline PWA — 公网部署快速指南

> 把 Baseline 装到你手机的主屏幕，用任意网络访问。
> 一条命令搞定。

---

## TL;DR

```bash
./start-public.sh
```

输出会给你一个 `https://xxx.trycloudflare.com` 的 URL。

在手机 Safari 里打开 → 分享 → 添加到主屏幕。

完成。

---

## 它到底做了什么

### 1️⃣ 启动后端 + 前端

`./start-public.sh` 启动 FastAPI（监听 `127.0.0.1:8765`），所有
分析能力 + 前端 UI 都从这里来。

### 2️⃣ 起一个 Cloudflare 隧道

`cloudflared tunnel --url http://127.0.0.1:8765` 在 Cloudflare 边缘节点
和你的电脑之间建一条加密管道。

外面的人通过 `https://xxx.trycloudflare.com` 访问 → Cloudflare 转发到
你电脑的 8765 端口 → FastAPI 处理 → 响应原路返回。

**你不需要 Cloudflare 账号、不需要域名、不需要开放路由器端口**。

### 3️⃣ Baseline 已经是 PWA

`frontend/dist/` 下：

| 文件 | 作用 |
|---|---|
| `manifest.webmanifest` | 告诉浏览器"这是个 App，不是网页" |
| `sw.js` | service worker — 缓存壳层，让 App 启动飞快 |
| `icon-192.png` / `icon-512.png` | 主屏图标 |
| `apple-touch-icon.png` | iOS 主屏图标（圆角自动加） |
| `index.html` | 已加 PWA meta 标签 + safe-area + 全屏样式 |

满足这些，浏览器就允许"添加到主屏幕"，并以**全屏 standalone**
模式打开（没有 Safari 顶部地址栏，看起来跟原生 App 一样）。

---

## 在手机上安装的步骤（iOS）

1. 在 Safari 里打开你的 `https://xxx.trycloudflare.com` URL
2. 等页面完全加载（首次会下载字体 + Babel，约 2-3 秒）
3. 点底部分享按钮 ⏏
4. 找到 **Add to Home Screen**（添加到主屏幕）
5. 名字默认是 "Baseline"，确认
6. 退出 Safari，主屏幕上应该出现一个赭红色方块加米黄圆点的图标

点开图标 = 全屏 App 模式，没有任何浏览器 chrome。

---

## 在手机上安装的步骤（Android）

1. Chrome 打开 URL
2. Chrome 自动弹"安装应用"提示，或菜单 → "安装应用"
3. 主屏幕图标 + Drawer 里都会出现

---

## 关键提醒

### 隧道 URL 每次重启都会变

Cloudflare 免费 Quick Tunnel 是**临时的**——你停掉 `cloudflared`、
重新跑 `./start-public.sh`，URL 会变。

**这意味着**：手机上的"Baseline" App 还在主屏幕，但**点开会 404**
（因为它指向旧 URL）。

**解法 A · 跑长期 tunnel（推荐）**：让脚本一直跑着不停。
开机后第一件事 `./start-public.sh`，后台跑一整天。

**解法 B · 命名 tunnel（一劳永逸）**：
1. 注册免费 Cloudflare 账号
2. 加一个域名（或用 `*.cloudflareaccess.com` 子域）
3. `cloudflared tunnel create baseline`
4. 配置 `~/.cloudflared/config.yml` 指向你的固定子域
5. URL 永久不变

如果你**只是自己用、电脑大部分时间开着**，解法 A 完全够用。

### 上传大视频要等

Cloudflare 免费隧道的 ssl handshake 偶尔慢 2-3 秒。看到"Uploading..."
不动别紧张，30 秒内会动起来。

### 你的电脑必须开机 + 不睡眠

公网访问 = 你的电脑就是服务器。它睡眠 / 关机，App 就连不上。

### 隧道随时可关

```
按 Ctrl-C 停止 ./start-public.sh
```

服务器和隧道都会停。需要再用就再跑一次。

---

## 验证清单

跑完 `./start-public.sh` 后，逐项确认：

```bash
# 1. 拿到 URL（复制 trycloudflare.com 这一行）
echo $URL

# 2. 健康检查
curl $URL/api/health

# 3. PWA 检查
curl -I $URL/manifest.webmanifest    # 必须 200，content-type: application/manifest+json
curl -I $URL/sw.js                   # 必须 200，content-type: text/javascript
curl -I $URL/icon-192.png            # 必须 200，image/png
```

全过 → 手机可以装。

---

## 仅本地用，不要公网

```bash
./start.sh                 # 同一 WiFi 可访问，不开公网
```

这条命令保留着不动。`start-public.sh` 只是它的"加上隧道"版本。

---

## 故障排查

| 现象 | 原因 | 解法 |
|---|---|---|
| `cloudflared not installed` | 没装 | `brew install cloudflared` |
| 公网 URL 拿到了但访问 502 | tunnel 还在握手 | 等 5-10 秒再试 |
| 手机能装但点开白屏 | 旧 cache | 设置→Safari→清除缓存，再装 |
| 启动报"端口被占用" | 8765 被旧进程占 | `pkill -f uvicorn` 然后再跑 |
| 视频上传到 50% 卡住 | 网络抖动 | Cloudflare 自动重连，等 30s |

---

## 下一步可能的演进

1. **永久 tunnel**：绑一个免费 cloudflareaccess.com 子域 → URL 不变
2. **真正云部署**：把后端搬到 Fly.io / Railway → 不依赖你电脑
3. **App Store 上架**：包成 iOS 真原生 App（用 Capacitor 把 PWA 包一层）

但**目前这套对个人使用已经完美**——投入 0 元，5 分钟搞定。
