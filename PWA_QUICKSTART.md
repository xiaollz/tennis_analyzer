# Baseline PWA — 公网部署快速指南

> 把 Baseline 装到你手机的主屏幕，用任意网络访问。
> 一条命令搞定，0 元，URL 永远不变。

---

## TL;DR

```bash
# 一次性配置（5 分钟）
cp .tunnel.config.example .tunnel.config
# 编辑 .tunnel.config，填入两个值（见下方）
# 然后：

./start-public.sh
```

输出会给你一个 `https://你-的-名字.ngrok-free.app` 的 URL。

在手机 Safari 里打开 → 分享 → 添加到主屏幕。**装一次永远能用**。

---

## 一次性配置：3 步搞定

### 1️⃣ 注册 ngrok（免费，不要信用卡）

https://dashboard.ngrok.com/signup

邮箱 + 密码，1 分钟搞定。

### 2️⃣ 拿两个值

#### a. **Authtoken**

登录后到：https://dashboard.ngrok.com/get-started/your-authtoken

复制那一长串（看起来像 `2abc...XYZ`，~50 字符）。

#### b. **静态域名**（免费 1 个）

到：https://dashboard.ngrok.com/cloud-edge/domains

点 **+ New Domain** → 起个名字（比如 `qsy-tennis`）→ 创建。

它会变成：`qsy-tennis.ngrok-free.app`

### 3️⃣ 写到 `.tunnel.config`

```bash
cd /Users/qsy/Desktop/tennis
cp .tunnel.config.example .tunnel.config
```

用编辑器打开 `.tunnel.config`，填两行：

```
NGROK_AUTHTOKEN=2abc...XYZ
NGROK_DOMAIN=qsy-tennis.ngrok-free.app
```

保存。这个文件是 gitignore 的，不会进仓库。

---

## 启动

```bash
./start-public.sh
```

打印类似：

```
  Booting Baseline server on http://127.0.0.1:8765 ...
  ✓ server up

  Starting ngrok tunnel → https://qsy-tennis.ngrok-free.app ...

  ┌──────────────────────────────────────────────────────────────┐
  │  Public URL (永久):                                            │
  │  https://qsy-tennis.ngrok-free.app                             │
  ├──────────────────────────────────────────────────────────────┤
  │  On your phone (first time):                                  │
  │    1. Open the URL in Safari                                  │
  │    2. Tap Share → Add to Home Screen                          │
  │                                                                │
  │  After that the URL never changes —                           │
  │  just tap the home-screen icon.                               │
  └──────────────────────────────────────────────────────────────┘
```

**Ctrl-C** 停止。

---

## 在手机上安装（iOS）

1. Safari 打开 `https://qsy-tennis.ngrok-free.app`
2. 等首次加载完（2-3 秒下载字体 + Babel）
3. 底部分享按钮 ⏏ → **Add to Home Screen**
4. 名字默认 "Baseline"，确认
5. 退出 Safari → 主屏幕出现赭红方块 + 米黄圆点的图标
6. 以后随时打开 ngrok / 球场 / 移动数据 / 任何网络 → 点图标 → 全屏 App

**关键**：以后**只要电脑跑着 `./start-public.sh`**，手机就能用。
URL 永远是 `qsy-tennis.ngrok-free.app`，**不用重新装**。

---

## Android Chrome

1. Chrome 打开 URL
2. 菜单 → "安装应用" / "添加到主屏幕"
3. 主屏幕图标，全屏体验

---

## 关键提醒

### ✅ URL 永久不变

这是 ngrok vs Cloudflare Quick Tunnel 的最大区别。手机一次安装，永远能用。

### ⚠️ 你的电脑必须开机 + 不睡眠

公网访问 = 你的电脑就是服务器。它睡眠 / 关机，App 就连不上。

打开 **System Settings → Battery → 高级 → 防止你的 Mac 在屏幕关闭时睡眠**，
方便长时间挂着。

### ⚠️ 免费档限制

- **1 GB / 月** 入站带宽 → 大约能传 200 个 5MB 视频
- **1 个静态域名** → 你只用 1 个，没问题
- **同时 1 个 endpoint** → 你只起 1 个，没问题

平时自己用绰绰有余。如果某月真的传爆了 1GB，也只是临时的——下个月自动重置。

### ⚠️ 别把 `.tunnel.config` 提交到 git

里面有你的 authtoken。已经默认 gitignore 了，但**别手动 force-add**。

---

## 切换 / 退出

### 切回 Cloudflare Quick Tunnel（临时 URL）

```bash
mv .tunnel.config .tunnel.config.bak
./start-public.sh
```

它会回退到 cloudflared 模式（你之前装好的）。

### 重置 ngrok 配置

直接编辑 `.tunnel.config` 改两个值即可。

### 仅本地用，不公网

```bash
./start.sh
```

只在 0.0.0.0:8765 监听，同 WiFi 可访问，不开公网。

---

## 验证清单

跑完 `./start-public.sh` 后，在另一个终端：

```bash
# 1. 健康检查
curl https://qsy-tennis.ngrok-free.app/api/health
# → {"status":"ok",...}

# 2. PWA 资源
curl -I https://qsy-tennis.ngrok-free.app/manifest.webmanifest
# → 200, content-type: application/manifest+json

# 3. 主页
open https://qsy-tennis.ngrok-free.app/
# → 浏览器打开 Baseline App
```

---

## 故障排查

| 现象 | 原因 | 解法 |
|---|---|---|
| `ngrok not installed but .tunnel.config has ngrok values` | brew 没装 | 直接装：`curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-arm64.zip -o /tmp/n.zip && unzip /tmp/n.zip -d /opt/homebrew/bin/ && chmod +x /opt/homebrew/bin/ngrok` |
| `ERR_NGROK_3200` 域名被占 | 你 dashboard 上没真正预留这个域名 | 回 dashboard → Domains 重新创建 |
| 401 unauthorized | authtoken 错 / 复制时多了空格 | 重新从 dashboard 复制 |
| 手机能装但点开白屏 | 旧 cache | 设置 → Safari → 清除缓存，再装 |
| 上传到 50% 卡住 | 网络抖动 | ngrok 会重连，等 30s |

---

## 下次要使用，三步

```bash
# 你的 Mac 上
cd /Users/qsy/Desktop/tennis
./start-public.sh

# 等看到 Public URL 之后，手机上点主屏 Baseline 图标
# 完成
```

URL 不变，所以这是稳定的日常用法。

---

## 补充：Cloudflare Quick Tunnel（无配置兜底）

如果你**不想配 ngrok**，直接 `./start-public.sh`，它会自动降级到 Cloudflare quick tunnel：

- ✅ 不用注册任何账号
- ❌ URL 每次重启都变 → 手机上每次要重新装
- 仅用于一次性演示 / 临时给朋友看

平时建议**还是配上 ngrok**——稳定永久 URL 才是主屏 App 的灵魂。
