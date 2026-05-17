# `SESSION_API.md` 接口端到端验证报告

## Run metadata

- **开始时间**:`2026-05-15T22:29:56Z` (UTC)
- **测试人**:Claude(自动化探针 + 增量记录)
- **VM**

  | | host | os_type | endpoint | check_status |
  |---|---|---|---|---|
  | Linux | `34.94.179.145` | `linux` | `http://34.94.179.145:5000` | `True` |
  | Windows | `34.57.85.163` | `windows` | `http://34.57.85.163:5000` | `True` |

- **客户端版本**

  | | path | sha |
  |---|---|---|
  | cua-bench / computer SDK | `agenthle/submodules/cua/libs/cua-bench` (editable) | `2a10d3260145` |
  | agents-last-exam | `agents-last-exam` | `65059659e583` |

- **Screenshot 落盘**:`/tmp/probe/{linux,windows}/<step>.png`
- **图例**:✅ PASS / ❌ FAIL / ⏭ SKIP / ⚠ EXPECTED-FAIL(接口存在但因环境缺依赖而预期报错)

---

## §1 Lifecycle + 属性

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `check_status()` | ✅ `True` (2.71s 冷,0.54s 热) | ✅ `True` (2.25s 冷,2.22s 热) | 内部 `interface.get_screen_size()`,冷首调要建底层 SDK 连接 |
| `wait_until_ready(timeout=10)` | ✅ `True` (0.05s) | ✅ `True` (0.23s) | 已就绪时立即返回 |
| `os_type` 属性 | ✅ `'linux'` | ✅ `'windows'` | |
| `vnc_url` 属性 | ⚠ `'http://localhost:8006/?autoconnect=true'` | ⚠ `'http://localhost:8006/?autoconnect=true'` | **不指向真实 VM**——client-only 模式且 ctor 未传 `vnc_url=` 时,恒返回 localhost 占位。要看真 VNC 必须 ctor 显式传 |
| `interface` 属性 | ✅ `OtelInterfaceWrapper`(可链式调底层) | ✅ `OtelInterfaceWrapper` | OTEL 启用时被包装一层,**行为透传** |
| `session.interface is session.computer.interface` | ✅ `True` | ✅ `True` | 文档断言成立 |
| `close()` | ✅ 静默成功 | ✅ 静默成功 | |

**小结**:接口全部按文档行为。`vnc_url` 在没显式传时是误导性占位,文档需补充。

---

## §2 Shell `run_command`

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `echo` 成功路径 | ✅ `rc=0 stdout='PROBE_OK\n'` | ✅ `rc=0 stdout='PROBE_OK\r\n'` | |
| 平台命令 | ✅ `uname -a` 拉到完整 kernel banner | ✅ `ver` 返回 `Microsoft Windows [Version 10.0.19045.6466]` | |
| `stderr` 捕获 | ✅ `ls /nonexistent-zzz` stderr 含 `No such file` | ✅ `dir Z:\nope` stderr 含 `cannot find the path` | |
| 多次调用 cwd 不继承 | ✅ `cd /tmp` 后下一次 `pwd` 回 `/opt/cua-server` | ✅ `cd C:\Windows` 后下一次 `cd` 回 `C:\Windows\system32` | 每次 RPC 起独立 shell,文档说法正确 |
| `cd && cmd` workaround | ✅ `cd /tmp && pwd` → `/tmp` | ✅ `cd /d C:\Windows && cd` → `C:\Windows` | |
| **多行命令** | ⚠ 实际**能**执行(`for i in 1 2 3; do echo loop_$i; done` 多行写法返回 3 行) | ❌ Windows 多行只跑首行(`A`,`echo B` 被丢) | **文档"不要塞 multi-line"在 Linux 上过严** |
| `check=False` 失败命令 | ❌ **`false` 返回 `rc=0 success=True`** | ❌ **`cmd /c exit 1` 返回 `rc=0 success=True`** | 🔴 **bug,见下** |
| `check=True` 失败命令 | ❌ **不抛 RuntimeError** | ❌ **不抛 RuntimeError** | 🔴 同上 bug 的连锁后果 |
| 验证旁路:`interface.run_command` 直调 | ✅ `false` → `returncode=1` 正确 | ✅ `cmd /c exit 1` → `returncode=1` 正确 | 底层 SDK 行为正常 |

### 🔴 BUG: `run_command` 永远返回 `rc=0` / `success=True`

**根因**:`cua-bench` 的 `RemoteDesktopSession.shell_command()`(`cua_bench/computers/remote.py:769`)读属性 `result.return_code`,但 cua-computer SDK 的 `CommandResult`(`interface/models.py:7`)真正属性名是 `returncode`(无下划线)。

```python
# remote.py:769
return_code = result.return_code if hasattr(result, "return_code") else 0  # 永远走 else 分支
```

**实测**:`hasattr(raw, "return_code")=False; hasattr(raw, "returncode")=True`。

**影响**:任何 `session.run_command()` 的返回 dict 里 `return_code` / `success` 字段 **完全不可信**。`check=True` 也因此永不抛错。

**临时绕过**(给 agent 实现者):

```python
raw = await session.interface.run_command(cmd)   # 跳过 session 包装,拿原始 CommandResult
if raw.returncode != 0:
    ...
```

**根治**:`remote.py:769` 改成 `result.returncode`(或加 fallback)。

**小结**:成功路径、stdout/stderr 捕获、cwd 隔离都按文档行为。但失败处理彻底失效;agent 当前 **不能用 `run_command` 检测命令是否失败**,必须走 `interface.run_command` 拿原始 `returncode`。多行执行 Linux 实际可用,文档过于保守。

---

## §3 Filesystem

测试基目录:Linux `/tmp/cua-probe`,Windows `C:\Users\Public\cua-probe`。

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `interface.create_dir(BASE)` | ✅ `directory_exists` 翻转 False→True,parent `list_dir` 出现条目 | ✅ 同左 | |
| `interface.create_dir(nested)` 递归 | ✅ `BASE/nest/deep` 直接建,中间 `nest` 自动建 | ✅ 同左 | 文档没提**create_dir 是递归的** |
| `write_file` / `read_file` 文本往返 | ✅ `"hello\nworld 你好\n"` 含 UTF-8 完整往返 | ✅ 同左 | |
| `write_bytes` / `read_bytes` 二进制往返 | ✅ `bytes(range(8))` 完整 | ✅ 同左 | |
| `interface.read_bytes(offset=2, length=3)` | ✅ 拿到 `b'\x02\x03\x04'` | ✅ 同左 | |
| `interface.get_file_size` | ✅ text=19, bin=8 完全匹配 | ✅ 同左 | |
| `list_dir(BASE)` | ✅ `['t.txt', 'b.bin']` | ✅ `['b.bin', 't.txt']`(顺序异于 Linux) | 仅返回 basename,与文档一致 |
| `file_exists` | ✅ 创建/删除翻转 | ✅ 同左 | |
| `directory_exists` | ✅ True / False 正常 | ✅ 同左 | |
| `interface.delete_file` | ✅ `file_exists` 翻转 True→False | ✅ 同左 | |
| `interface.delete_dir` 空目录 | ✅ | ✅ | |
| **`interface.delete_dir` 非空目录** | ❌ `RuntimeError: [Errno 39] Directory not empty` | ❌ `RuntimeError: [WinError 145] The directory is not empty` | 🟡 **非递归**;文档需补充 |
| 不存在路径上的 `directory_exists` / `file_exists` | ✅ `False`(不抛错) | ✅ 同左 | |

**小结**:所有读写路径都准确。两处文档需补:**`create_dir` 是递归的**(实际行为)、**`delete_dir` 不递归**(非空目录会抛 `RuntimeError`,要先清空或走 shell `rm -rf` / `rmdir /S /Q`)。回读、size、offset/length 都精确。

---

## §4 Screenshot + Actions

### 4.1 Screenshot / Snapshot / A11y

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `screenshot()` | ✅ 1.57 MB PNG,1920×1080,magic OK | ✅ 355 KB PNG,1024×768,magic OK | 落盘 `/tmp/probe/{linux,windows}/01_initial.png` |
| `get_snapshot()` | ✅ `Snapshot(windows=[])`(无活动窗口) | ✅ `Snapshot(windows=[1])`(命中 taskbar `pid=7180`) | Linux 桌面空,Windows taskbar 命中,均符合"只返回当前焦点窗口"约定 |
| `get_accessibility_tree()` | ✅ `dict` 含 `{'success', 'tree'}` 二键 | ✅ 同左 | 🟡 文档说"无支持返回 `{}`"——实际**总返回非空 dict**,里面套 `success` + `tree` |

### 4.2 Actions(`execute_action`)

屏幕尺寸:Linux `1920×1080`,Windows `1024×768`。点击坐标都用 `(W//2, H//2)`。

| Action | Linux | Windows | 验证手段 |
|---|---|---|---|
| `MoveToAction(100, 200)` | ✅ `cursor_position == {x:100,y:200}` | ✅ 同左 | `interface.get_cursor_position()` |
| `ClickAction(cx, cy)` | ✅ 无异常 | ✅ 无异常 | 只看不抛错,后随 `Escape` 兜底 |
| `RightClickAction(cx, cy)` | ✅ | ✅ | |
| `DoubleClickAction(cx, cy)` | ✅ | ✅ | |
| `MiddleClickAction(cx, cy)` | ✅ | ✅ | 没退化为 move_cursor;cua 现在原生支持 |
| `DragAction(300,300 → 500,400)` | ✅ `cursor_position == {x:500,y:400}` | ✅ 同左 | |
| `ScrollAction(direction="down", amount=300)` | ✅ | ✅ | 只看不抛错 |
| `ScrollAction(direction="up", amount=300)` | ✅ | ✅ | |
| `TypeAction("probe-no-target")` | ✅(无焦点 editor,内容丢弃) | ✅ 同左 | 真键入验证见 §7.1 |
| `KeyAction("Escape")` | ✅ | ✅ | |
| `HotkeyAction(["ctrl","a"])` | ✅ | ✅ | 真组合键验证见 §7.1 |
| `WaitAction(seconds=0.2)` | ✅ `dt=0.202s` | ✅ `dt=0.202s` | sleep 精度 OK |
| `DoneAction()` | ✅ 立即 return,无异常 | ✅ 同左 | |

### 4.3 便捷快捷方法

| 方法 | Linux | Windows | 备注 |
|---|---|---|---|
| `session.move_to(150, 250)` | ✅ cursor 匹配 | ✅ | |
| `session.click(cx, cy)` | ✅ | ✅ | |
| `session.right_click(cx, cy)` | ✅ | ✅ | |
| `session.double_click(cx, cy)` | ✅ | ✅ | |
| `session.drag(100,100, 200,200)` | ✅ cursor=`{x:200,y:200}` | ✅ | |
| **`session.scroll(direction="down", amount=300)`** | ❌ `TypeError: ScrollAction.__init__() got an unexpected keyword argument 'x'` | ❌ 同左 | 🔴 **bug,见下** |
| `session.type("ignored")` | ✅ | ✅ | |
| `session.key("Escape")` | ✅ | ✅ | |
| `session.hotkey(["ctrl","a"])` | ✅ | ✅ | |

### 🔴 BUG: `session.scroll()` 永远抛 TypeError

**根因**:`cua-bench/computers/remote.py:947` 的 `scroll` 实现给 `ScrollAction` 传了 `x=`/`y=` 字段,而 `cua_bench.types.ScrollAction` 数据类**只有** `direction` 和 `amount`:

```python
# remote.py:947 (broken)
ScrollAction(x=self._width // 2, y=self._height // 2,
             direction=direction, amount=amount)
# → TypeError: unexpected keyword argument 'x'
```

```python
# types.py:104 (实际 schema)
@dataclass
class ScrollAction:
    direction: Literal["up", "down"] = "up"
    amount: int = 100
```

**绕过**:直接 `await session.execute_action(ScrollAction(direction="down", amount=300))`(已实测通过)。

**根治**:`remote.py:947` 删掉 `x=` / `y=` 参数。

**小结**:Action / 截屏 / 快照都通,验证可达。**`session.scroll` 一调就抛**——文档 §4.3 这条要换成 `execute_action(ScrollAction(...))`。get_snapshot 在 Linux 空桌面正确返回空 list,Windows 命中 taskbar(pywinctl 找到第一个可见进程窗口)。a11y_tree 实际总返回 `{success, tree}` 而非空 dict。

---

## §7.1 `c.interface` 上 session 未封装的方法

### 鼠标 / 键盘细粒度

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `mouse_down(x,y,"left")` + `mouse_up(...)` | ✅ cursor 在 (300,400) | ✅ 同左 | |
| `mouse_down/up("middle")` | ✅ | ✅ | |
| `mouse_down/up("right")` | ✅ | ✅ | |
| `key_down("shift")` + `key_up("shift")` | ✅ 无异常 | ✅ 同左 | |
| `get_cursor_position()` | ✅ `move_cursor(123,234)` 后 `{x:123,y:234}` 精确 | ✅ 同左 | |

### 任意坐标滚轮

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `scroll(500, 500, 2)` | ✅ | ✅ | |
| `scroll_down(3)` | ✅ | ✅ | |
| `scroll_up(3)` | ✅ | ✅ | |

### 剪贴板

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `set_clipboard(M)` + `copy_to_clipboard()` 往返 | ✅ `PROBE-CB-MARKER-12345` 完整往返 | ✅ 同左 | |

### 应用启动 / URL 打开

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `launch("gedit"/"notepad")` | ✅ 返回 `pid=int`(`gedit→4090`) | ✅ 返回 `pid=int`(`notepad→1428`) | 返回 OS PID,不是 window id |
| `open(/tmp/...txt)` / `open(C:\Users\...\.txt)` | ✅ 用 xdg-open 拉起 gedit;`get_snapshot` 显示 `cua-probe-open.txt (/tmp) - gedit` | ✅ 用 start 拉起 notepad;`get_snapshot` 显示 `cua-probe-open - Notepad` | |

### 窗口管理

**测试对象**:gedit (Linux) / notepad (Windows),先 `launch` 拿 wid。

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `get_application_windows("gedit"/"notepad")` | ✅ `[58720504]` | ✅ `[66480]` | |
| `get_current_window_id()` | ✅ `58720504` | ✅ `66480` | |
| `get_window_name(wid)` / `get_window_title(wid)` | ✅ `'Untitled Document 1 - gedit'`,两接口结果相等 | ✅ `'Untitled - Notepad'`,相等 | alias 一致 |
| `get_window_size(wid)` | ✅ `(952, 799)` | ✅ `(1038, 690)` | |
| `get_window_position(wid)` | ✅ `(68, 105)` | ✅ `(-7, 45)` | Windows 边框 offset 是负数 |
| `set_window_size(wid, 640, 480)` | ❌ `RuntimeError: Failed to set window size` | ✅ 读回 `(640, 480)` 精确 | 🟡 Linux 服务端 wmctrl/xdotool 不支持 GNOME 的 client-side decorations |
| `set_window_position(wid, 200, 150)` | ❌ `RuntimeError: Failed to set window position` | ✅ 读回 `(200, 150)` 精确 | 🟡 同上 |
| `activate_window(wid)` | ✅ `get_current_window_id == wid` | ⚠ 不抛错但**焦点不一定真切过去**(Windows 浏览器抢回焦点了);后续 `Type` 测试用 `left_click` 强抢焦点 | |
| `maximize_window(wid)` | ❌ `RuntimeError: Failed to maximize window` | ✅ 尺寸变为 `(1040, 744)`(接近全屏) | 🟡 Linux 同上 |
| `minimize_window(wid)` | ❌ `RuntimeError: Failed to minimize window` | ✅ 执行;接的 `get_active_window_bounds` 报 `Failed to get active window bounds`(可能因为没有活动窗口了) | 🟡 Linux 同上 |
| `close_window(wid)` | ❌ `RuntimeError: Failed to close window` | ❌ `RuntimeError: Failed to close window` | 🟡 两边都失败;绕过用 shell kill |
| `get_active_window_bounds()` | ⚠ 日志先报 "REST API failed, trying WebSocket fallback",但 fallback 通常成功 | 同左 | 实际是 fallback 路径在工作 |
| Type+Hotkey 真验证(`type_text("Probe-Type-W-7733")` + `ctrl+a` + `ctrl+c` → `copy_to_clipboard`) | ✅ 剪贴板含 marker(先 click-to-focus 编辑器) | ✅ 剪贴板含 marker(同) | **必须先 `left_click(cx,cy)` 抢焦点**,光靠 `activate_window` 不够稳 |

### 桌面环境 / 坐标变换 / playwright

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `get_desktop_environment()` | ⚠ `'unknown'`(VM 上没 export `XDG_SESSION_DESKTOP`/`DESKTOP_SESSION`,即使桌面是 GNOME 也返 unknown) | ✅ `'windows'` | Linux 探测靠 env vars 不太靠谱 |
| **`to_screen_coordinates(100, 200)`** | ❌ `TypeError: Logger.debug() takes 2 positional arguments but 12 were given` | ❌ 同左 | 🔴 **bug,见下** |
| **`to_screenshot_coordinates(100, 200)`** | ❌ 同上 | ❌ 同上 | 🔴 同 |
| `playwright_exec("page.goto", {...})` | ⚠ 不抛错,返回 `{'success': False, 'error': '...Browser initialization failed: Browser...'}` | ⚠ 不抛错,返回 `{'success': False, 'error': '...400: Unknown command: page.goto'}` | 接口路由通,但 VM 内无 playwright;Windows 干脆没注册命令名 |

### 🔴 BUG: `to_screen_coordinates` / `to_screenshot_coordinates` 总抛 TypeError

**根因**:`computer/interface/generic.py:682-688` 调 `self.logger.debug(...)`,但实测 `self.logger` 上挂的 `debug` 签名是 `(self, message)` 单参,call 时不知为何被 traceback 报为 12 个位置参数。具体调度链需要再追(嫌疑:tracing 或 OTEL 在某条路径上替换了 logger)。

**实际现象**:两个坐标变换接口 100% 抛错,完全不可用。

**绕过**(agent 实现者):screen ↔ screenshot 缩放比一般 = 1(`screenshot_width / get_screen_size["width"]`);若实在需要,自己拉 screenshot + `get_screen_size` 手算。

**小结**:细粒度鼠标键盘、滚轮变体、剪贴板都稳。`launch` / `open` / 窗口**读**接口都通。Linux 上**所有窗口写操作**(set_size/set_position/maximize/minimize)在这个 GNOME VM 上炸,Windows 这边除 `close_window` 外都行。Type+Hotkey 真键入两边都能验证,但 **必须先 `left_click` 抢焦点**——单调 `activate_window` 焦点未必真切过去。坐标变换 100% 抛错,`get_desktop_environment` 在 Linux 上不可靠。

---

## §7.2 `c` 自身(non-interface)

| 接口 | Linux | Windows | 备注 |
|---|---|---|---|
| `c.pty.create(cmd, on_data, ...)` + `send_stdin` + `wait` | ✅ exit_code=0,`PTY_OK` 在 stdout(`bash`) | ✅ exit_code=0,`PTY_OK` 在 stdout(`cmd.exe` 或 `powershell.exe`) | Windows **必须 stdin 之间 ≥2s 延迟**,1s 不够;返回带 ANSI 转义,需自己清洗 |
| `c.python_exec(fn, *args)` | ✅ `add_one(2,3)→5` | ✅ 同左 | **`fn` 必须从真 .py 文件读源码**,heredoc/REPL 定义会抛 `Cannot retrieve source code` |
| `c.python_command(use_system_python=True)` 装饰器 | ✅ `get_uname()` 返回 `{system:Linux,node:user-virtual-machine}` | ✅ 返回 `{system:Windows,node:DESKTOP-66IFUB8}` | 同上,必须文件源码 |
| `c.python_exec_background(fn)` | ✅ pid=4619;3 秒后 `ps -p` 空(进程结束) | ✅ pid=1356;3 秒后 `tasklist` 无匹配 | |
| `c.venv_install("cua-test-venv", ["cowsay"])` | ✅ `returncode=0`,`directory_exists(~/.venvs/cua-test-venv)=True`;uv 用 CPython 3.14.3 | ✅ `returncode=0`,`directory_exists(C:\Users\User\.venvs\cua-test-venv)=True` | |
| `c.venv_cmd("cua-test-venv", "python -c 'import cowsay...'")` | ✅ stdout 含 `hello` 牛形图案 | ✅ 同左 | |
| `c.venv_exec("cua-test-venv", py_major_version)` | ✅ 返回 `3` | ✅ 同左 | 与 venv_install 装的 CPython 大版本一致 |
| `c.venv_exec_background("cua-test-venv", bg_task, requirements=[])` | ✅ pid 返回 | ✅ pid 返回 | |
| **`c.pip_install(["cowsay"])`** | ❌ `Permission denied` 写 `/usr/local/lib/python3.10/dist-packages/`(cua-server 非 root) | ✅ rc=0;`python -c "import cowsay"` 成功(版本 6.1) | 🟡 **Linux 上不可用**,要 root;实际部署里 cua-server 都是非 root 跑 |
| `c.get_screenshot_size(png_bytes)` | ✅ `{'width':1920,'height':1080}`(同步,无 RPC) | ✅ `{'width':1024,'height':768}` | |
| `c.tracing` 属性 | ✅ `ComputerTracing` 实例,`is_tracing=False` | ✅ 同左 | 没开启录制 |

**小结**:绝大多数接口通过。`python_exec`/`venv_exec`/`python_command` 系列 **强制要求 fn 定义在 .py 文件**(`inspect.getsource` 依赖),REPL / heredoc / `exec()` 内定义的函数会抛 `Cannot retrieve source code`——文档需补这一条。Windows `c.pty` 需要 ≥2s stdin 间隔。`pip_install` 在 Linux 上 **完全不可用**(默认装到系统 Python 需要 root,而 cua-server 实际以非 root 跑)。

---

## 文档需修订 / 新发现

### 🔴 真 bug(cua-bench 代码)

1. **`session.run_command()` 返回 `return_code` 永远是 0**
   - `cua-bench/computers/remote.py:769`:`result.return_code if hasattr(result,"return_code") else 0`
   - SDK `CommandResult` 实际属性叫 `returncode`(无下划线)
   - 后果:`success` 字段不可信,`check=True` 永不抛错
   - 修复:`remote.py:769` 改为 `result.returncode`(并加 `hasattr` 兜底)

2. **`session.scroll(direction, amount)` 一调就抛 TypeError**
   - `cua-bench/computers/remote.py:947` 给 `ScrollAction` 传了 `x`/`y` 字段,但 `ScrollAction` 数据类没这俩字段
   - 修复:删 `x`/`y` 关键字参数

3. **`interface.to_screen_coordinates` / `to_screenshot_coordinates` 一调就抛 TypeError**
   - `Logger.debug() takes 2 positional arguments but 12 were given` —— 在 `computer/interface/generic.py:682` 处的 `self.logger.debug(...)` 调用上
   - 现象:即使表面上是单字符串,运行时仍报 12 个位置参数,具体调度链未定(嫌疑:tracing wrapper 替换了 logger)
   - 影响:坐标变换 100% 不可用

### 🟡 文档应补充 / 修订(`agents-last-exam/docs/SESSION_API.md`)

- **§4.3 `session.scroll(direction, amount)`**:加红字 "🔴 当前 broken,改用 `execute_action(ScrollAction(direction=..., amount=...))`"
- **§2 `run_command` 返回 dict 的 `return_code`/`success`**:加注 "🔴 当前永远是 0/True;要查真返回码请走 `session.interface.run_command(cmd).returncode`"
- **§3 `interface.create_dir`**:实测**递归**(中间目录自动建),文档应明示
- **§3 `interface.delete_dir`**:实测**非递归**(`Errno 39` / `WinError 145`),需补充 "非空目录会抛 RuntimeError,先清空或走 shell `rm -rf` / `rmdir /S /Q`"
- **§4.1 `get_accessibility_tree`**:文档说"无支持返回 `{}`",实测**总返回非空 dict** `{success, tree}`,要从 `tree` 字段取真实树
- **§4.1 `get_snapshot`**:文档说"只返回当前焦点窗口",实测 Linux 空桌面时返回 `windows=[]`(正确),Windows 命中 taskbar(也合理)
- **§7.1 `interface.set_window_size/position/maximize_window/minimize_window/close_window`**:文档没说**Linux GNOME VM 上全部不可用**(server-side wmctrl/xdotool 不支持 GNOME 的 client-side decorations);需要标注 "Linux 上稳定性差,Windows OK,close_window 两边都失败"
- **§7.1 `interface.activate_window`**:**不能保证焦点真切过去**(Windows 上常常无效),agent 真要键入前需要 `left_click(cx, cy)` 兜底抢焦点
- **§7.1 `interface.launch(app, args)`**:返回的是 **OS PID**(int),不是 window id;agent 不能直接用这个 pid 调窗口管理接口(那些要 wid)
- **§7.1 `interface.get_desktop_environment`**:Linux 上常常返回 `'unknown'`(VM 上没 export `XDG_SESSION_DESKTOP`/`DESKTOP_SESSION`,即使桌面是 GNOME),不可靠
- **§7.1 删除 `set_wallpaper` 行**(用户已确认)
- **§7.2 `c.pty.create`**:Windows 上 `send_stdin` 之间需要 ≥2s 延迟,1s 不够;stdout 含 ANSI 转义序列,要自己清洗
- **§7.2 `c.python_exec` / `venv_exec` / `python_command` 装饰器**:函数定义**必须在真 .py 文件里**,REPL / heredoc / `exec()` 内定义会抛 `Cannot retrieve source code for function ...`(`inspect.getsource` 依赖)
- **§7.2 `c.pip_install`**:Linux 上 **不可用**——`uv pip install --system` 写 `/usr/local/lib/python3.X/dist-packages/` 需要 root,实际部署 cua-server 都是非 root。Linux 上要装系统级包请走 `interface.run_command("sudo apt install ...")` 或 `venv_install`(用户家目录,可用)
- **§1 `vnc_url` 属性**:client-only 模式且 ctor 未显式传 `vnc_url=` 时,**恒返回 `http://localhost:8006/?autoconnect=true`**——不是真 VM 的 VNC URL;agent 要 VNC 必须在 ctor 显式传 `vnc_url=` 参数
- **新发现的 helper(在 `agenthle/submodules/cua` fork 里多出的)**:`session.exists(path)`(file_exists/directory_exists 合一)、`session.makedirs(path)`(create_dir 别名)、`session.remove_file(path)`(file/dir 通用删除,Windows 退化为 powershell `Remove-Item -Recurse`)。这 3 个 SESSION_API.md 未提,但在 fork 版本里是稳定可用的兜底,可考虑加进文档

---

## 完整性自检

- **结束时间**:`2026-05-15T22:47:55Z` (UTC),总耗时 ≈ 18 分钟

- [x] 1. 每一接口行 Linux/Windows 都有值,非 SKIP 的都有证据(stdout/cursor/wid/clipboard/screenshot)
- [x] 2. 报告头部记录开始/结束时间、两 VM endpoint + `check_status=True`、cua sha + ale sha
- [x] 3. cleanup 检查清单全为 False:
  - Linux `/tmp/cua-probe`,`/home/user/.venvs/cua-test-venv`,`/tmp/cua-probe-open.txt` → `directory_exists=False file_exists=False`
  - Linux `pgrep -f gedit` → 空
  - Linux `python3 -c "import cowsay"` → `ModuleNotFoundError`(从未成功安装)
  - Windows `C:\Users\Public\cua-probe`,`C:\Users\User\.venvs\cua-test-venv`,`C:\Users\Public\cua-probe-open.txt` → 全 False
  - Windows `python -c "import cowsay"` → `ModuleNotFoundError`(已 uninstall)
  - Windows `get_application_windows("notepad")` → `[]`
  - 两边剪贴板最终设为 `' '`(空字符串会触发 `Failed to get clipboard content`,这点也是个小坑)
- [x] 4. Linux `/tmp/cua-probe` 不存在;Windows `C:\Users\Public\cua-probe` 不存在(同 #3)
- [x] 5. 文档需修订清单含 `set_wallpaper` 删除项(已记)
