# MCP 深度指南：从协议原理到实战开发

> 本文面向有 AI Agent 使用经验、了解 MCP 基本概念但希望深入理解协议全貌的开发者。

---

## 一、MCP 是什么

**一句话**：MCP（Model Context Protocol）是 AI 应用连接外部工具和数据的**统一通信协议**。

### 类比理解

把 AI 应用想象成你的笔记本电脑，各种外部工具（数据库、搜索引擎、文件系统、API）想象成外设（键盘、显示器、硬盘）。

**没有 MCP 之前**：每个外设用不同的接口——有的用 USB-A，有的用 HDMI，有的用雷电口。你每接一个设备，都要找对应的线、装对应的驱动。

**有了 MCP 之后**：所有设备统一用 USB-C。一根线、一个标准，即插即用。

MCP 就是 AI 世界的 USB-C 标准。

```
                    ┌──────────────┐
                    │   AI 应用     │   ← 你的笔记本电脑
                    │  (Host)      │
                    └──────┬───────┘
                           │ USB-C 接口 = MCP 协议
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴────┐ ┌────┴─────┐
        │ 数据库查询  │ │ 搜索引擎 │ │ 文件系统  │  ← 外设
        │ MCP Server │ │ Server │ │ Server   │
        └───────────┘ └────────┘ └──────────┘
```

---

## 二、为什么需要 MCP

### 没有 MCP 的世界

在 MCP 出现之前，AI Agent 要调用外部工具，通常有这几种方式：

| 方式 | 做法 | 痛点 |
|------|------|------|
| **Function Calling** | 在 prompt 里定义函数 schema，模型输出调用意图，应用层执行 | 每个 LLM 厂商格式不同，换模型就要改代码 |
| **LangChain Tools** | 用框架封装工具类 | 绑定特定框架，工具不能跨框架复用 |
| **自定义 API** | 自己写接口对接 | 每接一个工具都要写适配代码，N 个工具 × M 个应用 = N×M 的工作量 |

**核心痛点**：每次连接都是"定制化"的。工具开发者和应用开发者之间没有共同语言。

### MCP 解决了什么

```
没有 MCP：N 个应用 × M 个工具 = N × M 个适配器
有了 MCP：N 个应用 + M 个工具 = N + M 个适配器（各自实现一次协议即可）
```

具体来说：

1. **标准化接口**：工具开发者只要实现一次 MCP Server，任何支持 MCP 的应用都能用
2. **动态发现**：应用启动时自动发现 Server 提供了哪些工具，不需要硬编码
3. **双向通信**：Server 不仅能被调用，还能反过来请求 LLM（Sampling），支持更复杂的 Agent 行为
4. **安全边界**：协议层面定义了权限控制、用户确认等安全机制

### MCP vs Function Calling vs LangChain Tools vs OpenAPI

| 维度 | MCP | Function Calling | LangChain Tools | OpenAPI |
|------|-----|-----------------|-----------------|---------|
| **本质** | 通信协议 | LLM 能力 | 框架抽象 | API 描述规范 |
| **标准化** | ✅ 开放协议 | ❌ 厂商各异 | ❌ 框架绑定 | ✅ 通用标准 |
| **双向通信** | ✅ Server 可请求 LLM | ❌ 单向调用 | ❌ 单向调用 | ❌ 单向调用 |
| **动态发现** | ✅ 运行时发现 | ❌ 预定义 | ❌ 代码中注册 | ✅ 通过 spec 文件 |
| **有状态连接** | ✅ 支持会话 | ❌ 无状态 | 视实现而定 | ❌ 无状态 |
| **生态** | 快速增长 | LLM 厂商内生态 | Python 生态 | 广泛 |
| **适用场景** | AI 应用↔工具 | 单次 LLM 调用 | Python Agent | 通用 API 对接 |

**一个关键区别**：Function Calling 是"模型的能力"——模型输出 JSON 表示想调用什么函数。MCP 是"应用的协议"——定义了应用和工具之间怎么通信。它们是不同层面的东西，可以互补：模型通过 Function Calling 表达意图 → 应用通过 MCP 执行调用。

---

## 三、协议架构

### Host / Client / Server 三角关系

MCP 的架构由三个角色组成：

```
┌──────────────────────────────────────────────────┐
│                    Host（宿主）                     │
│  例如：Claude Desktop、Cursor、你自己的 Agent 应用   │
│                                                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│   │ Client A │  │ Client B │  │ Client C │       │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└────────┼──────────────┼──────────────┼────────────┘
         │              │              │
    ┌────┴─────┐  ┌────┴─────┐  ┌────┴─────┐
    │ Server A │  │ Server B │  │ Server C │
    │ (文件系统) │  │ (GitHub) │  │ (数据库)  │
    └──────────┘  └──────────┘  └──────────┘
```

用**餐厅类比**来理解这三个角色：

| MCP 角色 | 餐厅类比 | 职责 |
|----------|---------|------|
| **Host** | 餐厅本身 | 提供整体环境，管理所有服务员，决定开放哪些服务 |
| **Client** | 服务员 | 每个服务员对接一个厨房窗口，负责传菜和点单 |
| **Server** | 后厨工位 | 每个工位专门做一类菜（文件操作、数据库查询等） |

**关键点**：
- 一个 Host 可以创建多个 Client
- 每个 Client 与恰好一个 Server 保持 1:1 连接
- Client 之间相互隔离，Server A 不知道 Server B 的存在

### 通信协议：JSON-RPC 2.0

MCP 所有消息都基于 JSON-RPC 2.0 格式。这个格式非常简单：

**请求（Request）**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "query_database",
    "arguments": {
      "sql": "SELECT * FROM users LIMIT 5"
    }
  }
}
```

**响应（Response）**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "查询返回 5 条记录..."
      }
    ]
  }
}
```

**通知（Notification）**：没有 `id` 字段，不需要回复
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/progress",
  "params": {
    "progressToken": "abc",
    "progress": 50,
    "total": 100
  }
}
```

### 连接生命周期

一次完整的 MCP 会话经历以下阶段：

```
Client                                    Server
  │                                         │
  │  ①  initialize (发送客户端能力)           │
  │ ───────────────────────────────────────> │
  │                                         │
  │  ②  initialize response (返回服务器能力)  │
  │ <─────────────────────────────────────── │
  │                                         │
  │  ③  initialized (确认初始化完成)          │
  │ ───────────────────────────────────────> │
  │                                         │
  │          === 正常通信阶段 ===              │
  │                                         │
  │  ④  tools/list (发现有哪些工具)           │
  │ ───────────────────────────────────────> │
  │  ⑤  返回工具列表                         │
  │ <─────────────────────────────────────── │
  │                                         │
  │  ⑥  tools/call (调用某个工具)             │
  │ ───────────────────────────────────────> │
  │  ⑦  返回执行结果                         │
  │ <─────────────────────────────────────── │
  │                                         │
  │          === 关闭阶段 ===                 │
  │                                         │
  │  ⑧  close / disconnect                  │
  │ ───────────────────────────────────────> │
```

**能力协商**是关键步骤。Client 和 Server 在握手时互相告知自己支持什么功能：

```json
// Client 告诉 Server：我支持 sampling 和 roots
{
  "capabilities": {
    "sampling": {},
    "roots": { "listChanged": true }
  }
}

// Server 告诉 Client：我提供 tools 和 resources
{
  "capabilities": {
    "tools": { "listChanged": true },
    "resources": { "subscribe": true }
  }
}
```

这样双方只使用对方支持的功能，避免调用了对方不支持的方法导致出错。

### 传输层

MCP 支持三种传输方式：

#### 1. stdio（标准输入输出）

最简单的方式。Host 启动 Server 进程，通过 stdin/stdout 通信。

```
Host 进程 ──stdin──> Server 进程
Host 进程 <──stdout── Server 进程
```

**案例**：Claude Desktop 配置一个本地 MCP Server：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/docs"]
    }
  }
}
```

Host 通过 `npx` 启动 Server 进程，两者通过 stdio 管道通信。

**适用场景**：本地工具、开发调试、安全性要求高（不经过网络）

#### 2. Streamable HTTP（推荐的远程方式）

2025 年 3 月引入，替代了之前的 SSE 方式。只需一个 HTTP 端点。

```
Client ────POST /mcp───> Server
       <───响应/流式响应────
```

**为什么淘汰 SSE，改用 Streamable HTTP？**

| 对比点 | SSE（旧方案） | Streamable HTTP（新方案） |
|--------|-------------|------------------------|
| 端点数 | 两个（`/sse` + `/sse/messages`） | 一个（`/mcp`） |
| 通信方向 | 本质单向，需额外端点实现双向 | 原生双向 |
| 资源消耗 | 长连接常驻，空闲也占资源 | 按需连接，简单请求无需持久连接 |
| 扩展性 | 难（受限于长连接数量） | 好（符合标准 HTTP 模型） |
| HTTP/2&3 兼容 | 有兼容性问题 | 完全兼容 |

简单说：Streamable HTTP 做到了"需要流式就流式，不需要就普通请求"，更灵活也更省资源。

**适用场景**：远程服务、生产部署、需要横向扩展

#### 3. stdio vs Streamable HTTP 怎么选？

```
你的 Server 跑在本地？         → stdio
你的 Server 跑在远程/云上？     → Streamable HTTP
你在开发调试阶段？             → stdio（简单）
你要部署给多个用户用？         → Streamable HTTP
```

---

## 四、核心概念详解

MCP 定义了 5 个核心原语（Primitives），可以分为两组：

```
Server → Client（Server 提供给 Client 的能力）
├── Resources  资源
├── Tools      工具
└── Prompts    提示模板

Client → Server（Client 提供给 Server 的能力）
├── Sampling   采样（让 Server 可以请求 LLM）
└── Roots      根目录（告诉 Server 可以操作的范围）
```

### 4.1 Resources（资源）

**是什么**：Server 暴露给 Client 的数据内容。类比为"文件共享"。

**和 API endpoint 的区别**：
- API endpoint 是"做一个动作"（执行查询、发送请求）
- Resource 是"给你一份数据"（这里有个文件、这里有条记录）

**案例**：一个文件系统 MCP Server 暴露资源

```json
// Client 请求：列出可用资源
{ "method": "resources/list" }

// Server 响应：
{
  "resources": [
    {
      "uri": "file:///Users/me/docs/readme.md",
      "name": "项目说明文档",
      "mimeType": "text/markdown"
    },
    {
      "uri": "db://users/active-count",
      "name": "当前活跃用户数",
      "mimeType": "text/plain"
    }
  ]
}

// Client 请求读取某个资源：
{
  "method": "resources/read",
  "params": { "uri": "file:///Users/me/docs/readme.md" }
}

// Server 返回资源内容：
{
  "contents": [
    {
      "uri": "file:///Users/me/docs/readme.md",
      "mimeType": "text/markdown",
      "text": "# My Project\n这是项目说明..."
    }
  ]
}
```

**Resource Templates**：支持 URI 模板，动态匹配资源

```json
{
  "uriTemplate": "db://users/{userId}/profile",
  "name": "用户资料",
  "mimeType": "application/json"
}
```

Client 可以请求 `db://users/12345/profile` 来获取特定用户的资料。

**关键理解**：Resource 是被动提供数据的，**不会产生副作用**。你读一个资源不会改变任何东西，就像你打开一个文件看看内容一样。

### 4.2 Tools（工具）

**是什么**：Server 暴露给 AI 模型调用的**可执行函数**。这是 MCP 中最核心也最常用的概念。

**和 Resource 的区别**：
- Resource = 看一看（只读、无副作用）
- Tool = 做一做（执行操作、可能有副作用）

**案例**：一个数据库 MCP Server 暴露的工具

```json
// Client 请求：有哪些工具？
{ "method": "tools/list" }

// Server 响应：
{
  "tools": [
    {
      "name": "query_database",
      "description": "执行 SQL 查询并返回结果。仅支持 SELECT 语句。",
      "inputSchema": {
        "type": "object",
        "properties": {
          "sql": {
            "type": "string",
            "description": "要执行的 SQL 查询语句"
          },
          "database": {
            "type": "string",
            "description": "目标数据库名",
            "default": "production"
          }
        },
        "required": ["sql"]
      }
    }
  ]
}
```

调用工具：

```json
// Client 请求：调用工具
{
  "method": "tools/call",
  "params": {
    "name": "query_database",
    "arguments": {
      "sql": "SELECT name, email FROM users WHERE active = true LIMIT 3"
    }
  }
}

// Server 响应：
{
  "content": [
    {
      "type": "text",
      "text": "| name | email |\n|------|-------|\n| Alice | alice@example.com |\n| Bob | bob@example.com |\n| Carol | carol@example.com |"
    }
  ]
}
```

**工具的完整流程**：

```
用户提问 "查一下活跃用户有多少"
       ↓
Host 把问题 + 工具列表一起发给 LLM
       ↓
LLM 决定："我要调用 query_database 工具"
       ↓
Host 通过 Client 向 Server 发 tools/call
       ↓
Server 执行 SQL，返回结果
       ↓
Host 把结果再交给 LLM
       ↓
LLM 生成回答："目前有 1,234 个活跃用户"
```

**inputSchema 用的是 JSON Schema**：这是一个已有的广泛使用的标准，LLM 天然能理解 JSON Schema 描述的参数格式。

### 4.3 Prompts（提示模板）

**是什么**：Server 预定义的、可复用的提示词模板。

这个概念不太好理解，用一个案例来说明：

**案例**：一个代码审查 MCP Server

```json
// Server 提供的 prompt 模板：
{
  "prompts": [
    {
      "name": "code_review",
      "description": "对一段代码进行安全审查",
      "arguments": [
        {
          "name": "code",
          "description": "要审查的代码",
          "required": true
        },
        {
          "name": "language",
          "description": "编程语言",
          "required": false
        }
      ]
    }
  ]
}

// Client 获取填充后的 prompt：
{
  "method": "prompts/get",
  "params": {
    "name": "code_review",
    "arguments": {
      "code": "SELECT * FROM users WHERE id = '" + userId + "'",
      "language": "sql"
    }
  }
}

// Server 返回组装好的消息：
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "请对以下 sql 代码进行安全审查，重点检查注入风险、权限问题和性能隐患：\n\n```sql\nSELECT * FROM users WHERE id = '\" + userId + \"'\n```\n\n请按以下格式输出：\n1. 风险等级（高/中/低）\n2. 问题描述\n3. 修复建议"
      }
    }
  ]
}
```

**Prompt 的价值**：
- Server 开发者是领域专家，他们知道最好的提问方式
- 把专家级的 prompt 封装起来，用户只需填参数
- 类似于 SQL 中的"存储过程"——封装复杂逻辑，暴露简单接口

**和 Tool 的区别**：
- Tool 是给 **AI 模型**调用的（模型决定何时调用）
- Prompt 是给**用户**使用的（用户选择使用哪个模板）

### 4.4 Sampling（采样）

**是什么**：允许 Server **反向请求 Client/Host 去调用 LLM**。

这是 MCP 最独特的设计之一。通常我们认为数据流是：

```
用户 → Host → LLM → Host → Server（调工具）→ Host → LLM → 用户
```

但 Sampling 允许：

```
Server 说："我需要 LLM 帮我分析一下这段数据"
   ↓
Client 收到请求，转交给 Host
   ↓
Host 调用 LLM
   ↓
LLM 返回结果
   ↓
Host 通过 Client 把结果交给 Server
```

**案例**：一个数据分析 MCP Server

```json
// Server 向 Client 发起 sampling 请求：
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "以下是过去 7 天的服务器错误日志摘要，请识别最关键的 3 个问题：\n\n[ERROR] 2026-04-01 OOM in worker-3 (5 次)\n[ERROR] 2026-04-02 Connection timeout to DB (23 次)\n[ERROR] 2026-04-03 Disk full on node-7 (1 次)..."
        }
      }
    ],
    "maxTokens": 500
  }
}

// Host 调用 LLM 后，返回分析结果给 Server：
{
  "role": "assistant",
  "content": {
    "type": "text",
    "text": "最关键的 3 个问题：\n1. 数据库连接超时（23 次）- 需要立即排查...\n2. ..."
  }
}
```

**为什么需要这个能力**：
1. Server 可以做"Agent 式"的复杂任务——自己收集数据，请 LLM 分析，再根据分析结果决定下一步
2. 实现递归/多步推理——Server 处理到一半发现需要 LLM 的判断
3. 让工具更智能——不只是执行命令，还能理解和推理

**安全设计**：Host 始终有控制权。Server 的 sampling 请求必须经过 Host 审核，Host 可以修改 prompt、限制 token、甚至拒绝请求。Server 看不到完整的 prompt（Host 可以注入系统提示），这确保了用户隐私。

### 4.5 Roots（根目录）

**是什么**：Client 告诉 Server "你可以操作的范围在哪里"。

**案例**：

```json
// Client 告诉 Server 可以操作的根目录：
{
  "method": "notifications/roots/list_changed"
}

// Server 查询 roots：
{
  "method": "roots/list"
}

// Client 返回：
{
  "roots": [
    {
      "uri": "file:///Users/me/projects/my-app",
      "name": "当前项目"
    },
    {
      "uri": "file:///Users/me/docs",
      "name": "文档目录"
    }
  ]
}
```

**关键理解**：Roots 是**建议性的**，不是强制的安全沙箱。它告诉 Server "这些是你应该关注的目录"，但协议层面不强制限制 Server 只能访问这些路径。真正的安全限制需要 Server 实现层面来保障。

---

## 五、MCP Server 开发实战

### 用 Python SDK 写一个天气查询 Server

这是一个完整的、可运行的例子：

```python
# weather_server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json
import httpx

# 创建 Server 实例
server = Server("weather-server")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """声明这个 Server 提供哪些工具"""
    return [
        Tool(
            name="get_weather",
            description="查询指定城市的当前天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名（英文），如 Beijing, Shanghai"
                    }
                },
                "required": ["city"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    if name == "get_weather":
        city = arguments["city"]

        # 调用天气 API（这里用 wttr.in 的免费接口）
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://wttr.in/{city}?format=j1"
            )
            data = response.json()

        current = data["current_condition"][0]
        result = (
            f"🌤 {city} 当前天气:\n"
            f"温度: {current['temp_C']}°C\n"
            f"体感: {current['FeelsLikeC']}°C\n"
            f"天气: {current['weatherDesc'][0]['value']}\n"
            f"湿度: {current['humidity']}%\n"
            f"风速: {current['windspeedKmph']} km/h"
        )

        return [TextContent(type="text", text=result)]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """启动 Server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 同时提供 Resource 和 Prompt

扩展上面的例子，加上资源和提示模板：

```python
from mcp.types import Resource, Prompt, PromptMessage, PromptArgument

# 暴露一个资源：支持的城市列表
@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="weather://supported-cities",
            name="支持查询的城市列表",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "weather://supported-cities":
        cities = ["Beijing", "Shanghai", "Shenzhen", "Guangzhou", "Hangzhou"]
        return json.dumps(cities, ensure_ascii=False)
    raise ValueError(f"Unknown resource: {uri}")


# 提供一个 prompt 模板：旅行天气建议
@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="travel_weather_advice",
            description="根据目的地天气给出旅行穿搭建议",
            arguments=[
                PromptArgument(
                    name="destination",
                    description="旅行目的地",
                    required=True
                )
            ]
        )
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    if name == "travel_weather_advice":
        dest = arguments["destination"]
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"我计划去 {dest} 旅行。请先查询当地天气，然后给我以下建议：\n"
                         f"1. 穿搭建议（上衣、下装、鞋子）\n"
                         f"2. 需要携带的物品（雨伞、防晒等）\n"
                         f"3. 适合的户外活动"
                )
            )
        ]
    raise ValueError(f"Unknown prompt: {name}")
```

### 调试与测试

**MCP Inspector**：官方提供的可视化调试工具

```bash
npx @modelcontextprotocol/inspector python weather_server.py
```

这会启动一个 Web 界面，你可以：
- 查看 Server 提供的所有 Tools / Resources / Prompts
- 手动调用工具、查看请求和响应的完整 JSON
- 测试各种边界情况

### 部署方式

**本地 stdio**（开发/个人使用）：

```json
// Claude Desktop 配置 ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/path/to/weather_server.py"]
    }
  }
}
```

**远程 Streamable HTTP**（生产/团队使用）：

```python
# 使用 Starlette 部署为 HTTP 服务
from mcp.server.streamable_http import StreamableHTTPServer

app = StreamableHTTPServer(server, path="/mcp")

# 然后用 uvicorn 启动：
# uvicorn weather_server:app --host 0.0.0.0 --port 8000
```

Client 配置：
```json
{
  "mcpServers": {
    "weather": {
      "url": "https://your-server.com/mcp"
    }
  }
}
```

---

## 六、MCP Client 端

### Client 如何发现和连接 Server

大多数情况下你不需要自己写 Client——IDE 插件和 Claude Desktop 已经内置了。但理解 Client 的工作方式有助于排查问题。

**连接流程**：

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 配置要连接的 Server
server_params = StdioServerParameters(
    command="python",
    args=["weather_server.py"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # ① 初始化连接（自动完成能力协商）
        await session.initialize()

        # ② 发现工具
        tools = await session.list_tools()
        print(f"可用工具: {[t.name for t in tools.tools]}")
        # 输出: 可用工具: ['get_weather']

        # ③ 调用工具
        result = await session.call_tool(
            "get_weather",
            arguments={"city": "Shenzhen"}
        )
        print(result.content[0].text)
```

### 多 Server 管理

当你的 Host 连接了多个 Server 时：

```
Host
├── Client A → Server: filesystem（文件操作）
├── Client B → Server: database（数据库查询）
└── Client C → Server: github（GitHub 操作）
```

**Host 需要做的**：
1. 汇总所有 Server 的工具列表，去重后一起给 LLM
2. LLM 选择调用某个工具时，Host 知道该路由到哪个 Client
3. 工具名要全局唯一（如果两个 Server 都有 `search` 工具，需要命名空间区分）

**实际配置案例**（Claude Desktop）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/docs"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxx"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sqlite", "/Users/me/data.db"]
    }
  }
}
```

---

## 七、安全模型

MCP 的安全设计围绕一个核心原则：**用户始终掌握最终控制权**。

### 信任边界

```
┌─────────────────────────────────────────────┐
│  用户信任域                                   │
│  ┌───────────────────────────────────┐      │
│  │  Host（用户直接操作的应用）          │      │
│  │  ┌─────────────┐                  │      │
│  │  │  Client      │                  │      │
│  │  └──────┬──────┘                  │      │
│  └─────────┼─────────────────────────┘      │
│            │  ← 信任边界                     │
│  ┌─────────┼─────────────────────────┐      │
│  │  Server │（第三方代码）              │      │
│  │  可能是本地进程，也可能是远程服务     │      │
│  └─────────────────────────────────────┘      │
└─────────────────────────────────────────────┘
```

### Human-in-the-Loop

MCP 协议要求**敏感操作必须经过用户确认**：

```
LLM: "我想调用 delete_file 工具删除 /tmp/old_data.csv"
       ↓
Host 弹窗: "AI 要删除文件 /tmp/old_data.csv，是否允许？ [允许] [拒绝]"
       ↓
用户点击 [允许]
       ↓
Host 发送 tools/call 给 Server
```

**哪些操作应该要求确认**：
- 写入/删除文件
- 执行数据库写操作
- 发送消息/邮件
- 调用付费 API
- 任何不可逆的操作

这是 Host 的责任，不是 Server 的。Server 只管执行，Host 负责把关。

### OAuth 2.1 认证（远程 Server）

当 Server 部署在远程时，需要认证机制。MCP 采用 OAuth 2.1：

```
Client                   Server              Authorization Server
  │                        │                        │
  │  ① 请求连接             │                        │
  │ ──────────────────────> │                        │
  │                        │                        │
  │  ② 返回 401 + 认证信息  │                        │
  │ <────────────────────── │                        │
  │                        │                        │
  │  ③ 重定向用户去授权                               │
  │ ───────────────────────────────────────────────> │
  │                        │                        │
  │  ④ 用户登录并授权                                 │
  │ <─────────────────────────────────────────────── │
  │                        │                        │
  │  ⑤ 携带 token 重新请求  │                        │
  │ ──────────────────────> │                        │
  │                        │  ⑥ 验证 token            │
  │                        │ ──────────────────────> │
  │                        │                        │
  │  ⑦ 正常通信             │                        │
  │ <────────────────────── │                        │
```

2025-06-18 版本新增：
- MCP Server 被明确定义为 **OAuth 资源服务器**
- 强制要求 **RFC 8707 Resource Indicators**，防止恶意 Server 窃取 token

### 常见安全风险

| 风险 | 描述 | 防范 |
|------|------|------|
| **工具注入** | 恶意 Server 的工具描述中包含误导性指令 | Host 不要盲目信任工具 description |
| **Prompt 注入** | Server 通过 Resource/Tool 返回的内容中嵌入恶意指令 | LLM 层面做输入净化 |
| **Token 劫持** | 恶意 Server 利用 OAuth 流程获取过多权限 | RFC 8707 Resource Indicators |
| **数据泄露** | Server 将用户数据发往第三方 | 审计 Server 代码，使用可信来源 |
| **越权操作** | Server 超出预期范围操作 | Roots 限定 + Host 层面权限控制 |

---

## 八、生态与实际应用

### 主流 MCP 客户端

| 客户端 | 类型 | MCP 支持情况 |
|--------|------|-------------|
| **Claude Desktop** | AI 对话应用 | 原生支持，最完整的 MCP 实现 |
| **Cursor** | AI IDE | 深度集成，支持 stdio 和 HTTP |
| **VS Code (Copilot)** | IDE 插件 | 2025 年起支持 MCP |
| **Windsurf** | AI IDE | 支持 MCP Server |
| **Continue** | IDE 插件 | 开源，支持 MCP |
| **Cline** | VS Code 插件 | 社区驱动，支持 MCP |

### 热门 MCP Server

| Server | 功能 | 典型用途 |
|--------|------|---------|
| **filesystem** | 文件读写、目录操作 | AI 读取/编辑本地文件 |
| **github** | PR、Issue、代码搜索 | AI 辅助代码审查 |
| **sqlite / postgres** | 数据库查询 | AI 自然语言查数据 |
| **brave-search** | 网页搜索 | AI 实时信息检索 |
| **puppeteer** | 浏览器自动化 | AI 操作网页 |
| **slack** | 消息发送/读取 | AI 辅助团队协作 |

### 企业级架构模式

```
                        ┌──────────────┐
                        │   API 网关    │  ← 认证、限流、日志
                        └──────┬───────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐
     │ MCP Server    │ │ MCP Server  │ │ MCP Server  │
     │ (内部知识库)   │ │ (工单系统)   │ │ (监控告警)   │
     └───────┬───────┘ └──────┬──────┘ └──────┬──────┘
             │                │                │
     ┌───────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐
     │ RAG 向量库    │ │   Jira API  │ │ Prometheus  │
     └──────────────┘ └─────────────┘ └─────────────┘
```

这种模式下：
- 每个内部系统封装为独立 MCP Server
- 统一通过 API 网关管理认证和权限
- AI Agent 可以跨系统协作（查知识库 → 创建工单 → 设置告警）

---

## 九、局限性与未来方向

### 当前不足

1. **生态碎片化**：Server 质量参差不齐，缺乏认证/审核机制。你在社区找到的 Server 可能有安全问题。

2. **调试体验**：虽然有 Inspector，但生产环境的问题排查（比如连接断开、超时）工具还不够成熟。

3. **性能开销**：每次工具调用都走 JSON-RPC，对延迟敏感的场景（如实时对话）可能是瓶颈。没有原生的批量调用支持（2025-06-18 版本甚至移除了 JSON-RPC 批处理）。

4. **状态管理**：有状态连接增加了复杂度。Server 重启后会话丢失，需要 Client 重新初始化。

5. **安全模型依赖 Host**：协议定义了安全原则，但实际执行全靠 Host 实现。如果 Host 没做好权限控制，协议本身挡不住。

### 演进方向

| 方向 | 状态 | 说明 |
|------|------|------|
| **Streamable HTTP** | ✅ 已落地 | 替代 SSE，成为远程传输标准 |
| **OAuth 2.1 + RFC 8707** | ✅ 已落地 | 增强远程 Server 认证安全 |
| **Elicitation** | ✅ 2025-06-18 新增 | Server 可在执行过程中向用户追问信息 |
| **结构化输出** | ✅ 2025-06-18 新增 | 工具调用结果支持结构化数据 |
| **Server 注册表/市场** | 🔄 进行中 | 类似 npm registry，统一的 Server 发现和安装 |
| **断点续传** | 🔄 规划中 | 连接断开后恢复未完成的请求 |
| **多模态支持** | 🔄 进行中 | 图片、音频等非文本内容的标准化传输 |

### 总结

MCP 的核心价值是**把 N×M 问题变成 N+M 问题**。它不是什么革命性的新技术，而是一个务实的标准化工作——就像 HTTP 之于 Web、SQL 之于数据库。

它的成功不取决于协议本身多完美，而取决于**生态**——有多少 Host 支持它、有多少高质量 Server 可用。从目前的趋势看（Claude、Cursor、VS Code、OpenAI 都在接入），MCP 正在成为 AI 工具集成的事实标准。

---

## 推荐资源

- 📘 [MCP 官方规范](https://spec.modelcontextprotocol.io/) — 协议的权威定义
- 📘 [MCP 官方文档](https://modelcontextprotocol.io/) — 入门教程和指南
- 📘 [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Python 开发 MCP Server/Client
- 📘 [TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — TypeScript 开发
- 🔧 [MCP Inspector](https://github.com/modelcontextprotocol/inspector) — 可视化调试工具
- 🗂️ [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers) — 社区 Server 合集
