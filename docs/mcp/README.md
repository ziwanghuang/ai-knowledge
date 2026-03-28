# 🔌 MCP — Model Context Protocol

## 概述

MCP（Model Context Protocol，模型上下文协议）是由 Anthropic 提出的开放标准协议，旨在为 AI 模型提供统一的外部工具和数据源接入方式。MCP 定义了 AI 应用与外部服务之间的标准化通信接口，使 LLM 能够安全、高效地访问各种工具和资源。类比 USB-C 接口：为 AI 应用提供统一的「连接器」标准。

## 核心知识体系

### 1. MCP 基础概念

- **协议架构**：Client-Server 模式，AI 应用作为 Client，工具服务作为 Server
- **消息格式**：基于 JSON-RPC 2.0 的标准化消息格式
- **核心原语**：
  - **Tools（工具）**：Agent 可调用的函数/操作（如搜索、数据库查询、文件操作）
  - **Resources（资源）**：Agent 可读取的上下文信息（如文件内容、数据库记录）
  - **Prompts（提示模板）**：预定义的交互模板，标准化常见操作流程
- **传输层**：支持 stdio、HTTP/SSE、Streamable HTTP 等多种传输方式

### 2. 协议工作流程

```
AI 应用 (Host)
    ↕ MCP Client
    ↕ 传输层 (stdio / HTTP+SSE / Streamable HTTP)
    ↕ MCP Server
    ↕ 外部工具 / 数据源
```

1. **初始化**：Client 与 Server 建立连接，协商能力
2. **发现**：Client 获取 Server 提供的工具/资源列表
3. **调用**：模型决定调用某个工具，Client 发送请求
4. **响应**：Server 执行操作并返回结果

### 3. MCP Server 开发

- **SDK 支持**：Python SDK、TypeScript SDK、Java SDK 等
- **工具定义**：使用 JSON Schema 描述工具的输入输出规范
- **资源暴露**：将文件、数据库、API 等封装为标准资源，实现 URI 模式和内容提供
- **生命周期管理**：初始化、请求处理、关闭
- **错误处理**：标准化的错误码与错误信息

### 4. MCP Client 集成

- **IDE 集成**：Cursor、VS Code (Copilot)、Windsurf 等
- **Agent 框架集成**：在 LangChain、LlamaIndex 等框架中集成 MCP Client
- **AI 平台**：Claude Desktop、ChatGPT 等
- **工具发现与能力协商**

### 5. Transport 传输层

| 传输方式 | 特点 | 适用场景 |
|---------|------|---------|
| **stdio** | 标准输入输出 | 本地进程间通信，最常用 |
| **SSE（Server-Sent Events）** | HTTP 流式传输 | 远程服务 |
| **Streamable HTTP** | 新一代 HTTP 传输，支持双向通信 | 远程服务（推荐） |

### 6. 常见 MCP Server 类型

| 类型 | 示例 |
|------|------|
| 文件系统 | 读写本地文件、目录操作 |
| 数据库 | SQLite、PostgreSQL、MySQL 查询 |
| Web 服务 | GitHub API、Slack、Jira 集成 |
| 搜索引擎 | Brave Search、Google Search |
| 开发工具 | Git 操作、Docker 管理 |
| 知识库 | 向量数据库检索、文档管理 |

### 7. 安全与最佳实践

- **权限控制**：最小权限原则，限制工具的访问范围
- **输入验证**：严格校验工具调用参数
- **审计日志**：记录所有工具调用行为
- **沙箱隔离**：在受限环境中执行敏感操作
- **用户确认**：关键操作需用户显式授权

### 8. 生态与互操作

- **官方 Server 集合**：文件系统、GitHub、Slack、数据库等
- **社区生态**：开源 MCP Server 仓库、工具市场
- **与其他规范的互操作**：OpenAI Function Calling、Google Tools 等规范的兼容与转换

## 学习路线建议

1. 理解 MCP 协议的设计理念与基本架构
2. 使用现有的 MCP Server（如文件系统、搜索）
3. 学习使用 Python/TypeScript SDK 开发自定义 MCP Server
4. 掌握工具定义、资源暴露与错误处理
5. 在 Agent 框架中集成 MCP Client
6. 实践生产级 MCP 服务的安全与运维

## 推荐资源

- 📘 [MCP 官方文档](https://modelcontextprotocol.io/)
- 📘 [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- 📘 [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- 🔗 [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)
- 📘 [MCP 规范](https://spec.modelcontextprotocol.io/)
