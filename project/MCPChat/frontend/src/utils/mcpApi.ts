// MCP 后端 API 客户端
const API_BASE_URL = 'http://localhost:8002';

// 类型定义
export interface MCPToolResponse {
  name: string;
  description: string;
  type: 'stdio' | 'sse';
  active: boolean;
  config_json: string;
}

export interface MCPToolConfig {
  name: string;
  description?: string;
  type: 'stdio' | 'sse';
  config: any;
}

export interface TestConnectionResponse {
  success: boolean;
  message: string;
}

export interface ToggleResponse {
  status: string;
  active: boolean;
}

export interface DeleteResponse {
  status: string;
}

export interface RecommendedTool {
  name: string;
  description: string;
  recommend_reason: string;
  installed: boolean;
  type: 'stdio' | 'sse';
  default_config: any;
}

export interface BatchInstallRequest {
  tools: MCPToolConfig[];
}

// API 函数

/**
 * 获取工具列表
 */
export async function getMCPToolList(): Promise<MCPToolResponse[]> {
  const response = await fetch(`${API_BASE_URL}/mcp/list`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch tool list: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 切换工具激活状态
 */
export async function toggleMCPTool(
  toolName: string,
  active: boolean
): Promise<ToggleResponse> {
  const response = await fetch(`${API_BASE_URL}/mcp/toggle/${encodeURIComponent(toolName)}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ active }),
  });

  if (!response.ok) {
    throw new Error(`Failed to toggle tool: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 测试连接
 */
export async function testMCPConnection(
  toolConfig: MCPToolConfig
): Promise<TestConnectionResponse> {
  console.log('📡 [API] 发送测试连接请求:', {
    url: `${API_BASE_URL}/mcp/test_connection`,
    method: 'POST',
    body: toolConfig
  });

  const response = await fetch(`${API_BASE_URL}/mcp/test_connection`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(toolConfig),
  });

  console.log('📡 [API] 收到响应:', {
    status: response.status,
    statusText: response.statusText,
    ok: response.ok
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    console.error('❌ [API] 测试连接失败:', errorData);
    throw new Error(errorData.detail || `Failed to test connection: ${response.statusText}`);
  }

  const result = await response.json();
  console.log('✅ [API] 测试连接成功:', result);
  return result;
}

/**
 * 安装/更新工具
 */
export async function installMCPTool(
  toolConfig: MCPToolConfig
): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/mcp/install`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(toolConfig),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to install tool: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 删除工具
 */
export async function deleteMCPTool(toolName: string): Promise<DeleteResponse> {
  const response = await fetch(`${API_BASE_URL}/mcp/${encodeURIComponent(toolName)}`, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to delete tool: ${response.statusText}`);
  }

  return response.json();
}

/**
 * AI 智能搜索推荐工具
 */
export async function searchMCPToolsAI(query: string): Promise<RecommendedTool[]> {
  const response = await fetch(`${API_BASE_URL}/mcp/search_ai`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(`Failed to search tools: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 批量安装工具
 */
export async function batchInstallMCPTools(
  tools: MCPToolConfig[]
): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/mcp/install_batch`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ tools }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to batch install tools: ${response.statusText}`);
  }

  return response.json();
}