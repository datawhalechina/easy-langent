// Type definitions for the application

// 工具调用状态
export interface ToolCallBlock {
  id: string;
  type: 'tool_call';
  toolName: string;
  input: any;
  output?: any;
  status: 'running' | 'completed';
  timestamp: number;
  isExpanded?: boolean; // 是否展开查看详情
}

// 文本块类型
export interface TextBlock {
  id: string;
  type: 'text';
  content: string;
  timestamp: number;
}

// 消息块类型 - 支持流式响应（保留兼容性）
export interface MessageChunk {
  id: string;
  type: 'thought' | 'tool_call' | 'content';
  content: string;
  timestamp: number;
  isStreaming?: boolean; // 是否正在流式输出
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content?: string; // 用户消息的内容
  chunks?: MessageChunk[]; // AI消息的chunks（旧格式，保留兼容性）
  blocks?: (TextBlock | ToolCallBlock)[]; // 新格式：文本块和工具块混合
  timestamp: number;
  isComplete?: boolean; // 消息是否完成
  isStreaming?: boolean; // 是否正在流式输出（用于显示光标）
}

export interface ThoughtStep {
  type: 'thought' | 'action' | 'observation';
  content: string;
}

export interface SearchResult {
  title: string;
  site: string;
  description: string;
  icon: string;
  url?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface MCPTool {
  id: string;
  name: string;
  description: string;
  icon: string;
  iconBg: string;
  introduction: string; // 工具的详细介绍
  config: MCPConfig;
  enabled: boolean;
  version?: string;
  author?: string;
  type?: 'stdio' | 'sse'; // 后端返回的类型
}

export interface MCPConfig {
  mcpServers: {
    [key: string]: {
      command: string;
      args: string[];
      env?: Record<string, string>;
    };
  };
}

export interface WishAnalysisResult {
  wish: string;
  suggestedTools: SuggestedTool[];
}

export interface SuggestedTool {
  name: string;
  description: string;
  icon: string;
  iconBg: string;
  functions: string[];
  config: MCPConfig;
  isNew: boolean;
  recommendReason?: string; // AI 推荐理由
  installed?: boolean; // 是否已安装
  type?: 'stdio' | 'sse'; // 工具类型
  defaultConfig?: any; // 默认配置
}