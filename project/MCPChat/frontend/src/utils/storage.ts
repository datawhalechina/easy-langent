// Local storage utilities for conversations and MCP tools

import { Conversation, MCPTool } from '../types';

const CONVERSATIONS_KEY = 'deepseek_conversations';
const MCP_TOOLS_KEY = 'deepseek_mcp_tools';
const ACTIVE_CONVERSATION_KEY = 'deepseek_active_conversation';

export const storage = {
  // Conversations
  getConversations: (): Conversation[] => {
    if (typeof window === 'undefined') return [];
    const data = localStorage.getItem(CONVERSATIONS_KEY);
    return data ? JSON.parse(data) : [];
  },

  saveConversations: (conversations: Conversation[]) => {
    if (typeof window === 'undefined') return;
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
  },

  getActiveConversationId: (): string | null => {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(ACTIVE_CONVERSATION_KEY);
  },

  setActiveConversationId: (id: string) => {
    if (typeof window === 'undefined') return;
    localStorage.setItem(ACTIVE_CONVERSATION_KEY, id);
  },

  // MCP Tools
  getMCPTools: (): MCPTool[] => {
    if (typeof window === 'undefined') return [];
    const data = localStorage.getItem(MCP_TOOLS_KEY);
    return data ? JSON.parse(data) : [];
  },

  saveMCPTools: (tools: MCPTool[]) => {
    if (typeof window === 'undefined') return;
    localStorage.setItem(MCP_TOOLS_KEY, JSON.stringify(tools));
  },
};
