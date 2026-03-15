import { useState, useEffect, useRef } from 'react';
import { Menu, Wrench, Send } from 'lucide-react';
import { Toaster } from 'sonner@2.0.3';
import { toast } from 'sonner@2.0.3';
import { LeftSidebar } from './components/LeftSidebar';
import { RightSidebar } from './components/RightSidebar';
import { WelcomeScreen } from './components/WelcomeScreen';
import { MessageBubble } from './components/MessageBubble';
import { LoadingIndicator } from './components/LoadingIndicator';
import { WishResultModal } from './components/WishResultModal';
import { ConfigModal } from './components/ConfigModal';
import { AddToolModal } from './components/AddToolModal';
import { GlobalSettingsModal } from './components/GlobalSettingsModal';
import { ConfirmDialog } from './components/ConfirmDialog';
import { Conversation, Message, MCPTool, WishAnalysisResult, MessageChunk, TextBlock, ToolCallBlock } from './types';
import { storage } from './utils/storage';
import { sendChatStream, SSEEvent } from './utils/sseClient';
import * as mcpApi from './utils/mcpApi';
import { convertBackendToolToMCPTool, convertMCPToolToBackendConfig, convertRecommendedToolToSuggested } from './utils/mcpHelpers';

// Unique ID generator with counter to avoid collisions
let idCounter = 0;
const generateUniqueId = (prefix: string = '') => {
  idCounter++;
  return `${prefix}${Date.now()}-${idCounter}-${Math.random().toString(36).substr(2, 9)}`;
};

// Migrate old data to ensure unique IDs
const migrateConversationData = (conversations: Conversation[]): Conversation[] => {
  return conversations.map(conv => ({
    ...conv,
    messages: conv.messages.map(msg => ({
      ...msg,
      id: msg.id || generateUniqueId('msg-'),
      chunks: msg.chunks?.map(chunk => ({
        ...chunk,
        id: chunk.id || generateUniqueId('chunk-')
      })),
      blocks: msg.blocks?.map(block => ({
        ...block,
        id: block.id || generateUniqueId('block-')
      }))
    }))
  }));
};

// Default Tavily Tool
const DEFAULT_TAVILY_TOOL: MCPTool = {
  id: 'tavily-search',
  name: 'Tavily 搜索',
  description: '强大的网络搜索工具，可以实时获取最新信息',
  icon: '🔍',
  iconBg: 'bg-blue-50 text-blue-500',
  introduction: 'Tavily 是一个专为 AI 应用优化的搜索引擎，能够快速准确地获取网络信息。',
  config: {
    mcpServers: {
      'tavily-search': {
        command: 'npx',
        args: ['-y', '@tavily/mcp-server'],
        env: {
          TAVILY_API_KEY: 'your-api-key-here'
        }
      }
    }
  },
  enabled: true,
  version: 'v1.0.0',
  author: 'Tavily'
};

export default function App() {
  // UI State
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(false);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(false);
  const [rightSidebarWidth, setRightSidebarWidth] = useState(384);
  const [isLoading, setIsLoading] = useState(false);

  // Modal State
  const [wishResultModal, setWishResultModal] = useState<{
    isOpen: boolean;
    result: WishAnalysisResult | null;
  }>({ isOpen: false, result: null });
  const [configModal, setConfigModal] = useState<{
    isOpen: boolean;
    tool: MCPTool | null;
  }>({ isOpen: false, tool: null });
  const [isGlobalSettingsOpen, setIsGlobalSettingsOpen] = useState(false);
  const [addToolModal, setAddToolModal] = useState<{
    isOpen: boolean;
    tool: MCPTool | null;
  }>({ isOpen: false, tool: null });
  const [deleteConfirm, setDeleteConfirm] = useState<{
    isOpen: boolean;
    toolId: string | null;
    toolName: string | null;
  }>({ isOpen: false, toolId: null, toolName: null });

  // Data State
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [mcpTools, setMcpTools] = useState<MCPTool[]>([]);
  const [userInput, setUserInput] = useState('');

  // Refs
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load data from localStorage on mount
  useEffect(() => {
    const savedConversations = storage.getConversations();
    const savedTools = storage.getMCPTools();
    const savedActiveId = storage.getActiveConversationId();

    if (savedConversations.length > 0) {
      setConversations(migrateConversationData(savedConversations));
      if (savedActiveId) {
        setActiveConversationId(savedActiveId);
      }
    }

    if (savedTools.length > 0) {
      // Merge with default tool
      const hasDefault = savedTools.some(t => t.id === 'tavily-search');
      setMcpTools(hasDefault ? savedTools : [DEFAULT_TAVILY_TOOL, ...savedTools]);
    }
  }, []);

  // Save to localStorage when data changes
  useEffect(() => {
    storage.saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    storage.saveMCPTools(mcpTools);
  }, [mcpTools]);

  useEffect(() => {
    if (activeConversationId) {
      storage.setActiveConversationId(activeConversationId);
    }
  }, [activeConversationId]);

  // Auto-scroll chat container
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [conversations, activeConversationId, isLoading]);
  
  useEffect(() => {
    loadMCPTools();
  }, []);

  // Get current conversation
  const currentConversation = conversations.find((c) => c.id === activeConversationId);

  // Create new chat
  const handleNewChat = () => {
    const newConv: Conversation = {
      id: generateUniqueId('conv-'),
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    setConversations([newConv, ...conversations]);
    setActiveConversationId(newConv.id);

    // Reset all tools except Tavily
    setMcpTools((tools) =>
      tools.map((tool) => ({
        ...tool,
        enabled: tool.id === 'tavily-search'
      }))
    );
  };

  // Rename conversation
  const handleRenameConversation = (id: string, newTitle: string) => {
    setConversations((prevConvs) =>
      prevConvs.map((conv) =>
        conv.id === id
          ? { ...conv, title: newTitle, updatedAt: Date.now() }
          : conv
      )
    );
  };

  // Delete conversation
  const handleDeleteConversation = (id: string) => {
    setConversations((prevConvs) => prevConvs.filter((conv) => conv.id !== id));
    
    // If deleting active conversation, switch to another one or clear
    if (id === activeConversationId) {
      const remaining = conversations.filter((conv) => conv.id !== id);
      setActiveConversationId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  // 🔥 模块一：从后端加载 MCP 工具列表
  const loadMCPTools = async () => {
    try {
      const backendTools = await mcpApi.getMCPToolList();
      const tools = backendTools.map(convertBackendToolToMCPTool);
      setMcpTools(tools);
      console.log('✅ 已从后端加载工具列表:', tools);
    } catch (error) {
      console.warn('⚠️ 后端加载失败,使用本地存储:', error);
      const savedTools = storage.getMCPTools();
      if (savedTools.length > 0) {
        setMcpTools(savedTools);
      }
    }
  };

  // 🔥 模块二:测试连接 (给 ConfigModal 使用)
  const handleTestConnection = async (
    toolName: string,
    description: string,
    type: string,
    config: any
  ): Promise<{ success: boolean; message: string }> => {
    console.log('🔌 [测试连接] 开始测试:', { toolName, description, type, config });
    
    try {
      const result = await mcpApi.testMCPConnection({
        name: toolName,
        description,
        type: type as 'stdio' | 'sse',
        config
      });
      
      console.log('✅ [测试连接] 后端返回:', result);
      return result;
    } catch (error) {
      console.error('❌ [测试连接] 后端请求失败:', error);
      return {
        success: false,
        message: error instanceof Error ? error.message : '连接测试失败'
      };
    }
  };

  // Send message with SSE streaming
  const handleSendMessageSSE = async (message?: string) => {
    const text = message || userInput.trim();
    if (!text) return;

    // Create conversation if none exists
    let convId = activeConversationId;
    if (!convId) {
      const newConv: Conversation = {
        id: generateUniqueId('conv-'),
        title: text.substring(0, 30),
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now()
      };
      setConversations([newConv, ...conversations]);
      setActiveConversationId(newConv.id);
      convId = newConv.id;
    }

    // Add user message
    const userMessage: Message = {
      id: generateUniqueId('user-msg-'),
      role: 'user',
      content: text,
      timestamp: Date.now()
    };

    setConversations((prevConvs) =>
      prevConvs.map((conv) =>
        conv.id === convId
          ? {
              ...conv,
              messages: [...conv.messages, userMessage],
              title: conv.messages.length === 0 ? text.substring(0, 30) : conv.title,
              updatedAt: Date.now()
            }
          : conv
      )
    );

    setUserInput('');

    try {
      // Create initial assistant message with blocks
      const assistantMessageId = generateUniqueId('assistant-msg-');
      const blocks: (TextBlock | ToolCallBlock)[] = [];
      
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        blocks: blocks,
        timestamp: Date.now(),
        isComplete: false,
        isStreaming: true
      };

      // Add empty assistant message
      setConversations((prevConvs) =>
        prevConvs.map((conv) =>
          conv.id === convId
            ? {
                ...conv,
                messages: [...conv.messages, assistantMessage],
                updatedAt: Date.now()
              }
            : conv
        )
      );

      // Get conversation history for context
      const currentConv = conversations.find(c => c.id === convId);
      const history = currentConv?.messages.map(msg => ({
        role: msg.role,
        content: msg.content || (msg.blocks ? 
          msg.blocks
            .filter(b => b.type === 'text')
            .map(b => (b as TextBlock).content)
            .join('\n') 
          : '')
      })) || [];

      // Handle SSE events
      await sendChatStream(
        text,
        convId,  // 使用 conversation ID 作为 session_id
        (event: SSEEvent) => {
          setConversations((prevConvs) =>
            prevConvs.map((conv) => {
              if (conv.id !== convId) return conv;

              const updatedMessages = conv.messages.map((msg) => {
                if (msg.id !== assistantMessageId) return msg;

                const newBlocks = [...(msg.blocks || [])];

                switch (event.type) {
                  case 'token': {
                    // Append text to last text block or create new one
                    const lastBlock = newBlocks[newBlocks.length - 1];
                    if (lastBlock && lastBlock.type === 'text') {
                      // Append to existing text block
                      (lastBlock as TextBlock).content += event.data.content;
                    } else {
                      // Create new text block
                      const textBlock: TextBlock = {
                        id: generateUniqueId('text-'),
                        type: 'text',
                        content: event.data.content,
                        timestamp: Date.now()
                      };
                      newBlocks.push(textBlock);
                    }
                    break;
                  }

                  case 'tool_start': {
                    // Create tool block
                    const toolBlock: ToolCallBlock = {
                      id: generateUniqueId('tool-'),
                      type: 'tool_call',
                      toolName: event.data.tool_name,
                      input: event.data.input,
                      status: 'running',
                      timestamp: Date.now(),
                      isExpanded: false
                    };
                    newBlocks.push(toolBlock);
                    break;
                  }

                  case 'tool_end': {
                    // Find the most recent running tool block with matching name
                    for (let i = newBlocks.length - 1; i >= 0; i--) {
                      const block = newBlocks[i];
                      if (
                        block.type === 'tool_call' &&
                        (block as ToolCallBlock).toolName === event.data.tool_name &&
                        (block as ToolCallBlock).status === 'running'
                      ) {
                        (block as ToolCallBlock).output = event.data.output;
                        (block as ToolCallBlock).status = 'completed';
                        break;
                      }
                    }
                    break;
                  }

                  case 'finish': {
                    // Mark message as complete
                    return { ...msg, blocks: newBlocks, isComplete: true, isStreaming: false };
                  }

                  case 'error': {
                    // Handle error from backend
                    console.error('Backend error:', event.data.message);
                    toast.error('后端错误', {
                      description: event.data.message || '后端处理出错',
                      duration: 5000
                    });
                    return { ...msg, blocks: newBlocks, isComplete: true, isStreaming: false };
                  }
                }

                return { ...msg, blocks: newBlocks };
              });

              return { ...conv, messages: updatedMessages, updatedAt: Date.now() };
            })
          );
        },
        (error) => {
          console.error('SSE error:', error);
          toast.error('连接失败', {
            description: '无法连接到后端服务，请检查服务是否启动',
            duration: 5000
          });
          setIsLoading(false);
        }
      );

      // Mark as complete
      setConversations((prevConvs) =>
        prevConvs.map((conv) =>
          conv.id === convId
            ? {
                ...conv,
                messages: conv.messages.map((msg) =>
                  msg.id === assistantMessageId
                    ? { ...msg, isComplete: true, isStreaming: false }
                    : msg
                ),
                updatedAt: Date.now()
              }
            : conv
        )
      );
    } catch (error) {
      console.error('Error generating response:', error);
      toast.error('发送失败', {
        description: '消息发送失败,请重试',
        duration: 4000
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Send message with SSE streaming
  const handleSendMessage = async (message?: string) => {
    const text = message || userInput.trim();
    if (!text) return;

    setIsLoading(true);

    try {
      await handleSendMessageSSE(text);
    } catch (error) {
      console.error('SSE failed:', error);
      toast.error('发送消息失败', {
        description: error instanceof Error ? error.message : '无法连接到服务器',
        duration: 4000,
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Handle wish processing
  const handleProcessWish = async (wish: string) => {
    try {
      const recommendedTools = await mcpApi.searchMCPToolsAI(wish);
      const suggestedTools = recommendedTools.map(convertRecommendedToolToSuggested);
      setWishResultModal({
        isOpen: true,
        result: { wish, suggestedTools }
      });
    } catch (error) {
      console.error('AI search failed:', error);
      toast.error('智能分析失败', {
        description: error instanceof Error ? error.message : '无法连接到服务器',
        duration: 4000,
      });
    }
  };

  // Confirm adding tools from wish
  const handleConfirmAddTools = async (selectedIndices: number[]) => {
    if (!wishResultModal.result || selectedIndices.length === 0) return;
  
    try {
      const toolsToInstall = selectedIndices.map((index) => {
        const tool = wishResultModal.result!.suggestedTools[index];
        return {
          name: tool.name,
          description: tool.description,
          type: (tool.type || 'stdio') as 'stdio' | 'sse',
          config: tool.defaultConfig || tool.config
        };
      });
  
      // 批量安装工具
      await mcpApi.batchInstallMCPTools(toolsToInstall);
      
      // 🔥 后端默认是激活状态，我们需要手动设置为非激活
      // 并发调用 toggle 接口将所有工具设为 active: false
      const togglePromises = toolsToInstall.map(tool => 
        mcpApi.toggleMCPTool(tool.name, false).catch(err => {
          console.warn(`Failed to disable tool ${tool.name}:`, err);
        })
      );
      await Promise.all(togglePromises);
      
      // 重新加载工具列表
      await loadMCPTools();
  
      setWishResultModal({ isOpen: false, result: null });
  
      const toolNames = toolsToInstall.map((t) => t.name).join('、');
      toast.success(`已新增 ${toolNames}`, {
        description: '工具已添加，默认未激活。请点击工具名称配置后再激活！',
        duration: 4000,
      });
    } catch (error) {
      toast.error('批量添加失败', {
        description: error instanceof Error ? error.message : '无法添加工具',
        duration: 4000,
      });
    }
  };

  // Toggle tool enabled state (with connection test when enabling)
  const handleToggleTool = async (toolId: string) => {
    const tool = mcpTools.find((t) => t.id === toolId);
    if (!tool) return;
  
    if (!tool.enabled) {
      const toastId = toast.loading(`正在测试 ${tool.name} 连接...`);
  
      try {
        const backendConfig = convertMCPToolToBackendConfig(tool);
        const result = await mcpApi.testMCPConnection(backendConfig);
        
        if (result.success) {
          await mcpApi.toggleMCPTool(tool.name, true);
          
          toast.success(`${tool.name} 连接测试成功`, {
            id: toastId,
            description: result.message,
            duration: 3000,
          });
  
          setMcpTools((tools) =>
            tools.map((t) => t.id === toolId ? { ...t, enabled: true } : t)
          );
        } else {
          toast.error(`${tool.name} 连接测试失败`, {
            id: toastId,
            description: result.message,
            duration: 4000,
          });
        }
      } catch (error) {
        toast.error(`${tool.name} 连接测试失败`, {
          id: toastId,
          description: error instanceof Error ? error.message : '请检查配置后重试',
          duration: 4000,
        });
      }
    } else {
      try {
        await mcpApi.toggleMCPTool(tool.name, false);
        setMcpTools((tools) =>
          tools.map((t) => t.id === toolId ? { ...t, enabled: false } : t)
        );
        toast.info(`已停用 ${tool.name}`, { duration: 2000 });
      } catch (error) {
        toast.error('操作失败', {
          description: error instanceof Error ? error.message : '无法停用工具',
          duration: 3000,
        });
      }
    }
  };

  // Open config modal
  const handleOpenConfig = (toolId: string) => {
    const tool = mcpTools.find((t) => t.id === toolId);
    if (tool) {
      setConfigModal({ isOpen: true, tool });
    }
  };

  // Save tool config
  const handleSaveConfig = async (toolId: string, description: string, config: string) => {
    try {
      const parsedConfig = JSON.parse(config);
      const tool = mcpTools.find(t => t.id === toolId);
      if (!tool) return;
  
      await mcpApi.installMCPTool({
        name: tool.name,
        description,
        type: tool.type || 'stdio',
        config: parsedConfig
      });
  
      setMcpTools((tools) =>
        tools.map((t) =>
          t.id === toolId ? { ...t, description, config: parsedConfig } : t
        )
      );
  
      toast.success('配置已保存', {
        description: '工具配置更新成功',
        duration: 3000,
      });
    } catch (error) {
      console.error('Save config failed:', error);
      toast.error('保存失败', {
        description: error instanceof Error ? error.message : '配置格式错误',
        duration: 4000,
      });
      throw error;
    }
  };

  // Delete tool
  const handleDeleteTool = (toolId: string) => {
    const tool = mcpTools.find((t) => t.id === toolId);
    if (!tool) return;

    // Close config modal first
    setConfigModal({ isOpen: false, tool: null });

    // Show confirmation
    setDeleteConfirm({ isOpen: true, toolId, toolName: tool.name });
  };

  // Confirm delete tool
  const handleConfirmDeleteTool = async () => {
    const { toolId, toolName } = deleteConfirm;
    if (!toolId || !toolName) return;
  
    try {
      await mcpApi.deleteMCPTool(toolName);
      setMcpTools((tools) => tools.filter((t) => t.id !== toolId));
      
      toast.success(`已删除工具"${toolName}"`, { duration: 3000 });
    } catch (error) {
      toast.error('删除失败', {
        description: error instanceof Error ? error.message : '无法删除工具',
        duration: 4000,
      });
    } finally {
      setDeleteConfirm({ isOpen: false, toolId: null, toolName: null });
    }
  };

  // Add new tool manually
  const handleAddTool = (name: string, description: string, introduction: string, config: string) => {
    try {
      const parsedConfig = JSON.parse(config);
      const newTool: MCPTool = {
        id: generateUniqueId('tool-'),
        name,
        description,
        icon: 'tool',
        iconBg: 'bg-purple-50 text-purple-500',
        introduction,
        config: parsedConfig,
        enabled: true,
        version: 'v1.0.0',
        author: '自定义'
      };

      setMcpTools([...mcpTools, newTool]);

      toast.success(`已添加工具"${name}"`, {
        description: '工具已成功添加到列表',
        duration: 3000,
      });
    } catch (error) {
      toast.error('配置格式错误', {
        description: '请检查 JSON 配置格式是否正确',
        duration: 4000,
      });
      throw error;
    }
  };

  // Handle Enter key in textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="h-screen flex text-slate-800 overflow-hidden relative">
      {/* Left Sidebar */}
      <LeftSidebar
        isOpen={isLeftSidebarOpen}
        onClose={() => setIsLeftSidebarOpen(false)}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewChat={handleNewChat}
        onSelectConversation={setActiveConversationId}
        onRenameConversation={handleRenameConversation}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Main Content - Dynamic width based on panels */}
      <div 
        className="fixed top-0 bottom-0 flex flex-col bg-slate-50/50 transition-all duration-300 z-10"
        style={{ 
          left: '0',
          right: isRightSidebarOpen ? `${rightSidebarWidth}px` : '0'
        }}
      >
        {/* Top Navigation */}
        <div className="h-16 border-b border-slate-200 bg-white flex items-center justify-between px-4 shadow-sm z-10">
          {/* Left Section: Menu */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsLeftSidebarOpen(true)}
              className="p-2 text-slate-500 hover:text-indigo-600 hover:bg-slate-100 rounded-lg transition-colors cursor-pointer"
            >
              <Menu className="w-5 h-5" />
            </button>
          </div>

          {/* Center Section: Title + Gitee link */}
          <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center space-x-3">
            <h1 className="bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent text-2xl tracking-tight">
              MCPChat
            </h1>
            <a
              href="https://gitee.com/ye_sheng0839/mcp-agent"
              target="_blank"
              rel="noopener noreferrer"
              className={`transition-all cursor-pointer hover:opacity-80 ${
                isRightSidebarOpen ? 'hidden' : 'hidden lg:block'
              }`}
            >
              <img 
                src="/gitee.svg" 
                alt="Gitee" 
                className="w-6 h-6"
              />
            </a>
          </div>

          {/* Right Section: MCP Toolbox */}
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsRightSidebarOpen(!isRightSidebarOpen)}
              className={`flex items-center justify-center space-x-2 px-3 py-1.5 rounded-lg transition-colors border text-sm cursor-pointer ${
                isRightSidebarOpen
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-indigo-50 text-indigo-600 border-indigo-100 hover:bg-indigo-100'
              }`}
            >
              <Wrench className="w-4 h-4" />
              <span className={isRightSidebarOpen ? 'hidden' : 'hidden sm:inline'}>MCP 工具箱</span>
            </button>
          </div>
        </div>

        {/* Chat Container */}
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth"
        >
          {!currentConversation || currentConversation.messages.length === 0 ? (
            <WelcomeScreen onSendMessage={handleSendMessage} />
          ) : (
            <>
              {currentConversation.messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}
            </>
          )}
          {isLoading && <LoadingIndicator />}
        </div>

        {/* Input Area */}
        <div className="p-6 pt-2 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent">
          <div className="relative bg-white rounded-xl shadow-lg border border-slate-200 transition-all duration-200 focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-500/20">
            <textarea
              ref={textareaRef}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              className="w-full p-4 pr-12 bg-transparent border-none focus:ring-0 focus:outline-none resize-none max-h-48 text-slate-700 placeholder-slate-400 rounded-xl"
              placeholder="输入消息... (例如: 帮我写一个 Python 脚本)"
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={isLoading || !userInput.trim()}
              className="absolute right-2 bottom-2 p-2 w-10 h-10 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors flex items-center justify-center shadow-md z-10 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Right Sidebar - Fixed position with dynamic width */}
      <RightSidebar
        isOpen={isRightSidebarOpen}
        onClose={() => setIsRightSidebarOpen(false)}
        tools={mcpTools}
        onToggleTool={handleToggleTool}
        onOpenConfig={handleOpenConfig}
        onProcessWish={handleProcessWish}
        onOpenGlobalSettings={() => setIsGlobalSettingsOpen(true)}
        onOpenAddTool={() => setAddToolModal({ isOpen: true, tool: null })}
        onWidthChange={setRightSidebarWidth}
      />

      {/* Modals */}
      <WishResultModal
        isOpen={wishResultModal.isOpen}
        onClose={() => setWishResultModal({ isOpen: false, result: null })}
        result={wishResultModal.result}
        onConfirm={handleConfirmAddTools}
      />

      <ConfigModal
        isOpen={configModal.isOpen}
        onClose={() => setConfigModal({ isOpen: false, tool: null })}
        tool={configModal.tool}
        onSave={handleSaveConfig}
        onDelete={handleDeleteTool}
        onTestConnection={handleTestConnection}
      />

      <AddToolModal
        isOpen={addToolModal.isOpen}
        onClose={() => setAddToolModal({ isOpen: false, tool: null })}
        onAdd={handleAddTool}
      />

      <GlobalSettingsModal
        isOpen={isGlobalSettingsOpen}
        onClose={() => setIsGlobalSettingsOpen(false)}
      />

      {/* Delete Confirmation */}
      <ConfirmDialog
        isOpen={deleteConfirm.isOpen}
        title="删除工具"
        message={`确定要删除工具"${deleteConfirm.toolName}"吗？此操作无法撤销。`}
        onConfirm={handleConfirmDeleteTool}
        onCancel={() => setDeleteConfirm({ isOpen: false, toolId: null, toolName: null })}
        variant="danger"
      />

      {/* Toast Notifications */}
      <Toaster position="top-right" richColors closeButton />
    </div>
  );
}