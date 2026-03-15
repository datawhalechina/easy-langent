import { X, Wand2, Settings, Zap, Plus } from 'lucide-react';
import { toast } from 'sonner@2.0.3';
import { MCPTool } from '../types';
import { useState } from 'react';
import { Resizable } from 're-resizable';

interface RightSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  tools: MCPTool[];
  onToggleTool: (toolId: string) => void;
  onOpenConfig: (toolId: string) => void;
  onProcessWish: (wish: string) => void;
  onOpenGlobalSettings: () => void;
  onOpenAddTool: () => void;
  onWidthChange?: (width: number) => void;
}

export function RightSidebar({
  isOpen,
  onClose,
  tools,
  onToggleTool,
  onOpenConfig,
  onProcessWish,
  onOpenGlobalSettings,
  onOpenAddTool,
  onWidthChange
}: RightSidebarProps) {
  const [wishInput, setWishInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [width, setWidth] = useState(384); // 96 * 4 = 384px (w-96)

  const handleProcessWish = async () => {
    if (!wishInput.trim()) {
      toast.warning('请输入您的需求');
      return;
    }

    setIsProcessing(true);
    await onProcessWish(wishInput);
    setIsProcessing(false);
    setWishInput('');
  };

  const getIconComponent = (iconName: string) => {
    // Map icon names to components or return a default
    return <span className="text-lg">{iconName.substring(0, 2).toUpperCase()}</span>;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed right-0 top-0 bottom-0 z-30">
      <Resizable
        size={{ width, height: '100vh' }}
        onResizeStop={(e, direction, ref, d) => {
          const newWidth = width + d.width;
          setWidth(newWidth);
          onWidthChange?.(newWidth);
        }}
        minWidth={300}
        maxWidth={window.innerWidth * 0.5}
        enable={{ left: true }}
        handleStyles={{
          left: {
            width: '8px',
            left: '-4px',
            cursor: 'ew-resize',
          }
        }}
        className="h-full bg-white shadow-2xl flex flex-col border-l border-slate-200"
      >
        {/* 上部：许愿池 */}
        <div className="h-auto flex flex-col border-b border-slate-200 bg-gradient-to-b from-indigo-50/50 to-white flex-shrink-0">
          <div className="p-5 pb-2 flex justify-between items-start">
            <div>
              <h3 className="text-slate-700 flex items-center">
                <Wand2 className="w-5 h-5 text-indigo-500 mr-2" />
                Agent 能力扩展
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                AI 自动寻找并配置 MCP 工具。
              </p>
            </div>
            <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="px-5 pb-5 flex-1 flex flex-col justify-center">
            <div className="relative group">
              <textarea
                value={wishInput}
                onChange={(e) => setWishInput(e.target.value)}
                className="w-full h-24 p-3 text-sm border border-indigo-100 rounded-xl focus:ring-2 focus:ring-indigo-200 focus:border-indigo-400 transition-all bg-white shadow-sm resize-none focus:outline-none"
                placeholder="例如：我想让 Agent 能够操作浏览器，或者读取我的 Notion 笔记..."
              />
              <div className="absolute bottom-2 right-2 text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
                支持自然语言
              </div>
            </div>
            <button
              onClick={handleProcessWish}
              disabled={isProcessing}
              className="mt-3 w-full py-2.5 bg-white border border-indigo-200 text-indigo-600 text-sm rounded-lg hover:bg-indigo-50 hover:text-indigo-700 transition-colors shadow-sm flex items-center justify-center disabled:opacity-50"
            >
              {isProcessing ? (
                <>
                  <div className="w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin mr-2" />
                  正在分析需求...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  智能分析并添加工具
                </>
              )}
            </button>
          </div>
        </div>

        {/* 下部：MCP 工具列表 */}
        <div className="flex-1 flex flex-col overflow-hidden bg-white">
          <div className="p-4 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
            <h3 className="text-slate-700 text-sm">MCP工具列表</h3>
            <button
              onClick={onOpenGlobalSettings}
              className="text-slate-400 hover:text-indigo-600 p-1.5 rounded-md hover:bg-white transition-colors"
              title="全局 MCP 设置"
            >
              <Settings className="w-4 h-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-2">
            {tools.map((tool) => (
              <div
                key={tool.id}
                className="group p-3 rounded-xl border border-slate-200 bg-white hover:border-indigo-300 transition-all shadow-sm"
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1 min-w-0">
                    <div
                      onClick={() => onOpenConfig(tool.id)}
                      className="text-slate-700 text-sm cursor-pointer hover:text-indigo-600 hover:underline decoration-dashed truncate"
                    >
                      {tool.name}
                    </div>
                    <div className="text-xs text-slate-400 mt-0.5 truncate">
                      {tool.description}
                    </div>
                  </div>
                  {/* Toggle Switch */}
                  <label className="relative inline-flex items-center cursor-pointer mt-1 ml-2 flex-shrink-0">
                    <input
                      type="checkbox"
                      checked={tool.enabled}
                      onChange={() => onToggleTool(tool.id)}
                      className="sr-only peer"
                    />
                    <div className="w-9 h-5 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-600"></div>
                  </label>
                </div>
              </div>
            ))}
            
            {/* Add New Tool Button */}
            <button
              onClick={onOpenAddTool}
              className="w-full p-3 rounded-xl border-2 border-dashed border-slate-300 bg-white hover:border-indigo-400 hover:bg-indigo-50/50 transition-all shadow-sm group cursor-pointer"
            >
              <div className="flex items-center justify-center text-slate-500 group-hover:text-indigo-600 transition-colors">
                <Plus className="w-4 h-4 mr-2" />
                <span className="text-sm">添加新工具</span>
              </div>
            </button>
          </div>
        </div>
      </Resizable>
    </div>
  );
}