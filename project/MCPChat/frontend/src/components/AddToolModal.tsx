import { X, Plus, Check, Plug } from 'lucide-react';
import { useState } from 'react';
import { MCPConfig } from '../types';
import { toast } from 'sonner@2.0.3';

interface AddToolModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (name: string, description: string, introduction: string, config: string) => void;
}

export function AddToolModal({ isOpen, onClose, onAdd }: AddToolModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [config, setConfig] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [isAdded, setIsAdded] = useState(false);
  const [isTested, setIsTested] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  if (!isOpen) return null;

  const defaultConfigTemplate = JSON.stringify(
    {
      mcpServers: {
        "your-tool-name": {
          command: "npx",
          args: ["-y", "@your/mcp-server"],
          env: {
            API_KEY: "your_api_key_here"
          }
        }
      }
    },
    null,
    2
  );

  const handleTestConnection = async () => {
    if (!config.trim() && !defaultConfigTemplate) {
      toast.warning('请先填写配置信息');
      return;
    }

    setIsTesting(true);
    
    // Simulate connection test
    await new Promise((resolve) => setTimeout(resolve, 1500));
    
    setIsTesting(false);
    setIsTested(true);
    
    toast.success('连接测试成功', {
      description: '耗时 120ms',
      duration: 3000,
    });
  };

  const handleAdd = async () => {
    if (!name.trim()) {
      return;
    }

    if (!isTested) {
      toast.warning('请先测试连接');
      return;
    }

    setIsAdding(true);
    await new Promise((resolve) => setTimeout(resolve, 800));
    onAdd(name, description, description, config || defaultConfigTemplate);
    setIsAdding(false);
    setIsAdded(true);
    
    setTimeout(() => {
      setIsAdded(false);
      setName('');
      setDescription('');
      setConfig('');
      setIsTested(false);
      onClose();
    }, 800);
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-2xl w-[520px] max-w-[90%] max-h-[85vh] overflow-hidden flex flex-col">
        <div className="p-5 border-b border-slate-100 flex justify-between items-center bg-gradient-to-r from-indigo-50/50 to-purple-50/50 flex-shrink-0">
          <div className="flex items-center">
            <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center mr-3">
              <Plus className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h3 className="text-slate-800">添加自定义 MCP 工具</h3>
              <p className="text-xs text-slate-500">手动配置您的 MCP Server</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-4 flex-1 overflow-y-auto">
          {/* 工具名称 */}
          <div>
            <label className="block text-xs text-slate-700 mb-1.5 uppercase tracking-wide">
              工具名称 *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2.5 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all"
              placeholder="例如: Puppeteer MCP"
            />
          </div>

          {/* 功能介绍 */}
          <div>
            <label className="block text-xs text-slate-700 mb-1.5 uppercase tracking-wide">
              功能介绍
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full h-24 px-3 py-2.5 border border-slate-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all resize-none"
              placeholder="详细介绍这个工具的功能、使用场景等..."
            />
          </div>

          {/* MCP 配置 */}
          <div>
            <label className="block text-xs text-slate-700 mb-1.5 uppercase tracking-wide">
              MCP 配置 (JSON)
            </label>
            <textarea
              value={config || defaultConfigTemplate}
              onChange={(e) => {
                setConfig(e.target.value);
                setIsTested(false); // Reset test status when config changes
              }}
              className="w-full h-48 border border-slate-200 rounded-lg text-xs font-mono p-3 bg-slate-50 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all resize-none text-slate-600"
              placeholder={defaultConfigTemplate}
            />
            <p className="text-[10px] text-slate-400 mt-1">
              配置 MCP Server 的启动命令、参数和环境变量
            </p>
          </div>
        </div>

        <div className="p-6 pt-4 border-t border-slate-100 flex justify-between items-center flex-shrink-0 bg-slate-50/50">
          {/* Test Connection Button */}
          <button
            onClick={handleTestConnection}
            disabled={isTesting}
            className={`text-xs flex items-center px-3 py-2 rounded transition-colors ${
              isTested 
                ? 'text-green-600 bg-green-50' 
                : 'text-green-600 hover:text-green-700 hover:bg-green-50'
            }`}
          >
            {isTesting ? (
              <>
                <div className="w-3 h-3 border-2 border-green-600 border-t-transparent rounded-full animate-spin mr-1.5" />
                测试中...
              </>
            ) : isTested ? (
              <>
                <Check className="w-3 h-3 mr-1.5" />
                已测试
              </>
            ) : (
              <>
                <Plug className="w-3 h-3 mr-1.5" />
                测试连接
              </>
            )}
          </button>

          {/* Action Buttons */}
          <div className="flex space-x-3">
            <button
              onClick={onClose}
              className="px-5 py-2.5 text-sm text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
            >
              取消
            </button>
            <button
              onClick={handleAdd}
              disabled={isAdding || !name.trim() || !isTested}
              className="px-6 py-2.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 transition-colors shadow-md shadow-indigo-200 flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isAdded ? (
                <>
                  <Check className="w-4 h-4 mr-1" />
                  已添加
                </>
              ) : isAdding ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  添加中...
                </>
              ) : (
                <>
                  <Plus className="w-4 h-4 mr-1" />
                  添加工具
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}