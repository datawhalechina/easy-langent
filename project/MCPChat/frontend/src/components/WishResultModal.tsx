import { X, CheckCircle, Info, Check } from 'lucide-react';
import { WishAnalysisResult } from '../types';
import { useState } from 'react';

interface WishResultModalProps {
  isOpen: boolean;
  onClose: () => void;
  result: WishAnalysisResult | null;
  onConfirm: (selectedIndices: number[]) => void;
}

export function WishResultModal({ isOpen, onClose, result, onConfirm }: WishResultModalProps) {
  const [selectedTools, setSelectedTools] = useState<Set<number>>(new Set());

  if (!isOpen || !result) return null;

  const handleToggleTool = (index: number) => {
    const tool = result.suggestedTools[index];
    // 如果已安装，不允许选择
    if (tool.installed) return;

    const newSelected = new Set(selectedTools);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedTools(newSelected);
  };

  const handleConfirm = () => {
    onConfirm(Array.from(selectedTools));
    setSelectedTools(new Set()); // Reset selection
  };

  const handleClose = () => {
    setSelectedTools(new Set()); // Reset selection
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-2xl w-[500px] max-w-[90%] max-h-[80vh] overflow-hidden transform transition-all scale-100 flex flex-col">
        <div className="p-5 border-b border-slate-100 flex justify-between items-center bg-green-50/50 flex-shrink-0">
          <h3 className="text-slate-800 flex items-center">
            <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
            需求分析完成
          </h3>
          <button onClick={handleClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 flex-1 overflow-y-auto">
          <p className="text-sm text-slate-600 mb-5 leading-relaxed">
            为了实现您的需求，建议添加以下MCP工具：
          </p>

          {/* 工具列表 - 可滚动区域 */}
          <div className="space-y-3 mb-6">
            {result.suggestedTools.map((tool, index) => {
              const isInstalled = tool.installed;
              const isSelected = selectedTools.has(index);

              return (
                <div
                  key={index}
                  className={`flex items-start p-4 border rounded-xl transition-all ${
                    isInstalled
                      ? 'border-slate-200 bg-slate-50 opacity-75'
                      : isSelected
                      ? 'border-indigo-400 bg-indigo-50/50 shadow-sm'
                      : 'border-slate-200 hover:border-indigo-300 bg-white hover:bg-indigo-50/30'
                  } ${!isInstalled ? 'cursor-pointer' : 'cursor-not-allowed'}`}
                  onClick={() => !isInstalled && handleToggleTool(index)}
                >
                  {/* Checkbox or Installed Badge */}
                  <div className="flex-shrink-0 mt-0.5">
                    {isInstalled ? (
                      <div className="w-4 h-4 rounded bg-green-100 flex items-center justify-center">
                        <Check className="w-3 h-3 text-green-600" />
                      </div>
                    ) : (
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleToggleTool(index)}
                        className="w-4 h-4 text-indigo-600 bg-white border-2 border-slate-300 rounded focus:ring-2 focus:ring-indigo-500 focus:ring-offset-0 cursor-pointer checked:bg-indigo-600 checked:border-indigo-600 hover:border-indigo-400 transition-colors"
                        onClick={(e) => e.stopPropagation()}
                      />
                    )}
                  </div>

                  <div className="ml-3 flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <div className="text-slate-800 text-sm flex items-center">
                        {tool.name}
                        {isInstalled && (
                          <span className="ml-2 text-[10px] px-2 py-0.5 bg-green-100 text-green-700 rounded-full">
                            已安装
                          </span>
                        )}
                      </div>
                      {tool.type && (
                        <span className="text-[10px] px-2 py-0.5 bg-slate-100 text-slate-600 rounded-full flex-shrink-0">
                          {tool.type.toUpperCase()}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 mb-1.5">{tool.description}</p>
                    
                    {/* AI 推荐理由 */}
                    {tool.recommendReason && (
                      <div className="mt-2 text-[11px] text-indigo-600 bg-indigo-50/50 px-2 py-1.5 rounded-lg border border-indigo-100">
                        <span className="opacity-75">💡 推荐理由：</span>
                        {tool.recommendReason}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="bg-yellow-50 p-4 rounded-xl text-xs text-yellow-800 border border-yellow-100 flex items-start">
            <Info className="w-4 h-4 mr-2 mt-0.5 text-yellow-600 flex-shrink-0" />
            <div>
              点击确认后，选中的工具将安装到右侧列表（默认未激活）。请在列表中点击工具名称完成 API Key 等配置并测试后激活。
            </div>
          </div>
        </div>

        <div className="flex justify-end space-x-3 p-6 pt-0 border-t border-slate-100 flex-shrink-0">
          <button
            onClick={handleClose}
            className="px-5 py-2.5 text-sm text-slate-500 hover:text-slate-700"
          >
            取消
          </button>
          <button
            onClick={handleConfirm}
            disabled={selectedTools.size === 0}
            className="px-6 py-2.5 bg-slate-900 hover:bg-slate-800 text-white text-sm rounded-lg shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-slate-900"
          >
            确认添加 {selectedTools.size > 0 && `(${selectedTools.size})`}
          </button>
        </div>
      </div>
    </div>
  );
}