import { CheckCircle, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ToolCardProps {
  toolName: string;
  input: any;
  output?: any;
  status: 'running' | 'completed';
  onToggleExpand?: () => void;
  isExpanded?: boolean;
}

export function ToolCard({ 
  toolName, 
  input, 
  output, 
  status,
  onToggleExpand,
  isExpanded = false
}: ToolCardProps) {
  const [localExpanded, setLocalExpanded] = useState(false);
  const expanded = onToggleExpand !== undefined ? isExpanded : localExpanded;
  
  const handleToggle = () => {
    if (onToggleExpand) {
      onToggleExpand();
    } else {
      setLocalExpanded(!localExpanded);
    }
  };

  if (status === 'running') {
    return (
      <div className="my-2 p-3 bg-blue-50 border border-blue-100 rounded-lg flex items-center gap-3">
        <Loader2 className="w-4 h-4 text-blue-600 animate-spin flex-shrink-0" />
        <div className="flex-1">
          <div className="text-sm text-blue-700">
            正在使用 <span className="font-medium">{toolName}</span>...
          </div>
          {input && Object.keys(input).length > 0 && (
            <div className="text-xs text-blue-600 mt-1">
              参数: {JSON.stringify(input)}
            </div>
          )}
        </div>
      </div>
    );
  }

  // completed 状态
  return (
    <div className="my-2">
      <div
        onClick={handleToggle}
        className="p-3 bg-gray-50 border border-gray-200 rounded-lg flex items-center gap-3 cursor-pointer hover:bg-gray-100 transition-colors"
      >
        <CheckCircle className="w-4 h-4 text-green-600 flex-shrink-0" />
        <div className="flex-1">
          <div className="text-sm text-gray-700">
            已完成 <span className="font-medium">{toolName}</span>
          </div>
          {input && Object.keys(input).length > 0 && (
            <div className="text-xs text-gray-500 mt-1">
              参数: {JSON.stringify(input)}
            </div>
          )}
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />
        )}
      </div>

      {/* 展开的详情区域 */}
      {expanded && output && (
        <div className="mt-2 rounded-lg overflow-hidden border border-gray-200">
          <div className="bg-gray-800 px-3 py-2 text-xs text-gray-300 flex items-center justify-between">
            <span>工具返回结果</span>
            <span className="text-gray-500">JSON</span>
          </div>
          <div className="max-h-64 overflow-auto">
            <SyntaxHighlighter
              language="json"
              style={vscDarkPlus}
              customStyle={{
                margin: 0,
                padding: '12px',
                fontSize: '12px',
                lineHeight: '1.5',
                background: '#1e1e1e'
              }}
              showLineNumbers={false}
            >
              {typeof output === 'string' ? output : JSON.stringify(output, null, 2)}
            </SyntaxHighlighter>
          </div>
        </div>
      )}
    </div>
  );
}
