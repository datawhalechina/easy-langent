import { useState } from 'react';
import { ChevronDown, ChevronRight, Brain, Wrench } from 'lucide-react';
import { highlightPython } from '../utils/markdown';

interface CollapsibleSectionProps {
  type: 'thought' | 'tool_call';
  content: string;
  defaultCollapsed?: boolean;
}

export function CollapsibleSection({ type, content, defaultCollapsed = true }: CollapsibleSectionProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);

  const config = {
    thought: {
      icon: Brain,
      title: '💭 思考过程',
      bgColor: 'bg-indigo-50',
      borderColor: 'border-indigo-200',
      textColor: 'text-indigo-700',
      iconColor: 'text-indigo-500'
    },
    tool_call: {
      icon: Wrench,
      title: '🔧 工具调用',
      bgColor: 'bg-amber-50',
      borderColor: 'border-amber-200',
      textColor: 'text-amber-700',
      iconColor: 'text-amber-500'
    }
  };

  const { icon: Icon, title, bgColor, borderColor, textColor, iconColor } = config[type];

  return (
    <div className={`${bgColor} border ${borderColor} rounded-lg overflow-hidden mb-3`}>
      {/* Header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className={`w-full flex items-center justify-between px-4 py-2.5 hover:bg-opacity-80 transition-all ${textColor}`}
      >
        <div className="flex items-center space-x-2">
          <Icon className={`w-4 h-4 ${iconColor}`} />
          <span className="text-sm font-medium">{title}</span>
        </div>
        {isCollapsed ? (
          <ChevronRight className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
      </button>

      {/* Content */}
      {!isCollapsed && (
        <div className="px-4 pb-3 pt-1">
          {type === 'thought' ? (
            <div className="text-sm text-slate-700 whitespace-pre-wrap font-mono bg-white p-3 rounded border border-slate-200">
              {content}
            </div>
          ) : (
            <div className="code-block-tool my-0">
              <pre className="!m-0 !p-3 overflow-x-auto bg-[#0f172a] rounded">
                <code 
                  className="font-mono text-sm leading-relaxed block"
                  dangerouslySetInnerHTML={{ __html: highlightPython(content) }}
                />
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
