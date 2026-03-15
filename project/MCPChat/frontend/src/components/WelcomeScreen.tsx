import { Bot } from 'lucide-react';

interface WelcomeScreenProps {
  onSendMessage: (message: string) => void;
}

export function WelcomeScreen({ onSendMessage }: WelcomeScreenProps) {
  const prompts = [
    '如何用 Python 调用 DeepSeek API？',
    '分析一下最近的 AI 行业动态'
  ];

  return (
    <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-60">
      <div className="w-16 h-16 bg-white rounded-2xl shadow-md flex items-center justify-center text-3xl text-indigo-500 mb-2">
        <Bot className="w-10 h-10" />
      </div>
      <h2 className="text-xl text-slate-700">你好我是 MCPChat</h2>
      <p className="text-slate-500 max-w-md text-sm">
        我可以协助你完成各种任务。
        <br />
        点击右上角 <strong>"MCP 工具箱"</strong> 来组装更多强大的工具。
      </p>
      <div className="flex space-x-2 mt-4">
        {prompts.map((prompt, index) => (
          <button
            key={index}
            onClick={() => onSendMessage(prompt)}
            className="px-3 py-1.5 bg-white border border-slate-200 rounded-full text-xs text-slate-600 hover:border-indigo-300 hover:text-indigo-600 transition-all shadow-sm"
          >
            Prompt: {prompt.split('？')[0].split('一下')[0]}
          </button>
        ))}
      </div>
    </div>
  );
}