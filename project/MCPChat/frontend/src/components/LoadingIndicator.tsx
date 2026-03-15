export function LoadingIndicator() {
  return (
    <div className="flex items-center fade-in mb-6 ml-2">
      <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white text-xs mr-3 shadow-md">
        AI
      </div>
      <div className="flex space-x-1">
        <div
          className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
          style={{ animationDelay: '0s' }}
        ></div>
        <div
          className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
          style={{ animationDelay: '0.1s' }}
        ></div>
        <div
          className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"
          style={{ animationDelay: '0.2s' }}
        ></div>
      </div>
    </div>
  );
}
