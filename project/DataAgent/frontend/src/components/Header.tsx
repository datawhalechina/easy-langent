import { Button } from './ui/button';

interface HeaderProps {
  onChatClear: () => void;
}

export function Header({ onChatClear }: HeaderProps) {
  return (
    <header className="flex items-center justify-between p-4 bg-secondary border-b-2 border-black relative z-20">
      <div className="flex items-center">
        <Button 
          onClick={onChatClear}
          variant="outline"
          className="bg-white hover:bg-accent"
        >
          清除对话
        </Button>
      </div>
      
      <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-3">
        <h1 className="text-2xl font-bold text-black">
          Data Agent
        </h1>
        <span className="text-black text-sm font-medium">
          by 烨笙
        </span>
      </div>
      
      <div className="flex items-center">
        <a 
          href="https://gitee.com/ye_sheng0839/data-agent" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center justify-center w-10 h-10"
        >
          <img 
            src="/gitee.svg" 
            alt="Gitee" 
            className="w-full h-full"
          />
        </a>
      </div>
    </header>
  );
}