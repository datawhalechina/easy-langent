import { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { Send, Bot, User } from 'lucide-react';
import { API_ENDPOINTS } from '../config/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const initialMessages: Message[] = [
  {
    id: '1',
    type: 'assistant',
    content: '您好！我是数据分析助手，可以帮您分析上传的数据集。请在右侧上传CSV文件，然后选择变量进行分析，我会为您提供详细的分析结果和建议。',
    timestamp: new Date()
  }
];

interface ChatInterfaceProps {
  clearTrigger: number;
  onImageGenerated?: (imageUrl: string) => void;
}

export function ChatInterface({ clearTrigger, onImageGenerated }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (clearTrigger > 0) {
      setMessages(initialMessages);
      setInputValue('');
      setIsLoading(false);
    }
  }, [clearTrigger]);

  const buildMessageHistory = () => {
    return messages
      .filter(msg => 
        msg.id !== '1' && 
        msg.content.trim() !== '' 
      )
      .map(msg => ({
        type: msg.type === 'user' ? 'human' : 'ai',
        content: msg.content
      }));
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      type: 'assistant',
      content: '',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, assistantMessage]);

    try {
      const messageHistory = buildMessageHistory();
      const requestBody = {
        input: {
          messages: [
            ...messageHistory,
            { type: 'human', content: userMessage.content }
          ]
        }
      };

      const response = await fetch(API_ENDPOINTS.AGENT_STREAM, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let buffer = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const decodedChunk = decoder.decode(value, { stream: true });
          buffer += decodedChunk;
          
          const blocks = buffer.split('\n\n');
          if (!buffer.endsWith('\n\n')) {
            buffer = blocks.pop() || '';
          } else {
            buffer = '';
          }
          
          for (const block of blocks) {
            if (!block.trim()) continue;

            const lines = block.split('\n');
            let eventType = '';
            let dataContent = '';

            for (const line of lines) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim();
              } else if (line.startsWith('data:')) {
                dataContent = line.slice(5).trim();
              }
            }

            if (eventType === 'data' && dataContent) {
              try {
                const parsed = JSON.parse(dataContent);
                let messagesArray = parsed.model?.messages || parsed.messages || parsed.output?.messages;
                
                if (messagesArray && Array.isArray(messagesArray) && messagesArray.length > 0) {
                  const lastMessage = messagesArray[messagesArray.length - 1];
                  
                  if (lastMessage.content && typeof lastMessage.content === 'string' && lastMessage.content.trim()) {
                    accumulatedContent = lastMessage.content;
                    
                    const imageMatch = accumulatedContent.match(/IMAGE_GENERATED:\s*(\S+)/);
                    if (imageMatch && onImageGenerated) {
                      const filename = imageMatch[1];
                      const imageUrl = `http://localhost:8002/static/images/${filename}`;
                      onImageGenerated(imageUrl);
                    }
                    
                    setMessages(prev => prev.map(msg => 
                      msg.id === assistantMessageId 
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    ));
                  } else if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
                    if (!accumulatedContent) {
                      setMessages(prev => prev.map(msg => 
                        msg.id === assistantMessageId 
                          ? { ...msg, content: '正在分析中...' }
                          : msg
                      ));
                    }
                  }
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            } else if (eventType === 'end') {
              break;
            }
          }
        }
        
        if (buffer.trim()) {
          const dataEventRegex = /event:\s*data\s*\n\s*data:\s*({[\s\S]*?})(?=\s*\n\s*event:|\s*$)/g;
          let match;
          
          while ((match = dataEventRegex.exec(buffer)) !== null) {
            const dataContent = match[1];
            
            try {
              const parsed = JSON.parse(dataContent);
              const messagesArray = parsed.model?.messages || parsed.messages || parsed.output?.messages;
              
              if (messagesArray && Array.isArray(messagesArray) && messagesArray.length > 0) {
                const lastMessage = messagesArray[messagesArray.length - 1];
                
                if (lastMessage.type === 'ai' && lastMessage.content && typeof lastMessage.content === 'string' && lastMessage.content.trim()) {
                  accumulatedContent = lastMessage.content;
                  
                  const imageMatch = accumulatedContent.match(/IMAGE_GENERATED:\s*(\S+)/);
                  if (imageMatch && onImageGenerated) {
                    const filename = imageMatch[1];
                    const imageUrl = `http://localhost:8002/static/images/${filename}`;
                    onImageGenerated(imageUrl);
                  }
                  
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId 
                      ? { ...msg, content: accumulatedContent }
                      : msg
                  ));
                }
              }
            } catch (e) {
              console.error('Error parsing remaining buffer:', e);
            }
          }
        }
      }

      setIsLoading(false);
    } catch (error) {
      console.error('Error calling agent:', error);
      
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? { ...msg, content: '抱歉，调用AI助手时出现错误。请检查后端服务是否正常运行。' }
          : msg
      ));
      
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-6 custom-scrollbar">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.type === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.type === 'assistant' && (
                <div className="w-10 h-10 bg-primary border-2 border-black flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              
              <div
                className={`max-w-[80%] p-4 whitespace-pre-wrap border-2 border-black ${
                  message.type === 'user'
                    ? 'bg-secondary text-black ml-auto'
                    : 'bg-white text-black'
                }`}
              >
                {message.content}
              </div>
              
              {message.type === 'user' && (
                <div className="w-10 h-10 bg-[#FFD166] border-2 border-black flex items-center justify-center flex-shrink-0 overflow-hidden">
                  <img 
                    src="https://api.dicebear.com/7.x/pixel-art/svg?seed=user&backgroundColor=FFD166" 
                    alt="User" 
                    className="w-full h-full"
                  />
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-10 h-10 bg-primary border-2 border-black flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white border-2 border-black p-4">
                <div className="flex space-x-2">
                  <div className="w-3 h-3 bg-primary border border-black"></div>
                  <div className="w-3 h-3 bg-secondary border border-black" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-3 h-3 bg-accent border border-black" style={{animationDelay: '0.2s'}}></div>
                </div>
              </div>
            </div>
          )}
        </div>
        </ScrollArea>
      </div>

      <div className="p-6 border-t-2 border-black">
        <div className="flex gap-3">
          <Textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="请输入您的问题..."
            className="flex-1 min-h-[50px] max-h-[120px] resize-none font-medium"
            disabled={isLoading}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="self-end"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}