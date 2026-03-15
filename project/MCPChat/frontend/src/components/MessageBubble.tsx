import { Message, TextBlock, ToolCallBlock } from '../types';
import { CollapsibleSection } from './CollapsibleSection';
import { ToolCard } from './ToolCard';
import { renderMarkdown } from '../utils/markdown';
import { useEffect, useRef, useState } from 'react';

interface MessageBubbleProps {
  message: Message;
  onToggleToolExpand?: (messageId: string, blockId: string) => void;
}

export function MessageBubble({ message, onToggleToolExpand }: MessageBubbleProps) {
  const contentRef = useRef<HTMLDivElement>(null);

  // 流式打字效果 - 支持新旧格式
  useEffect(() => {
    if (contentRef.current) {
      // 旧格式
      if (message.chunks) {
        const lastChunk = message.chunks[message.chunks.length - 1];
        if (lastChunk?.isStreaming && lastChunk.type === 'content') {
          contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
      }
      // 新格式
      if (message.blocks && message.isStreaming) {
        contentRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  }, [message.chunks, message.blocks, message.isStreaming]);

  // 用户消息
  if (message.role === 'user') {
    return (
      <div className="flex items-end mb-6 fade-in justify-end">
        <div className="bg-indigo-600 text-white p-3 rounded-2xl rounded-tr-none shadow-md max-w-[80%] text-sm">
          {message.content}
        </div>
      </div>
    );
  }

  // AI消息 - 新格式（使用 blocks）
  if (message.blocks) {
    return (
      <div className="flex items-start fade-in mb-8">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex-shrink-0 flex items-center justify-center text-white text-xs mr-3 mt-1 shadow-md">
          AI
        </div>
        <div className="flex-1 min-w-0">
          {message.blocks.map((block) => {
            // 文本块
            if (block.type === 'text') {
              const textBlock = block as TextBlock;
              return (
                <div
                  key={textBlock.id}
                  ref={contentRef}
                  className="prose prose-sm prose-slate max-w-none text-slate-700 bg-white p-5 rounded-xl rounded-tl-none shadow-sm border border-slate-100 mb-3"
                >
                  <div 
                    dangerouslySetInnerHTML={{ 
                      __html: renderMarkdown(textBlock.content) 
                    }} 
                  />
                  {message.isStreaming && block === message.blocks[message.blocks.length - 1] && (
                    <span className="inline-block w-2 h-4 bg-indigo-600 ml-1 animate-pulse"></span>
                  )}
                </div>
              );
            }

            // 工具调用块
            if (block.type === 'tool_call') {
              const toolBlock = block as ToolCallBlock;
              return (
                <ToolCard
                  key={toolBlock.id}
                  toolName={toolBlock.toolName}
                  input={toolBlock.input}
                  output={toolBlock.output}
                  status={toolBlock.status}
                  isExpanded={toolBlock.isExpanded}
                  onToggleExpand={
                    onToggleToolExpand
                      ? () => onToggleToolExpand(message.id, toolBlock.id)
                      : undefined
                  }
                />
              );
            }

            return null;
          })}
        </div>
      </div>
    );
  }

  // AI消息 - 旧格式（使用 chunks，保留兼容性）
  return (
    <div className="flex items-start fade-in mb-8">
      <div className="w-8 h-8 rounded-lg bg-indigo-600 flex-shrink-0 flex items-center justify-center text-white text-xs mr-3 mt-1 shadow-md">
        AI
      </div>
      <div className="flex-1 min-w-0">
        {message.chunks && message.chunks.map((chunk) => {
          // 思考过程
          if (chunk.type === 'thought') {
            return (
              <CollapsibleSection
                key={chunk.id}
                type="thought"
                content={chunk.content}
                defaultCollapsed={true}
              />
            );
          }

          // 工具调用
          if (chunk.type === 'tool_call') {
            return (
              <CollapsibleSection
                key={chunk.id}
                type="tool_call"
                content={chunk.content}
                defaultCollapsed={true}
              />
            );
          }

          // 实际回复内容
          if (chunk.type === 'content') {
            return (
              <div
                key={chunk.id}
                ref={contentRef}
                className="prose prose-sm prose-slate max-w-none text-slate-700 bg-white p-5 rounded-xl rounded-tl-none shadow-sm border border-slate-100 mb-3"
              >
                <div 
                  dangerouslySetInnerHTML={{ 
                    __html: renderMarkdown(chunk.content) 
                  }} 
                />
                {chunk.isStreaming && (
                  <span className="inline-block w-2 h-4 bg-indigo-600 ml-1 animate-pulse"></span>
                )}
              </div>
            );
          }

          return null;
        })}
      </div>
    </div>
  );
}