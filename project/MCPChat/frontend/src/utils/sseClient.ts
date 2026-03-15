// SSE 流式数据处理工具

export interface SSEEvent {
  type: 'token' | 'tool_start' | 'tool_end' | 'finish' | 'error';
  data: any;
}

/**
 * 解析 SSE 格式的数据
 * 每一行以 "data: " 开头，后面是 JSON 对象
 */
export function parseSSEData(text: string): SSEEvent[] {
  const events: SSEEvent[] = [];
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      try {
        const jsonStr = line.substring(6).trim(); // 移除 "data: " 前缀
        if (jsonStr) {
          const event = JSON.parse(jsonStr);
          events.push(event);
        }
      } catch (error) {
        console.error('Failed to parse SSE event:', line, error);
      }
    }
  }

  return events;
}

/**
 * 发起 SSE 流式请求
 */
export async function sendChatStream(
  message: string,
  sessionId: string,
  onEvent: (event: SSEEvent) => void,
  onError?: (error: Error) => void
): Promise<void> {
  try {
    const response = await fetch('http://localhost:8002/chat_stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: message,  // 后端期望的字段是 query
        session_id: sessionId,  // 后端期望的字段是 session_id
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    // 读取流
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();

      if (done) {
        // 处理剩余的 buffer
        if (buffer.trim()) {
          const events = parseSSEData(buffer);
          events.forEach(onEvent);
        }
        break;
      }

      // 解码数据块
      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;

      // 按行分割处理
      const lines = buffer.split('\n');
      // 保留最后一行（可能不完整）
      buffer = lines.pop() || '';

      // 处理完整的行
      const completeText = lines.join('\n');
      if (completeText) {
        const events = parseSSEData(completeText);
        events.forEach(onEvent);
      }
    }
  } catch (error) {
    console.error('SSE stream error:', error);
    if (onError) {
      onError(error as Error);
    }
  }
}