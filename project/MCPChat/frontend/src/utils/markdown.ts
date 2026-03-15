// Markdown rendering utilities

const escapeHtml = (text: string): string => {
  const map: Record<string, string> = {
    '&': '&',
    '<': '<',
    '>': '>',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
};

export const renderMarkdown = (text: string): string => {
  let html = text;
  
  // Images - ![alt](url)
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, url) => {
    return `<img src="${url}" alt="${escapeHtml(alt)}" class="markdown-image rounded-lg shadow-md my-4 max-w-full" loading="lazy" />`;
  });
  
  // Links - [text](url)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, text, url) => {
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="text-indigo-600 hover:text-indigo-800 underline">${escapeHtml(text)}</a>`;
  });
  
  // Code blocks - simple rendering without syntax highlighting
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
    const language = lang || '';
    const escapedCode = escapeHtml(code.trim());
    return `<div class="code-block my-4">
  ${language ? `<div class="code-lang">${language}</div>` : ''}
  <pre><code>${escapedCode}</code></pre>
</div>`;
  });
  
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
  
  // Bold
  html = html.replace(/\*\*([^\*]+)\*\*/g, '<strong>$1</strong>');
  
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  
  // Lists
  html = html.replace(/^\* (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*?<\/li>\n?)+/gs, '<ul>$&</ul>');
  
  // Paragraphs - split by double newlines and wrap in <p> tags
  const paragraphs = html.split(/\n\n+/).filter(p => p.trim());
  html = paragraphs.map(p => {
    // Don't wrap if already has block-level tags
    if (p.match(/^<(h[123]|ul|div|img)/)) {
      return p;
    }
    return `<p>${p.replace(/\n/g, '<br/>')}</p>`;
  }).join('\n');
  
  return html;
};

// Python语法高亮
export const highlightPython = (code: string): string => {
  let highlighted = escapeHtml(code);
  
  // Keywords
  const keywords = /\b(from|import|def|class|if|else|elif|for|while|return|try|except|with|as|in|is|not|and|or|None|True|False|print|async|await|yield|lambda|pass|break|continue|raise|finally|assert|del|global|nonlocal)\b/g;
  highlighted = highlighted.replace(keywords, '<span class="text-purple-400 font-semibold">$1</span>');
  
  // Strings (both single and double quotes)
  highlighted = highlighted.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="text-green-400">$&</span>');
  
  // Comments
  highlighted = highlighted.replace(/(#.*$)/gm, '<span class="text-gray-500 italic">$&</span>');
  
  // Function/class names in definitions
  highlighted = highlighted.replace(/\b(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)/g, '$1 <span class="text-yellow-300 font-semibold">$2</span>');
  
  // Numbers
  highlighted = highlighted.replace(/\b(\d+\.?\d*)\b/g, '<span class="text-blue-400">$1</span>');
  
  // Decorators
  highlighted = highlighted.replace(/(@[a-zA-Z_][a-zA-Z0-9_]*)/g, '<span class="text-pink-400">$1</span>');
  
  return highlighted;
};