import { X, Plus, MessageSquare, Pencil, Trash2, Check, XIcon } from 'lucide-react';
import { toast } from 'sonner@2.0.3';
import { Conversation } from '../types';
import { useState } from 'react';
import { ConfirmDialog } from './ConfirmDialog';

interface LeftSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  conversations: Conversation[];
  activeConversationId: string | null;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onDeleteConversation: (id: string) => void;
}

export function LeftSidebar({
  isOpen,
  onClose,
  conversations,
  activeConversationId,
  onNewChat,
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation
}: LeftSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [deleteConfirm, setDeleteConfirm] = useState<{ id: string; title: string } | null>(null);

  const formatTime = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));
    
    if (days === 0) {
      const hours = Math.floor(diff / (1000 * 60 * 60));
      if (hours === 0) {
        return '刚刚';
      }
      return `${hours}小时前`;
    } else if (days === 1) {
      return '昨天';
    } else if (days < 7) {
      return `${days}天前`;
    } else {
      return new Date(timestamp).toLocaleDateString('zh-CN');
    }
  };

  const handleNewChat = () => {
    onNewChat();
    onClose();
  };

  const handleSelectConversation = (id: string) => {
    onSelectConversation(id);
    onClose();
  };

  const startEditing = (conv: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(conv.id);
    setEditingTitle(conv.title);
  };

  const saveEdit = (id: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (editingTitle.trim()) {
      onRenameConversation(id, editingTitle.trim());
    }
    setEditingId(null);
    setEditingTitle('');
  };

  const cancelEdit = (e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(null);
    setEditingTitle('');
  };

  const handleBlur = (id: string) => {
    // 使用 setTimeout 确保点击按钮的事件先执行
    setTimeout(() => {
      if (editingId === id) {
        saveEdit(id);
      }
    }, 150);
  };

  const handleDelete = (id: string, title: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirm({ id, title });
  };

  const confirmDelete = () => {
    if (deleteConfirm) {
      onDeleteConversation(deleteConfirm.id);
      toast.success('对话已删除', {
        description: `"${deleteConfirm.title}" 已从历史记录中删除`,
        duration: 3000,
      });
      setDeleteConfirm(null);
    }
  };

  const handleKeyDown = (id: string, e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      saveEdit(id);
    } else if (e.key === 'Escape') {
      setEditingId(null);
      setEditingTitle('');
    }
  };

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-40 transition-opacity"
        />
      )}

      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 w-64 bg-white shadow-2xl z-50 transform transition-transform duration-300 flex flex-col border-r border-slate-200 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="p-4 flex items-center justify-between border-b border-slate-100">
          <div className="flex items-center">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-md">
              <span className="text-sm">AI</span>
            </div>
            <span className="ml-3 text-lg text-slate-700">MCPChat</span>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          <button
            onClick={handleNewChat}
            className="w-full flex items-center p-3 rounded-lg bg-indigo-50 text-indigo-700 hover:bg-indigo-100 transition-colors border border-indigo-100 mb-4"
          >
            <Plus className="w-5 h-5 mr-3" />
            <span>新建对话</span>
          </button>

          <div className="text-xs text-slate-400 mb-2 px-2 uppercase tracking-wider">
            历史记录
          </div>

          {conversations.length === 0 ? (
            <div className="text-sm text-slate-400 text-center py-8">
              暂无对话记录
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group flex items-center p-3 rounded-lg border transition-all ${
                  conv.id === activeConversationId
                    ? 'bg-indigo-50 border-indigo-100'
                    : 'hover:bg-slate-50 border-transparent hover:border-slate-100'
                }`}
              >
                <MessageSquare className="w-4 h-4 text-slate-400 mr-3 flex-shrink-0" />
                {editingId === conv.id ? (
                  <div className="flex-1 flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="text"
                      value={editingTitle}
                      onChange={(e) => setEditingTitle(e.target.value)}
                      onKeyDown={(e) => handleKeyDown(conv.id, e)}
                      onBlur={() => handleBlur(conv.id)}
                      className="flex-1 px-2 py-1 text-sm border border-indigo-300 rounded focus:outline-none focus:border-indigo-500"
                      autoFocus
                    />
                    <button
                      onClick={(e) => saveEdit(conv.id, e)}
                      className="p-1 text-green-600 hover:bg-green-50 rounded"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                    <button
                      onClick={cancelEdit}
                      className="p-1 text-slate-400 hover:bg-slate-50 rounded"
                    >
                      <XIcon className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <>
                    <div
                      onClick={() => handleSelectConversation(conv.id)}
                      className="overflow-hidden flex-1 cursor-pointer"
                    >
                      <div className="truncate text-sm text-slate-700">
                        {conv.title}
                      </div>
                      <div className="truncate text-xs text-slate-400">
                        {formatTime(conv.updatedAt)}
                      </div>
                    </div>
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => startEditing(conv, e)}
                        className="p-1 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded"
                        title="重命名"
                      >
                        <Pencil className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={(e) => handleDelete(conv.id, conv.title, e)}
                        className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded"
                        title="删除"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))
          )}
        </div>


      </div>

      {/* Delete Confirmation Dialog */}
      {deleteConfirm && (
        <ConfirmDialog
          isOpen={true}
          title="删除对话"
          message={`确定要删除对话"${deleteConfirm.title}"吗？此操作无法撤销。`}
          confirmText="删除"
          cancelText="取消"
          onConfirm={confirmDelete}
          onCancel={() => setDeleteConfirm(null)}
          variant="danger"
        />
      )}
    </>
  );
}