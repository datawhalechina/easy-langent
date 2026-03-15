import { AlertTriangle, Trash2, Info, AlertCircle } from 'lucide-react';

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  onConfirm: () => void;
  onCancel: () => void;
  variant?: 'danger' | 'warning' | 'info';
}

export function ConfirmDialog({
  isOpen,
  title,
  message,
  confirmText = '确定',
  cancelText = '取消',
  onConfirm,
  onCancel,
  variant = 'danger'
}: ConfirmDialogProps) {
  if (!isOpen) return null;

  const variantStyles = {
    danger: {
      iconBg: 'bg-red-50 border-red-100',
      iconColor: 'text-red-500',
      headerBg: 'bg-gradient-to-r from-red-50/50 to-orange-50/50',
      button: 'bg-red-500 hover:bg-red-600 text-white shadow-sm',
      icon: Trash2
    },
    warning: {
      iconBg: 'bg-yellow-50 border-yellow-100',
      iconColor: 'text-yellow-500',
      headerBg: 'bg-gradient-to-r from-yellow-50/50 to-amber-50/50',
      button: 'bg-yellow-500 hover:bg-yellow-600 text-white shadow-sm',
      icon: AlertTriangle
    },
    info: {
      iconBg: 'bg-blue-50 border-blue-100',
      iconColor: 'text-blue-500',
      headerBg: 'bg-gradient-to-r from-blue-50/50 to-indigo-50/50',
      button: 'bg-blue-500 hover:bg-blue-600 text-white shadow-sm',
      icon: Info
    }
  };

  const styles = variantStyles[variant];
  const IconComponent = styles.icon;

  return (
    <>
      {/* Overlay */}
      <div
        onClick={onCancel}
        className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center animate-in fade-in duration-200"
      >
        {/* Dialog */}
        <div
          onClick={(e) => e.stopPropagation()}
          className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 animate-in zoom-in-95 duration-200 overflow-hidden border border-slate-100"
        >
          {/* Header with gradient background */}
          <div className={`px-6 pt-6 pb-4 ${styles.headerBg} border-b border-slate-100`}>
            <div className="flex items-start gap-4">
              {/* Icon */}
              <div className={`w-12 h-12 rounded-xl ${styles.iconBg} border flex items-center justify-center flex-shrink-0`}>
                <IconComponent className={`w-6 h-6 ${styles.iconColor}`} />
              </div>
              
              <div className="flex-1">
                {/* Title */}
                <h3 className="text-slate-900 mb-1.5">
                  {title}
                </h3>
                
                {/* Message */}
                <p className="text-sm text-slate-600 leading-relaxed">
                  {message}
                </p>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="px-6 py-4 bg-slate-50/50 flex gap-3">
            <button
              onClick={onCancel}
              className="flex-1 px-4 py-2.5 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 hover:border-slate-300 transition-all"
            >
              {cancelText}
            </button>
            <button
              onClick={onConfirm}
              className={`flex-1 px-4 py-2.5 rounded-lg ${styles.button} transition-all`}
            >
              {confirmText}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}