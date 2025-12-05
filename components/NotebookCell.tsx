import React from 'react';
import { Play, CheckCircle, Loader2 } from 'lucide-react';

interface NotebookCellProps {
  title: string;
  language?: 'python' | 'sql' | 'markdown';
  status: 'idle' | 'running' | 'success';
  code?: string;
  children?: React.ReactNode;
  onRun?: () => void;
  readOnly?: boolean;
}

export const NotebookCell: React.FC<NotebookCellProps> = ({ 
  title, 
  language = 'python', 
  status, 
  code, 
  children,
  onRun,
  readOnly = false
}) => {
  return (
    <div className="mb-6 border border-gray-200 rounded-md bg-white shadow-sm overflow-hidden">
      {/* Cell Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200 text-xs text-gray-500 font-mono">
        <div className="flex items-center space-x-2">
          <span className="font-bold text-gray-600">{language.toUpperCase()}</span>
          <span>{title}</span>
        </div>
        <div className="flex items-center space-x-3">
           {status === 'running' && <span className="flex items-center text-blue-600"><Loader2 className="w-3 h-3 mr-1 animate-spin"/> Running...</span>}
           {status === 'success' && <span className="flex items-center text-green-600"><CheckCircle className="w-3 h-3 mr-1"/> Executed</span>}
           {!readOnly && onRun && (
             <button 
              onClick={onRun}
              disabled={status === 'running'}
              className="p-1 hover:bg-gray-200 rounded transition-colors"
             >
               <Play className="w-3 h-3 fill-current text-gray-600" />
             </button>
           )}
        </div>
      </div>

      {/* Code Input Area */}
      {code && (
        <div className="px-4 py-3 bg-[#fbfbfb] border-b border-gray-100 overflow-x-auto">
          <pre className="text-sm font-mono text-gray-800 whitespace-pre-wrap">
            {code}
          </pre>
        </div>
      )}

      {/* Output/Result Area */}
      {children && (
        <div className="p-4 bg-white animate-in fade-in slide-in-from-top-1 duration-300">
          {children}
        </div>
      )}
    </div>
  );
};