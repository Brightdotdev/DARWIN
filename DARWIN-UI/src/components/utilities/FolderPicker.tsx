import { useState, useRef, useEffect, useCallback, useMemo } from 'react';

/* ─── Helper: darken a hex color ─── */
const darkenColor = (hex: string, percent: number): string => {
  let color = hex.startsWith('#') ? hex.slice(1) : hex;
  if (color.length === 3) {
    color = color.split('').map(c => c + c).join('');
  }
  const num = parseInt(color, 16);
  let r = (num >> 16) & 0xff;
  let g = (num >> 8) & 0xff;
  let b = num & 0xff;
  r = Math.max(0, Math.min(255, Math.floor(r * (1 - percent))));
  g = Math.max(0, Math.min(255, Math.floor(g * (1 - percent))));
  b = Math.max(0, Math.min(255, Math.floor(b * (1 - percent))));
  return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
};

/* ─── Types ─── */
export interface FileItem {
  id: string;
  name: string;
  metadata: {
    size: string;
    type: string;
    modified: string;
  };
}

type SortKey = 'name' | 'type' | 'size' | 'modified';

/* ─── Dummy data ─── */
const dummyFiles: FileItem[] = [
  { id: '1',  name: 'README.md',      metadata: { size: '2.4 KB',  type: 'Markdown',   modified: '2h ago' } },
  { id: '2',  name: 'package.json',   metadata: { size: '1.1 KB',  type: 'JSON',       modified: '1d ago' } },
  { id: '3',  name: 'App.jsx',        metadata: { size: '4.8 KB',  type: 'React',      modified: '3h ago' } },
  { id: '4',  name: 'styles.css',     metadata: { size: '3.2 KB',  type: 'CSS',        modified: '5h ago' } },
  { id: '5',  name: 'config.js',      metadata: { size: '0.8 KB',  type: 'JavaScript', modified: '1w ago' } },
  { id: '6',  name: 'data.csv',       metadata: { size: '12.5 KB', type: 'CSV',        modified: '2d ago' } },
  { id: '7',  name: 'notes.txt',      metadata: { size: '0.5 KB',  type: 'Text',       modified: '1h ago' } },
  { id: '8',  name: 'Dockerfile',     metadata: { size: '1.9 KB',  type: 'Docker',     modified: '3d ago' } },
  { id: '9',  name: 'README_old.md',  metadata: { size: '1.2 KB',  type: 'Markdown',   modified: '1mo ago' } },
  { id: '10', name: 'api.ts',         metadata: { size: '6.3 KB',  type: 'TypeScript', modified: '4h ago' } },
  { id: '11', name: 'utils.js',       metadata: { size: '2.1 KB',  type: 'JavaScript', modified: '6h ago' } },
  { id: '12', name: 'index.html',     metadata: { size: '0.9 KB',  type: 'HTML',       modified: '1d ago' } },
];

/* ─── File type colors ─── */
const typeColors: Record<string, string> = {
  React: '#61DAFB',
  JavaScript: '#F7DF1E',
  TypeScript: '#3178C6',
  CSS: '#264DE4',
  HTML: '#E34F26',
  JSON: '#000000',
  Markdown: '#083FA1',
  CSV: '#217346',
  Text: '#9CA3AF',
  Docker: '#2496ED',
};

const getTypeColor = (type: string) => typeColors[type] || '#6B7280';

/* ─── Inline keyframes ─── */
const folderStyles = `
  @keyframes badge-pop {
    0% { transform: scale(0); }
    80% { transform: scale(1.2); }
    100% { transform: scale(1); }
  }
  @keyframes paper-enter {
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes chip-enter {
    0% { opacity: 0; transform: translateX(20px) scale(0.9); }
    100% { opacity: 1; transform: translateX(0) scale(1); }
  }
  @keyframes file-peek {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-2px); }
  }
  @keyframes bin-pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }
  @keyframes toast-enter {
    0% { opacity: 0; transform: translateY(-20px); }
    100% { opacity: 1; transform: translateY(0); }
  }
  @keyframes toast-exit {
    0% { opacity: 1; transform: translateY(0); }
    100% { opacity: 0; transform: translateY(-20px); }
  }
`;

/* ─── Component ─── */
interface FolderPickerProps {
  color?: string;
  size?: number;
  files?: FileItem[];
  className?: string;
  onSelectionChange?: (files: FileItem[]) => void;
}

const FolderPicker = ({
  color = '#5227FF',
  size = 1,
  files = dummyFiles,
  className = '',
  onSelectionChange = () => {},
}: FolderPickerProps) => {
  const [isFolderOpen, setIsFolderOpen] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<(FileItem & { addedAt: number })[]>([]);
  const [flyer, setFlyer] = useState<FileItem | null>(null);
  const [flyerStyle, setFlyerStyle] = useState<React.CSSProperties | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('name');
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const [isDragging, setIsDragging] = useState(false);
  const [toast, setToast] = useState<{ message: string; action?: () => void } | null>(null);

  const fileRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const binRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const undoStack = useRef<(FileItem & { addedAt: number })[]>([]);

  const folderBackColor = darkenColor(color, 0.08);
  const paper1 = darkenColor('#ffffff', 0.1);
  const paper2 = darkenColor('#ffffff', 0.05);
  const paper3 = '#ffffff';

  /* ─── Derived state ─── */
  const filteredFiles = useMemo(() => {
    let result = [...files];
    
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter(f => 
        f.name.toLowerCase().includes(q) || 
        f.metadata.type.toLowerCase().includes(q)
      );
    }
    
    result.sort((a, b) => {
      switch (sortKey) {
        case 'name': return a.name.localeCompare(b.name);
        case 'type': return a.metadata.type.localeCompare(b.metadata.type);
        case 'size': return a.metadata.size.localeCompare(b.metadata.size);
        case 'modified': return a.metadata.modified.localeCompare(b.metadata.modified);
        default: return 0;
      }
    });
    
    return result;
  }, [files, searchQuery, sortKey]);

  const selectedIds = useMemo(() => 
    new Set(selectedFiles.map(f => f.id)),
    [selectedFiles]
  );

  /* ─── Effects ─── */
  useEffect(() => {
    onSelectionChange(selectedFiles);
  }, [selectedFiles, onSelectionChange]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (!isModalOpen) return;
      
      if (e.key === 'Escape') {
        handleClose();
        return;
      }
      
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const delta = e.key === 'ArrowDown' ? 1 : -1;
        setFocusedIndex(prev => {
          const next = Math.max(0, Math.min(filteredFiles.length - 1, prev + delta));
          const file = filteredFiles[next];
          if (file) fileRefs.current[file.id]?.focus();
          return next;
        });
      }
      
      if (e.key === 'Enter' && focusedIndex >= 0) {
        const file = filteredFiles[focusedIndex];
        if (file && !selectedIds.has(file.id)) handleSelectFile(file);
      }
      
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault();
        handleUndo();
      }
      
      if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isModalOpen, filteredFiles, focusedIndex, selectedIds]);

  /* ─── Toast timer ─── */
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(timer);
  }, [toast]);

  /* ─── Actions ─── */
  const handleFolderClick = () => {
    if (isFolderOpen) return;
    setIsFolderOpen(true);
    setTimeout(() => {
      setIsModalOpen(true);
      setFocusedIndex(-1);
    }, 280);
  };

  const handleClose = () => {
    setIsModalOpen(false);
    setSearchQuery('');
    setFocusedIndex(-1);
    setTimeout(() => setIsFolderOpen(false), 300);
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) handleClose();
  };

  const showToast = (message: string, action?: () => void) => {
    setToast({ message, action });
  };

  const addFileToBin = useCallback((file: FileItem) => {
    setSelectedFiles(prev => {
      if (prev.find(f => f.id === file.id)) return prev;
      undoStack.current = [...prev];
      return [...prev, { ...file, addedAt: Date.now() }];
    });
    showToast(`Added ${file.name}`, () => handleRemoveFile(file.id));
  }, []);


  const handleUndo = () => {
  if (undoStack.current.length > 0) {
    setSelectedFiles(undoStack.current);
    undoStack.current = [];
    showToast('Undo successful');
  }
};

  /* ─── Fly animation ─── */
  const handleSelectFile = useCallback((file: FileItem) => {
    if (selectedIds.has(file.id)) return;
    
    const cardEl = fileRefs.current[file.id];
    const binEl = binRef.current;
    if (!cardEl || !binEl) {
      addFileToBin(file);
      return;
    }

    const cardRect = cardEl.getBoundingClientRect();
    const binRect = binEl.getBoundingClientRect();

    setFlyer(file);
    setFlyerStyle({
      position: 'fixed',
      left: cardRect.left + cardRect.width / 2,
      top: cardRect.top + cardRect.height / 2,
      transform: 'translate(-50%, -50%) scale(1)',
      opacity: 1,
      transition: 'none',
      zIndex: 9999,
      pointerEvents: 'none',
    });

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setFlyerStyle(prev => ({
          ...prev,
          left: binRect.left + binRect.width / 2,
          top: binRect.top + 30,
          transform: 'translate(-50%, -50%) scale(0.5)',
          opacity: 0.7,
          transition: 'all 0.55s cubic-bezier(0.34, 1.56, 0.64, 1)',
        }));
      });
    });

    setTimeout(() => {
      addFileToBin(file);
      setFlyer(null);
      setFlyerStyle(null);
    }, 560);
  }, [addFileToBin, selectedIds]);

  const handleDoubleClick = (e: React.MouseEvent, file: FileItem) => {
    e.stopPropagation();
    handleSelectFile(file);
  };

  /* ─── Drag & Drop ─── */
  const handleDragStart = (e: React.DragEvent, file: FileItem) => {
    e.dataTransfer.setData('application/json', JSON.stringify(file));
    e.dataTransfer.setData('text/plain', file.id);
    e.dataTransfer.effectAllowed = 'copy';
    setIsDragging(true);
    
    const el = e.currentTarget as HTMLElement;
    el.style.opacity = '0.4';
    el.style.transform = 'scale(0.95)';
  };

  const handleDragEnd = (e: React.DragEvent) => {
    setIsDragging(false);
    const el = e.currentTarget as HTMLElement;
    el.style.opacity = '';
    el.style.transform = '';
  };

  const handleBinDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
    if (binRef.current) {
      binRef.current.style.transform = 'scale(1.03)';
      binRef.current.style.boxShadow = '0 15px 50px rgba(82, 39, 255, 0.3)';
    }
  };

  const handleBinDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (binRef.current && !binRef.current.contains(e.relatedTarget as Node)) {
      binRef.current.style.transform = '';
      binRef.current.style.boxShadow = '';
    }
  };

  const handleBinDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (binRef.current) {
      binRef.current.style.transform = '';
      binRef.current.style.boxShadow = '';
    }

    let file: FileItem | null = null;
    
    try {
      const jsonData = e.dataTransfer.getData('application/json');
      if (jsonData) file = JSON.parse(jsonData) as FileItem;
    } catch { /* ignore */ }

    if (!file) {
      try {
        const fileId = e.dataTransfer.getData('text/plain');
        file = files.find(f => f.id === fileId) || null;
      } catch { /* ignore */ }
    }

    if (file?.id && !selectedIds.has(file.id)) {
      addFileToBin(file);
    }
  };

  const handleRemoveFile = (fileId: string) => {
    setSelectedFiles(prev => {
      undoStack.current = [...prev];
      return prev.filter(f => f.id !== fileId);
    });
    const file = files.find(f => f.id === fileId);
    if (file) showToast(`Removed ${file.name}`, () => addFileToBin(file));
  };

  const getPaperBg = (index: number) => {
    if (index % 3 === 2) return paper1;
    if (index % 3 === 1) return paper2;
    return paper3;
  };

  const visibleFiles = files.slice(0, 3);

  return (
    <div className={className}>
      <style>{folderStyles}</style>

      {/* ===== Toast ===== */}
      {toast && (
        <div
          className="fixed top-6 left-1/2 -translate-x-1/2 z-[300] flex items-center gap-3 px-5 py-3 rounded-xl bg-zinc-900 text-white shadow-[0_10px_40px_rgba(0,0,0,0.3)] border border-zinc-700"
          style={{ animation: 'toast-enter 0.3s ease' }}
        >
          <span className="text-sm">{toast.message}</span>
          {toast.action && (
            <button
              className="text-xs font-semibold text-blue-400 hover:text-blue-300 transition-colors"
              onClick={() => {
                toast.action?.();
                setToast(null);
              }}
            >
              Undo
            </button>
          )}
        </div>
      )}

      {/* ===== Folder Trigger ===== */}
      <div
        className="inline-flex flex-col items-center gap-3"
        style={{ transform: `scale(${size})` }}
      >
        <div
          className={`
            relative cursor-pointer transition-transform duration-200 ease-out outline-none
            ${isFolderOpen ? '' : 'hover:-translate-y-1.5'}
          `}
          style={{ width: 80, height: 64 }}
          onClick={handleFolderClick}
          role="button"
          tabIndex={0}
          aria-label={`Open folder with ${files.length} files`}
          onKeyDown={(e) => e.key === 'Enter' && handleFolderClick()}
        >
          {/* Folder back */}
          <div
            className="absolute inset-0 rounded-r-xl rounded-bl-xl transition-all duration-300"
            style={{ backgroundColor: folderBackColor }}
          >
            {/* Tab */}
            <div
              className="absolute rounded-t-md"
              style={{ 
                backgroundColor: folderBackColor, 
                bottom: '98%', 
                left: 0, 
                width: 28, 
                height: 10 
              }}
            />
            
            {/* Mini file representations */}
            {visibleFiles.length > 0 && !isFolderOpen && (
              <div className="absolute inset-0 flex items-end justify-center pb-1.5 px-2 gap-0.5 z-[1]">
                {visibleFiles.map((file, i) => (
                  <div
                    key={file.id}
                    className="rounded-sm flex items-center justify-center"
                    style={{
                      width: 14 - i * 2,
                      height: 18 - i * 2,
                      backgroundColor: getTypeColor(file.metadata.type),
                      opacity: 0.85,
                      animation: `file-peek ${2 + i * 0.3}s ease-in-out infinite`,
                      animationDelay: `${i * 0.2}s`,
                    }}
                  >
                    <span className="text-[5px] text-white font-bold truncate px-0.5">
                      {file.metadata.type.slice(0, 2).toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {files.length === 0 && !isFolderOpen && (
              <div className="absolute inset-0 flex items-center justify-center z-[1]">
                <span className="text-[8px] text-white/40 font-medium">Empty</span>
              </div>
            )}
            
            {/* Front flaps */}
            <div
              className={`
                absolute inset-0 z-[3] rounded-r-xl rounded-bl-xl origin-bottom transition-all duration-300
                ${isFolderOpen ? 'skew-x-[15deg] scale-y-[0.6]' : ''}
              `}
              style={{ backgroundColor: color }}
            />
            <div
              className={`
                absolute inset-0 z-[2] rounded-r-xl rounded-bl-xl origin-bottom transition-all duration-300
                ${isFolderOpen ? '-skew-x-[15deg] scale-y-[0.6]' : ''}
              `}
              style={{ backgroundColor: color }}
            />
          </div>

          {files.length > 0 && (
            <span
              className="absolute -top-1.5 -right-1.5 min-w-[20px] h-5 rounded-full bg-red-500 text-white text-[10px] font-bold flex items-center justify-center px-1 shadow-lg z-10"
              style={{ animation: 'badge-pop 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)' }}
            >
              {files.length}
            </span>
          )}
        </div>
        <p className="text-xs text-zinc-500 opacity-80">Click to open</p>
      </div>

      {/* ===== Modal ===== */}
      {isFolderOpen && (
        <div
          className={`
            fixed inset-0 flex items-center justify-center p-5
            transition-opacity duration-300
            ${isModalOpen ? 'opacity-100' : 'opacity-0'}
          `}
          style={{ 
            backgroundColor: 'rgba(9, 9, 11, 0.6)', 
            backdropFilter: 'blur(4px)',
            zIndex: 100 
          }}
          onClick={handleBackdropClick}
          role="presentation"
        >
          <div
            className={`
              relative w-[680px] max-w-[90vw] max-h-[85vh] flex flex-col overflow-hidden
              rounded-r-xl rounded-bl-xl shadow-[0_20px_60px_rgba(0,0,0,0.25)]
              transition-all duration-[400ms]
              ${isModalOpen ? 'scale-100 translate-y-0 opacity-100' : 'scale-[0.85] translate-y-[30px] opacity-0'}
            `}
            style={{
              background: `linear-gradient(135deg, ${folderBackColor} 0%, ${color} 100%)`,
            }}
            role="dialog"
            aria-modal="true"
            aria-labelledby="folder-modal-title"
          >
            {/* Tab */}
            <div
              className="absolute rounded-t-xl opacity-90"
              style={{ 
                backgroundColor: folderBackColor,
                bottom: '100%',
                left: 0,
                width: 140,
                height: 28
              }}
            />

            {/* Header with search and sort */}
            <div className="relative flex items-center gap-3 px-5 py-3 bg-black/15 border-b border-white/10">
              <span id="folder-modal-title" className="text-base font-semibold text-white tracking-wide shrink-0">
                {filteredFiles.length} Files
              </span>
              
              {/* Search */}
              <div className="flex-1 relative">
                <input
                  ref={searchRef}
                  type="text"
                  placeholder="Search files... (Press /)"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-1.5 rounded-lg bg-white/10 text-white text-sm placeholder:text-white/40 border border-white/10 focus:outline-none focus:border-white/30 transition-colors"
                />
                {searchQuery && (
                  <button
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-white/40 hover:text-white text-xs"
                    onClick={() => setSearchQuery('')}
                  >
                    ×
                  </button>
                )}
              </div>

              {/* Sort dropdown */}
              <select
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value as SortKey)}
                className="px-2 py-1.5 rounded-lg bg-white/10 text-white text-sm border border-white/10 focus:outline-none focus:border-white/30 cursor-pointer"
              >
                <option value="name" className="bg-zinc-800">Name</option>
                <option value="type" className="bg-zinc-800">Type</option>
                <option value="size" className="bg-zinc-800">Size</option>
                <option value="modified" className="bg-zinc-800">Modified</option>
              </select>

              <button
                className="w-8 h-8 rounded-lg bg-white/15 text-white text-xl flex items-center justify-center cursor-pointer transition-all hover:bg-white/25 hover:scale-105 shrink-0"
                onClick={handleClose}
                aria-label="Close folder"
              >
                ×
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 p-6 overflow-y-auto [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-white/20 [&::-webkit-scrollbar-thumb]:rounded">
              <p className="text-[13px] text-white/70 text-center mb-5">
                Double-click to select, drag to bin, or press Enter on focused card
              </p>

              {filteredFiles.length === 0 ? (
                <div className="text-center py-12 text-white/50">
                  <p className="text-lg mb-2">No files found</p>
                  <p className="text-sm">Try a different search term</p>
                </div>
              ) : (
                <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-5">
                  {filteredFiles.map((file, index) => {
                    const isSelected = selectedIds.has(file.id);
                    const typeColor = getTypeColor(file.metadata.type);
                    
                    return (
                      <div
                        key={file.id}
                        ref={(el) => {
                          if (el) fileRefs.current[file.id] = el;
                          else delete fileRefs.current[file.id];
                        }}
                        className={`
                          relative rounded-xl p-5 cursor-pointer select-none outline-none
                          shadow-[0_2px_6px_rgba(0,0,0,0.08)] transition-all duration-250
                          ${isSelected 
                            ? 'opacity-40 grayscale cursor-not-allowed' 
                            : 'hover:-translate-y-1.5 hover:scale-[1.03] hover:shadow-[0_12px_24px_rgba(0,0,0,0.15)] hover:z-[2] focus-visible:ring-2 focus-visible:ring-white/30'
                          }
                        `}
                        style={{
                          backgroundColor: getPaperBg(index),
                          opacity: isSelected ? 0.4 : 0,
                          transform: isSelected ? 'none' : 'translateY(16px)',
                          animation: isSelected ? 'none' : `paper-enter 0.4s ease forwards`,
                          animationDelay: `${index * 0.04}s`,
                        }}
                        onDoubleClick={(e) => !isSelected && handleDoubleClick(e, file)}
                        draggable={!isSelected}
                        onDragStart={(e) => !isSelected && handleDragStart(e, file)}
                        onDragEnd={handleDragEnd}
                        role="button"
                        tabIndex={isSelected ? -1 : 0}
                        aria-label={`${file.name}, ${file.metadata.type}, ${file.metadata.size}${isSelected ? ', already selected' : ''}`}
                        onKeyDown={(e) => {
                          if (isSelected) return;
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            handleSelectFile(file);
                          }
                        }}
                        onFocus={() => setFocusedIndex(index)}
                      >
                        {/* Selected checkmark */}
                        {isSelected && (
                          <div className="absolute top-2 right-2 w-6 h-6 rounded-full bg-green-500 flex items-center justify-center shadow-md z-10">
                            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        )}

                        <div
                          className="w-10 h-10 rounded-xl flex items-center justify-center text-white text-[13px] font-bold tracking-wider shadow-md mb-2.5 mx-auto"
                          style={{ backgroundColor: typeColor }}
                        >
                          {file.metadata.type.slice(0, 2).toUpperCase()}
                        </div>
                        <div className="text-sm font-semibold text-zinc-900 break-words leading-tight">
                          {file.name}
                        </div>
                        <div className="flex gap-2 mt-2.5 justify-center">
                          <span className="text-[11px] text-zinc-500 bg-black/5 px-2 py-0.5 rounded">
                            {file.metadata.type}
                          </span>
                          <span className="text-[11px] text-zinc-500 bg-black/5 px-2 py-0.5 rounded">
                            {file.metadata.size}
                          </span>
                        </div>
                        <div className="text-[10px] text-zinc-500 mt-1 opacity-0 transition-opacity duration-200 group-hover:opacity-80">
                          {isSelected ? 'Already in bin' : 'Double-click to select'}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ===== Flying File ===== */}
      {flyer && flyerStyle && (
        <div
          className="fixed pointer-events-none min-w-[100px] px-4 py-2 rounded-lg bg-white shadow-[0_4px_16px_rgba(0,0,0,0.2)] flex items-center justify-center"
          style={{ ...flyerStyle, zIndex: 9999 }}
        >
          <span className="text-[13px] font-semibold text-zinc-900 whitespace-nowrap">
            {flyer.name}
          </span>
        </div>
      )}

      {/* ===== Bin ===== */}
      <div
        className={`
          fixed bottom-6 right-6 w-[300px] max-h-[400px] flex flex-col overflow-hidden 
          rounded-r-xl rounded-bl-xl shadow-[0_10px_40px_rgba(0,0,0,0.25)] transition-all duration-300
          ${isDragging ? 'animate-[bin-pulse_1s_ease-in-out_infinite]' : ''}
        `}
        style={{
          background: `linear-gradient(135deg, ${folderBackColor} 0%, ${color} 100%)`,
          zIndex: 200,
        }}
        ref={binRef}
        onDragOver={handleBinDragOver}
        onDragLeave={handleBinDragLeave}
        onDrop={handleBinDrop}
      >
        {/* Bin tab */}
        <div
          className="absolute rounded-t-[11px] opacity-90"
          style={{ 
            backgroundColor: folderBackColor,
            bottom: '100%',
            left: 0,
            width: 90,
            height: 20
          }}
        />

        {/* Bin header */}
        <div className="relative flex items-center justify-between px-[18px] py-3 bg-black/15 border-b border-white/10">
          <span className="text-sm font-semibold text-white">Selected</span>
          <span className="min-w-6 h-6 rounded-full bg-white/20 text-white text-xs font-bold flex items-center justify-center px-2">
            {selectedFiles.length}
          </span>
        </div>

        {/* Bin content */}
        <div className="p-4 overflow-y-auto max-h-[300px] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-white/20 [&::-webkit-scrollbar-thumb]:rounded">
          {selectedFiles.length === 0 ? (
            <div 
              className="text-center py-5 text-white/60 text-[13px] border-2 border-dashed border-white/20 rounded-lg transition-all"
              onDragOver={handleBinDragOver}
              onDragLeave={handleBinDragLeave}
              onDrop={handleBinDrop}
            >
              {isDragging ? 'Drop here!' : 'Drop files here'}
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {selectedFiles.map((file) => (
                <div
                  key={`${file.id}-${file.addedAt}`}
                  className="bg-white/95 rounded-lg px-3.5 py-2.5 flex items-center justify-between gap-2.5 shadow-[0_1px_4px_rgba(0,0,0,0.08)] transition-all hover:-translate-x-0.5 hover:shadow-[0_3px_10px_rgba(0,0,0,0.12)]"
                  style={{ animation: 'chip-enter 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)' }}
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <div 
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: getTypeColor(file.metadata.type) }}
                    />
                    <span className="text-[13px] font-medium text-zinc-900 truncate">
                      {file.name}
                    </span>
                  </div>
                  <button
                    className="w-[22px] h-[22px] rounded-md flex items-center justify-center text-zinc-500 text-base leading-none cursor-pointer transition-all hover:bg-red-500 hover:text-white flex-shrink-0"
                    onClick={() => handleRemoveFile(file.id)}
                    aria-label={`Remove ${file.name}`}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FolderPicker;