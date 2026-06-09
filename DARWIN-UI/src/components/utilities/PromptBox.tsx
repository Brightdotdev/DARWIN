"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowUp,
  Paperclip,
  Square,
  X,
  StopCircle,
  Mic,
  Globe,
  BrainCog,
  FolderCode,
} from "lucide-react";

import { cn } from "../..//lib/utils";
import { Button } from "../../components/ui/button";
import { Textarea } from "../../components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../../components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "../../components/ui/dialog";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface PromptInputBoxProps {
  /** Called when the user submits a message (text + optional files). */
  onSend?: (message: string, files?: File[]) => void;
  /** Loading / generation state. Turns the send button into a stop button. */
  isLoading?: boolean;
  /** Placeholder when no mode is active. */
  placeholder?: string;
  /** Additional classes for the root container. */
  className?: string;
  /** Max textarea height in px or CSS string. */
  maxHeight?: number | string;
  /** Globally disable all interactions. */
  disabled?: boolean;
}

interface PromptInputContextType {
  isLoading: boolean;
  value: string;
  setValue: (value: string) => void;
  maxHeight: number | string;
  onSubmit?: () => void;
  disabled: boolean;
}

interface VoiceRecorderProps {
  isRecording: boolean;
  onStartRecording: () => void;
  onStopRecording: (duration: number) => void;
  visualizerBars?: number;
}

interface ImageViewDialogProps {
  imageUrl: string | null;
  onClose: () => void;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

const PromptInputContext = React.createContext<PromptInputContextType | null>(null);

function usePromptInput() {
  const context = React.useContext(PromptInputContext);
  if (!context) throw new Error("usePromptInput must be used within a PromptInput provider");
  return context;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOICE RECORDER  (bug-fixed: stable bars, correct timer lifecycle)
// ═══════════════════════════════════════════════════════════════════════════════

const VoiceRecorder = React.memo<VoiceRecorderProps>(
  ({ isRecording, onStartRecording, onStopRecording, visualizerBars = 32 }) => {
    const [time, setTime] = React.useState(0);
    const timeRef = React.useRef(0);
    const timerRef = React.useRef<ReturnType<typeof setInterval> | null>(null);
    const onStopRecordingRef = React.useRef(onStopRecording);
    onStopRecordingRef.current = onStopRecording;

    // Stable random bar heights — do NOT regenerate on every render
    const barHeights = React.useMemo(
      () => Array.from({ length: visualizerBars }, () => Math.max(15, Math.random() * 100)),
      [visualizerBars]
    );

    React.useEffect(() => {
      if (isRecording) {
        onStartRecording();
        setTime(0);
        timeRef.current = 0;
        timerRef.current = setInterval(() => {
          setTime((prev) => {
            const next = prev + 1;
            timeRef.current = next;
            return next;
          });
        }, 1000);
      } else {
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          onStopRecordingRef.current(timeRef.current);
        }
        setTime(0);
        timeRef.current = 0;
      }
      return () => {
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      };
    }, [isRecording, onStartRecording]);

    const formatTime = (seconds: number) => {
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    };

    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center w-full transition-all duration-300 py-3",
          isRecording ? "opacity-100" : "opacity-0 h-0"
        )}
        aria-hidden={!isRecording}
      >
        <div className="flex items-center gap-2 mb-3">
          <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
          <span className="font-mono text-sm text-white/80" aria-live="polite">
            {formatTime(time)}
          </span>
        </div>
        <div className="w-full h-10 flex items-center justify-center gap-0.5 px-4">
          {barHeights.map((height, i) => (
            <div
              key={i}
              className="w-0.5 rounded-full bg-white/50 animate-pulse"
              style={{
                height: `${height}%`,
                animationDelay: `${i * 0.05}s`,
                animationDuration: `${0.5 + Math.random() * 0.5}s`,
              }}
            />
          ))}
        </div>
      </div>
    );
  }
);
VoiceRecorder.displayName = "VoiceRecorder";

// ═══════════════════════════════════════════════════════════════════════════════
// IMAGE DIALOG  (uses shadcn Dialog — no custom primitive re-implementation)
// ═══════════════════════════════════════════════════════════════════════════════

const ImageViewDialog = React.memo<ImageViewDialogProps>(({ imageUrl, onClose }) => {
  if (!imageUrl) return null;
  return (
    <Dialog open={!!imageUrl} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="p-0 border-none bg-transparent shadow-none max-w-[90vw] md:max-w-[800px]">
        <DialogTitle className="sr-only">Image Preview</DialogTitle>
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="relative bg-[#1F2023] rounded-2xl overflow-hidden shadow-2xl"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl}
            alt="Full preview"
            className="w-full max-h-[80vh] object-contain rounded-2xl"
          />
        </motion.div>
      </DialogContent>
    </Dialog>
  );
});
ImageViewDialog.displayName = "ImageViewDialog";

// ═══════════════════════════════════════════════════════════════════════════════
// DECORATIVE DIVIDER
// ═══════════════════════════════════════════════════════════════════════════════

const CustomDivider = React.memo(() => (
  <div className="relative h-6 w-[1.5px] mx-1">
    <div
      className="absolute inset-0 bg-gradient-to-t from-transparent via-[#9b87f5]/70 to-transparent rounded-full"
      style={{
        clipPath:
          "polygon(0% 0%, 100% 0%, 100% 40%, 140% 50%, 100% 60%, 100% 100%, 0% 100%, 0% 60%, -40% 50%, 0% 40%)",
      }}
    />
  </div>
));
CustomDivider.displayName = "CustomDivider";

// ═══════════════════════════════════════════════════════════════════════════════
// PROMPT INPUT SYSTEM  (context-backed, composable)
// ═══════════════════════════════════════════════════════════════════════════════

interface PromptInputProps {
  isLoading?: boolean;
  value?: string;
  onValueChange?: (value: string) => void;
  maxHeight?: number | string;
  onSubmit?: () => void;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  onDragOver?: (e: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave?: (e: React.DragEvent<HTMLDivElement>) => void;
  onDrop?: (e: React.DragEvent<HTMLDivElement>) => void;
  onPaste?: (e: React.ClipboardEvent<HTMLDivElement>) => void;
}

const PromptInput = React.forwardRef<HTMLDivElement, PromptInputProps>(
  (
    {
      className,
      isLoading = false,
      maxHeight = 240,
      value,
      onValueChange,
      onSubmit,
      children,
      disabled = false,
      onDragOver,
      onDragLeave,
      onDrop,
      onPaste,
    },
    ref
  ) => {
    const [internalValue, setInternalValue] = React.useState(value || "");

    const handleChange = React.useCallback(
      (newValue: string) => {
        if (value === undefined) setInternalValue(newValue);
        onValueChange?.(newValue);
      },
      [value, onValueChange]
    );

    const contextValue = React.useMemo<PromptInputContextType>(
      () => ({
        isLoading,
        value: value ?? internalValue,
        setValue: onValueChange ?? handleChange,
        maxHeight,
        onSubmit,
        disabled,
      }),
      [isLoading, value, internalValue, onValueChange, handleChange, maxHeight, onSubmit, disabled]
    );

    return (
      <TooltipProvider>
        <PromptInputContext.Provider value={contextValue}>
          <div
            ref={ref}
            className={cn(
              "rounded-3xl border border-[#444444] bg-[#1F2023] p-2 shadow-[0_8px_30px_rgba(0,0,0,0.24)] transition-all duration-300",
              isLoading && "border-red-500/70",
              className
            )}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onPaste={onPaste}
          >
            {children}
          </div>
        </PromptInputContext.Provider>
      </TooltipProvider>
    );
  }
);
PromptInput.displayName = "PromptInput";

interface PromptInputTextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  disableAutosize?: boolean;
}

const PromptInputTextarea = React.forwardRef<HTMLTextAreaElement, PromptInputTextareaProps>(
  ({ className, onKeyDown, onChange, disableAutosize = false, placeholder, ...props }, ref) => {
    const { value, setValue, maxHeight, onSubmit, disabled } = usePromptInput();
    const textareaRef = React.useRef<HTMLTextAreaElement>(null);
    React.useImperativeHandle(ref, () => textareaRef.current!);

    React.useEffect(() => {
      if (disableAutosize || !textareaRef.current) return;
      const el = textareaRef.current;
      el.style.height = "auto";
      const maxH = typeof maxHeight === "number" ? maxHeight : parseInt(maxHeight, 10) || 240;
      el.style.height = `${Math.min(el.scrollHeight, maxH)}px`;
    }, [value, maxHeight, disableAutosize]);

    const handleKeyDown = React.useCallback(
      (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          onSubmit?.();
        }
        onKeyDown?.(e);
      },
      [onSubmit, onKeyDown]
    );

    const handleChange = React.useCallback(
      (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setValue(e.target.value);
        onChange?.(e);
      },
      [setValue, onChange]
    );

    return (
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        className={cn(
          "darwin-scroll min-h-[44px] resize-none border-none bg-transparent text-base text-gray-100 placeholder:text-gray-400 focus-visible:ring-0 focus-visible:ring-offset-0",
          className
        )}
        disabled={disabled}
        placeholder={placeholder}
        rows={1}
        {...props}
      />
    );
  }
);
PromptInputTextarea.displayName = "PromptInputTextarea";

const PromptInputActions = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ children, className, ...props }, ref) => (
    <div ref={ref} className={cn("flex items-center gap-2", className)} {...props}>
      {children}
    </div>
  )
);
PromptInputActions.displayName = "PromptInputActions";

interface PromptInputActionProps {
  tooltip: React.ReactNode;
  children: React.ReactElement;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}

const PromptInputAction: React.FC<PromptInputActionProps> = ({
  tooltip,
  children,
  className,
  side = "top",
}) => (
  <Tooltip>
    <TooltipTrigger>{children}</TooltipTrigger>
    <TooltipContent side={side} className={cn("z-50", className)}>
      {tooltip}
    </TooltipContent>
  </Tooltip>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN EXPORT: PromptInputBox
// ═══════════════════════════════════════════════════════════════════════════════

export const PromptInputBox = React.forwardRef<HTMLDivElement, PromptInputBoxProps>(
  (
    {
      onSend = () => {},
      isLoading = false,
      placeholder = "Type your message here...",
      className,
      maxHeight = 240,
      disabled = false,
    },
    ref
  ) => {
    const [input, setInput] = React.useState("");
    const [files, setFiles] = React.useState<File[]>([]);
    const [filePreviews, setFilePreviews] = React.useState<Record<string, string>>({});
    const [selectedImage, setSelectedImage] = React.useState<string | null>(null);
    const [isRecording, setIsRecording] = React.useState(false);
    const [showSearch, setShowSearch] = React.useState(false);
    const [showThink, setShowThink] = React.useState(false);
    const [showCanvas, setShowCanvas] = React.useState(false);

    const uploadInputRef = React.useRef<HTMLInputElement>(null);
    const prevPreviewsRef = React.useRef<Record<string, string>>({});

    // ── Object URL lifecycle (no memory leaks) ───────────────────────────────
    React.useEffect(() => {
      const prev = prevPreviewsRef.current;
      Object.entries(prev).forEach(([name, url]) => {
        if (!filePreviews[name]) URL.revokeObjectURL(url);
      });
      prevPreviewsRef.current = { ...filePreviews };
    }, [filePreviews]);

    const isImageFile = (file: File) => file.type.startsWith("image/");

    const processFile = React.useCallback((file: File) => {
      if (!isImageFile(file)) {
        console.warn("[Darwin] Only image files are allowed");
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        console.warn("[Darwin] File too large (max 10MB)");
        return;
      }
      setFiles([file]);
      const objectUrl = URL.createObjectURL(file);
      setFilePreviews({ [file.name]: objectUrl });
    }, []);

    const handleDragOver = React.useCallback((e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
    }, []);

    const handleDragLeave = React.useCallback((e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
    }, []);

    const handleDrop = React.useCallback(
      (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        const droppedFiles = Array.from(e.dataTransfer.files);
        const imageFiles = droppedFiles.filter(isImageFile);
        if (imageFiles.length > 0) processFile(imageFiles[0]);
      },
      [processFile]
    );

    // Scoped paste — no global document listeners
    const handlePaste = React.useCallback(
      (e: React.ClipboardEvent<HTMLDivElement>) => {
        const items = e.clipboardData?.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
          if (items[i].type.indexOf("image") !== -1) {
            const file = items[i].getAsFile();
            if (file) {
              e.preventDefault();
              processFile(file);
              break;
            }
          }
        }
      },
      [processFile]
    );

    const handleRemoveFile = React.useCallback((index: number) => {
      const fileToRemove = files[index];
      if (fileToRemove) {
        setFilePreviews((prev) => {
          const next = { ...prev };
          delete next[fileToRemove.name];
          return next;
        });
      }
      setFiles((prev) => prev.filter((_, i) => i !== index));
    }, [files]);

    const openImageModal = React.useCallback((imageUrl: string) => setSelectedImage(imageUrl), []);

    const handleToggleChange = React.useCallback((value: "search" | "think") => {
      if (value === "search") {
        setShowSearch((prev) => !prev);
        setShowThink(false);
      } else {
        setShowThink((prev) => !prev);
        setShowSearch(false);
      }
    }, []);

    const handleCanvasToggle = React.useCallback(() => setShowCanvas((prev) => !prev), []);

    const handleSubmit = React.useCallback(() => {
      if (!input.trim() && files.length === 0) return;

      let prefix = "";
      if (showSearch) prefix = "[Search: ";
      else if (showThink) prefix = "[Think: ";
      else if (showCanvas) prefix = "[Canvas: ";

      const message = prefix ? `${prefix}${input}]` : input;
      onSend(message, files);

      setInput("");
      setFiles([]);
      setFilePreviews({});
    }, [input, files, showSearch, showThink, showCanvas, onSend]);

    const handleStartRecording = React.useCallback(() => {
      console.log("[Darwin] Voice recording started");
    }, []);

    const handleStopRecording = React.useCallback(
      (duration: number) => {
        console.log(`[Darwin] Voice recording stopped after ${duration}s`);
        setIsRecording(false);
        onSend(`[Voice message - ${duration} seconds]`, []);
      },
      [onSend]
    );

    const hasContent = input.trim() !== "" || files.length > 0;

    const activePlaceholder = React.useMemo(() => {
      if (showSearch) return "Search the web...";
      if (showThink) return "Think deeply...";
      if (showCanvas) return "Create on canvas...";
      return placeholder;
    }, [showSearch, showThink, showCanvas, placeholder]);

    const isDisabled = isLoading || isRecording || disabled;

    return (
      <>
        <PromptInput
          value={input}
          onValueChange={setInput}
          isLoading={isLoading}
          onSubmit={handleSubmit}
          maxHeight={maxHeight}
          className={cn(
            "w-full bg-[#1F2023] border-[#444444] shadow-[0_8px_30px_rgba(0,0,0,0.24)] transition-all duration-300 ease-in-out",
            isRecording && "border-red-500/70",
            className
          )}
          disabled={isDisabled}
          ref={ref}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onPaste={handlePaste}
        >
          {/* File Previews */}
          <AnimatePresence>
            {files.length > 0 && !isRecording && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="flex flex-wrap gap-2 p-0 pb-1"
              >
                {files.map((file, index) => (
                  <div key={`${file.name}-${index}`} className="relative group">
                    {file.type.startsWith("image/") && filePreviews[file.name] && (
                      <button
                        type="button"
                        className="w-16 h-16 rounded-xl overflow-hidden transition-all duration-300 block p-0 border-0 bg-transparent"
                        onClick={() => openImageModal(filePreviews[file.name])}
                        aria-label={`Preview ${file.name}`}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={filePreviews[file.name]}
                          alt={file.name}
                          className="h-full w-full object-cover"
                        />
                        <span
                          className="absolute top-1 right-1 rounded-full bg-black/70 p-0.5"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemoveFile(index);
                          }}
                          role="button"
                          aria-label="Remove file"
                        >
                          <X className="h-3 w-3 text-white" />
                        </span>
                      </button>
                    )}
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Textarea */}
          <div
            className={cn(
              "transition-all duration-300",
              isRecording ? "h-0 overflow-hidden opacity-0" : "opacity-100"
            )}
          >
            <PromptInputTextarea placeholder={activePlaceholder} className="text-base" />
          </div>

          {/* Voice Recorder */}
          {isRecording && (
            <VoiceRecorder
              isRecording={isRecording}
              onStartRecording={handleStartRecording}
              onStopRecording={handleStopRecording}
            />
          )}

          {/* Actions */}
          <PromptInputActions className="flex items-center justify-between gap-2 p-0 pt-2">
            <div
              className={cn(
                "flex items-center gap-1 transition-opacity duration-300",
                isRecording ? "opacity-0 invisible h-0" : "opacity-100 visible"
              )}
            >
              {/* Upload */}
              <PromptInputAction tooltip="Upload image">
                <button
                  type="button"
                  onClick={() => uploadInputRef.current?.click()}
                  className="flex h-8 w-8 text-[#9CA3AF] items-center justify-center rounded-full transition-colors hover:bg-gray-600/30 hover:text-[#D1D5DB] disabled:opacity-50"
                  disabled={isRecording || disabled}
                  aria-label="Upload image"
                >
                  <Paperclip className="h-5 w-5 transition-colors" />
                  <input
                    ref={uploadInputRef}
                    type="file"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files?.[0]) processFile(e.target.files[0]);
                      if (e.target) e.target.value = "";
                    }}
                    accept="image/*"
                    aria-hidden
                  />
                </button>
              </PromptInputAction>

              {/* Mode Toggles */}
              <div className="flex items-center">
                {/* Search */}
                <button
                  type="button"
                  onClick={() => handleToggleChange("search")}
                  disabled={disabled || isLoading}
                  className={cn(
                    "rounded-full transition-all flex items-center gap-1 px-2 py-1 border h-8",
                    showSearch
                      ? "bg-[#1EAEDB]/15 border-[#1EAEDB] text-[#1EAEDB]"
                      : "bg-transparent border-transparent text-[#9CA3AF] hover:text-[#D1D5DB] disabled:opacity-50"
                  )}
                  aria-pressed={showSearch}
                  aria-label="Toggle web search"
                >
                  <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                    <motion.div
                      animate={{ rotate: showSearch ? 360 : 0, scale: showSearch ? 1.1 : 1 }}
                      whileHover={{ rotate: showSearch ? 360 : 15, scale: 1.1, transition: { type: "spring", stiffness: 300, damping: 10 } }}
                      transition={{ type: "spring", stiffness: 260, damping: 25 }}
                    >
                      <Globe className={cn("w-4 h-4", showSearch ? "text-[#1EAEDB]" : "text-inherit")} />
                    </motion.div>
                  </div>
                  <AnimatePresence>
                    {showSearch && (
                      <motion.span
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: "auto", opacity: 1 }}
                        exit={{ width: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="text-xs overflow-hidden whitespace-nowrap text-[#1EAEDB] flex-shrink-0"
                      >
                        Search
                      </motion.span>
                    )}
                  </AnimatePresence>
                </button>

                <CustomDivider />

                {/* Think */}
                <button
                  type="button"
                  onClick={() => handleToggleChange("think")}
                  disabled={disabled || isLoading}
                  className={cn(
                    "rounded-full transition-all flex items-center gap-1 px-2 py-1 border h-8",
                    showThink
                      ? "bg-[#8B5CF6]/15 border-[#8B5CF6] text-[#8B5CF6]"
                      : "bg-transparent border-transparent text-[#9CA3AF] hover:text-[#D1D5DB] disabled:opacity-50"
                  )}
                  aria-pressed={showThink}
                  aria-label="Toggle think mode"
                >
                  <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                    <motion.div
                      animate={{ rotate: showThink ? 360 : 0, scale: showThink ? 1.1 : 1 }}
                      whileHover={{ rotate: showThink ? 360 : 15, scale: 1.1, transition: { type: "spring", stiffness: 300, damping: 10 } }}
                      transition={{ type: "spring", stiffness: 260, damping: 25 }}
                    >
                      <BrainCog className={cn("w-4 h-4", showThink ? "text-[#8B5CF6]" : "text-inherit")} />
                    </motion.div>
                  </div>
                  <AnimatePresence>
                    {showThink && (
                      <motion.span
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: "auto", opacity: 1 }}
                        exit={{ width: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="text-xs overflow-hidden whitespace-nowrap text-[#8B5CF6] flex-shrink-0"
                      >
                        Think
                      </motion.span>
                    )}
                  </AnimatePresence>
                </button>

                <CustomDivider />

                {/* Canvas */}
                <button
                  type="button"
                  onClick={handleCanvasToggle}
                  disabled={disabled || isLoading}
                  className={cn(
                    "rounded-full transition-all flex items-center gap-1 px-2 py-1 border h-8",
                    showCanvas
                      ? "bg-[#F97316]/15 border-[#F97316] text-[#F97316]"
                      : "bg-transparent border-transparent text-[#9CA3AF] hover:text-[#D1D5DB] disabled:opacity-50"
                  )}
                  aria-pressed={showCanvas}
                  aria-label="Toggle canvas mode"
                >
                  <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
                    <motion.div
                      animate={{ rotate: showCanvas ? 360 : 0, scale: showCanvas ? 1.1 : 1 }}
                      whileHover={{ rotate: showCanvas ? 360 : 15, scale: 1.1, transition: { type: "spring", stiffness: 300, damping: 10 } }}
                      transition={{ type: "spring", stiffness: 260, damping: 25 }}
                    >
                      <FolderCode className={cn("w-4 h-4", showCanvas ? "text-[#F97316]" : "text-inherit")} />
                    </motion.div>
                  </div>
                  <AnimatePresence>
                    {showCanvas && (
                      <motion.span
                        initial={{ width: 0, opacity: 0 }}
                        animate={{ width: "auto", opacity: 1 }}
                        exit={{ width: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="text-xs overflow-hidden whitespace-nowrap text-[#F97316] flex-shrink-0"
                      >
                        Canvas
                      </motion.span>
                    )}
                  </AnimatePresence>
                </button>
              </div>
            </div>

            {/* Main Action Button */}
            <PromptInputAction
              tooltip={
                isLoading
                  ? "Stop generation"
                  : isRecording
                  ? "Stop recording"
                  : hasContent
                  ? "Send message"
                  : "Voice message"
              }
            >
              <Button
                variant="default"
                size="icon"
                className={cn(
                  "h-8 w-8 rounded-full transition-all duration-200",
                  isRecording
                    ? "bg-transparent hover:bg-gray-600/30 text-red-500 hover:text-red-400"
                    : hasContent
                    ? "bg-white hover:bg-white/80 text-[#1F2023]"
                    : "bg-transparent hover:bg-gray-600/30 text-[#9CA3AF] hover:text-[#D1D5DB]"
                )}
                onClick={() => {
                  if (isRecording) setIsRecording(false);
                  else if (hasContent) handleSubmit();
                  else setIsRecording(true);
                }}
                disabled={(isLoading && !hasContent) || disabled}
                aria-label={
                  isLoading
                    ? "Stop generation"
                    : isRecording
                    ? "Stop recording"
                    : hasContent
                    ? "Send message"
                    : "Start voice recording"
                }
              >
                {isLoading ? (
                  <Square className="h-4 w-4 fill-[#1F2023] animate-pulse" />
                ) : isRecording ? (
                  <StopCircle className="h-5 w-5 text-red-500" />
                ) : hasContent ? (
                  <ArrowUp className="h-4 w-4 text-[#1F2023]" />
                ) : (
                  <Mic className="h-5 w-5 text-[#1F2023] transition-colors" />
                )}
              </Button>
            </PromptInputAction>
          </PromptInputActions>
        </PromptInput>

        <ImageViewDialog imageUrl={selectedImage} onClose={() => setSelectedImage(null)} />
      </>
    );
  }
);
PromptInputBox.displayName = "PromptInputBox";

export default PromptInputBox;