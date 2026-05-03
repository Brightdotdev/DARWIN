# 📄 Document Parsing Pipeline — README

## Overview

This module is a command-line document parsing pipeline designed to:

* Scan files or directories
* Parse supported document formats into structured text
* Optionally extract and caption images
* Save outputs into a `.darwin`-style storage system
* Provide real-time progress updates (optionally via WebSocket)

It is built for performance using multiprocessing and supports both **fast** and **precise** parsing modes.

---

## 🚀 How to Run

From the root of your project:

```bash
python main.py [path] [options]
```

If no path is provided, it defaults to the current directory (`.`).

---

## 📥 Supported File Types

* `.pdf`
* `.docx`
* `.pptx`
* `.txt`
* `.md`
* `.json`

Unsupported formats are skipped automatically.

---

## ⚙️ Command-Line Arguments

### Core Arguments

| Argument       | Description                                             |
| -------------- | ------------------------------------------------------- |
| `path`         | File or directory to parse (default: current directory) |
| `--file`, `-f` | Parse a single file instead of a directory              |

---

### Processing Behavior

| Argument            | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `--recursive`, `-r` | Scan directories recursively                               |
| `--rerun`, `-R`     | Reprocess files even if already cached                     |
| `--precise`         | Use high-accuracy parsing (slower, better layout handling) |

---

### Image Handling

| Argument           | Description                                        |
| ------------------ | -------------------------------------------------- |
| `--images`, `-img` | Extract images from documents                      |
| `--caption`, `-c`  | Generate captions for images (requires `--images`) |

⚠️ Note: Captioning increases processing time significantly.

---

### Output Control

| Argument    | Description                               |
| ----------- | ----------------------------------------- |
| `--no-save` | Do not save parsed results (preview only) |

---

### WebSocket / Live Updates

| Argument    | Description                               |
| ----------- | ----------------------------------------- |
| `--ws-port` | Port for WebSocket server (default: 8765) |
| `--no-ws`   | Disable WebSocket server                  |

---

## 🧠 Processing Modes

### Fast Mode (default)

* Uses standard parsing
* Optimized for speed
* May lose layout fidelity in complex PDFs

### Precise Mode (`--precise`)

* Uses advanced layout-aware parsing
* Better for structured documents (tables, columns)
* Slower

---

## ⚡ Performance Details

* Uses `ProcessPoolExecutor` for parallel processing
* Worker count:

  * Full CPU usage normally
  * Reduced when image captioning is enabled (to avoid overload)

Files are processed in **ascending size order** for efficiency.

---

## 📊 What Happens During Execution

### 1. Input Resolution

* Determines whether you're parsing a file or directory
* Validates existence and file type

---

### 2. File Scanning

* Collects supported files
* Applies recursion if enabled

---

### 3. Cache Check

* Skips files already parsed unless `--rerun` is used

---

### 4. Parsing Begins

* Progress bar starts
* Files are distributed across worker processes

---

### 5. Per-File Processing

Each file:

* Gets parsed into:

  * `display_text` (human-readable)
  * `embedding_text` (optimized for embeddings)
* Extracts page count
* Handles errors gracefully

---

### 6. Logging + Progress Updates

You’ll see logs like:

```
Queueing 5 files (12.3 MB) [fast mode]
Processing 5 files with 8 workers…

file1.pdf — 10 pages · 25,000 chars
file2.docx — 3 pages · 5,200 chars
file3.pdf — ERROR: ValueError: corrupted file
```

---

### 7. Saving Results

If saving is enabled:

* Parsed documents are stored via `DarwinStore`
* Metadata (like page count) is registered

---

### 8. Final Summary

Example:

```
Total time: 8.2s
Successfully parsed 4 file(s)
Failed to parse 1 file(s)
```

---

## 📦 Output Structure

Each parsed document (`ParsedDocument`) contains:

```python
{
    display_text: str,
    embedding_text: str,
    filename: str,
    page_count: int,
    file_type: str,
    error: Optional[str]
}
```

---

## ❌ Error Handling

* Errors do NOT stop the pipeline
* Each file failure is logged and returned with an error field
* Example:

```
file.pdf - ValueError: Invalid PDF structure
```

---

## 🧩 Internal Components

* `parse_documents()` → orchestrates multiprocessing
* `parse_document()` → worker function per file
* `handle_parsing()` → main logic controller
* `DarwinStore` → handles persistence + caching
* `scan_directory()` → discovers files

---

## 🔌 WebSocket Behavior

* Starts automatically unless `--no-ws` is set
* Used for real-time progress updates (UI integration)
* Only one server instance runs

---

## 🧪 Example Commands

### Parse current directory

```bash
python main.py
# or with uv
uv run main.py
```

### Parse a specific file

```bash
python main.py --file report.pdf
# or with uv
uv run main.py --file report.pdf
```

### Recursive parsing with images

```bash
python main.py ./docs -r --images
# or with uv
uv run main.py ./docs -r --images
```

### Full processing with captions + precise mode

```bash
python main.py ./docs -r --images --caption --precise
# or with uv
uv run main.py ./docs -r --images --caption --precise
```

### Re-run everything without saving

```bash
python main.py ./docs -R --no-save
# or with uv
uv run main.py ./docs -R --no-save
```

---

## ⚠️ What to Expect

* Small files finish almost instantly
* Large PDFs or image-heavy docs will take longer
* Captioning images can slow things down heavily
* Memory usage scales with number of workers
* Some files may fail. That’s normal

---

## 🧯 Common Pitfalls

* Forgetting `--images` but using `--caption`
* Running without `--rerun` and wondering why nothing happens
* Parsing unsupported file types
* Overloading CPU with captioning on large batches

---

## ✅ Summary

This pipeline is:

* Fast by default
* Scalable via multiprocessing
* Fault-tolerant
* Flexible via CLI flags

If you want speed, stick to default settings. If you want accuracy, turn on `--precise`. If you want everything, expect it to take time.

---

If you want, I can also break down how to extend this pipeline or plug it into an API.
