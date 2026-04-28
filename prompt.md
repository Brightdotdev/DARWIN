
You are designing a **decoupled, event-driven terminal system** called **Darwin Terminal**.

The system must separate:

1. **Core Event API (Darwin Logger)**
2. **CLI Renderer (Terminal UI)**
3. **Frontend Event Emitter (WebSocket/JSON bridge)**

The system must NEVER couple logic to rendering.

---

# 1. Core: Darwin Logger (Event System)

This is the only source of truth. It emits structured events only.

It must NOT print anything.

### Events

### log

```python
log(message: str, log_type: LogType = "info", style: str = "")
```

* Stateless event
* Used for messages
* log_type must be an enum:

  * INFO
  * WARNING
  * ERROR
  * SUCCESS

---

### progress

```python
progress_start(id: str, total: int, label: str)
progress_update(id: str, current: int, status: str = "")
progress_end(id: str)
```

* Stateful tracking by id
* No printing
* Only emits events

---

### prompt (blocking event)

```python
prompt(id: str, message: str, options: list, action: str, params: dict)
```

* Must pause execution until resolved
* When a response is received:

  * route to registered action:

```python
action(choice=..., **params)
```

---

### Action Registry

```python
register_action(name: str, func: callable)
```

* Stores callable actions
* Used by prompt resolution system

---

# 2. Terminal UI (CLI Renderer)

Create a separate module:

```python
terminalui.py
```

Responsibilities:

* Subscribes to Darwin Logger events
* Uses **Rich library**
* Renders:

  * logs
  * progress bars
  * prompts

Rules:

* Must NOT contain business logic
* Must NOT emit events
* Only renders received events

ASCII branding required:

```
██████╗  █████╗ ██████╗ ██╗    ██╗██╗███╗   ██╗
██╔══██╗██╔══██╗██╔══██╗██║    ██║██║████╗  ██║
██║  ██║███████║██████╔╝██║ █╗ ██║██║██╔██╗ ██║
██║  ██║██╔══██║██╔══██╗██║███╗██║██║██║╚██╗██║
██████╔╝██║  ██║██║  ██║╚███╔███╔╝██║██║ ╚████║
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚═╝  ╚═══╝
```

---

# 3. Frontend Emitter (Darwin Front Bridge)

Create:

```python
darwin_logger.py
```

Responsibilities:

* Converts all events into JSON
* Sends them over WebSocket (or stdout fallback)
* Used by frontend UI (React / web terminal)

### Event format (strict)

```json
{
  "type": "log | progress | prompt",
  "payload": {}
}
```

---

### WebSocket Mode

* Streams events in real time
* Used by frontend terminal UI

### Fallback Mode

* Prints JSON lines (NDJSON)

---

# Architecture Rules

* NEVER print inside core logic
* NEVER mix rendering with event generation
* Terminal UI and Web UI are just subscribers
* Everything flows through events only

---

# Required Design Output

Return full Python implementation for:

1. `darwin_logger.py` (core event system + actions + blocking prompts)
2. `terminalui.py` (Rich-based CLI renderer)
3. WebSocket JSON emitter inside `darwin_logger.py`

---

# Extra Requirement

Both CLI and frontend must display the same ASCII “Darwin” banner on startup.

