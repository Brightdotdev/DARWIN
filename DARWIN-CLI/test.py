import asyncio
import websockets
import json

async def handler(ws):
    print("Client connected!")
    await ws.send(json.dumps({
        "type": "log",
        "payload": {"message": "Hello from test server!", "log_type": "success", "style": ""}
    }))
    # Keep alive
    try:
        async for msg in ws:
            print(f"Received: {msg}")
    except:
        pass

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Test server on ws://localhost:8765")
        await asyncio.Future()

asyncio.run(main())