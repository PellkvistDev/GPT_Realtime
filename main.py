import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from fastapi.websockets import WebSocketDisconnect

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 8000))

SYSTEM_MESSAGE = (
    "Du är en svensk telefonförsäljare som ringer ett cold call. Du heter Fabian och ringer från Supernormal Health. "
    "Du erbjuder hjälp med viktnedgång, kost och medicin. Sälj ett abonnemang för 3000 kr/mån. Svara kort, naturligt och vänligt."
    "Din röst ska vara engagerad och låta som en proffesionell säljare. Du är en glad person och vill verkligen sälja."
    "Använd alla säljtrick du kan för att övertala kunden. För att sälja så ska du boka in ett möte när som helst, där ni kan prata om yttligre detaljer."
)

VOICE = 'alloy'

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def incoming_call(request: Request):
    """TwiML for handling Twilio call"""
    host = request.url.hostname
    response = VoiceResponse()
    response.say("Tack. Ett ögonblick.")
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    print("Twilio client connected")
    await websocket.accept()

    extra_headers = [
        ("Authorization", f"Bearer {OPENAI_API_KEY}"),
        ("OpenAI-Beta", "realtime=v1"),
    ]

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview-2024-12-17',
         extra_headers=extra_headers
    ) as openai_ws:
        await setup_openai_session(openai_ws)

        stream_sid = None
        latest_timestamp = 0
        last_item_id = None
        mark_queue = []
        response_start_time = None

        async def receive_from_twilio():
            nonlocal stream_sid, latest_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Stream started: {stream_sid}")
                    elif data['event'] == 'media':
                        latest_timestamp = int(data['media']['timestamp'])
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }))
                    elif data['event'] == 'mark' and mark_queue:
                        mark_queue.pop(0)
            except WebSocketDisconnect:
                print("Twilio disconnected.")
                await openai_ws.close()

        async def send_to_twilio():
            nonlocal last_item_id, response_start_time
            try:
                async for message in openai_ws:
                    data = json.loads(message)

                    if data.get("type") == "response.audio.delta":
                        payload = base64.b64encode(base64.b64decode(data['delta'])).decode('utf-8')
                        await websocket.send_json({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": payload}
                        })

                        if response_start_time is None:
                            response_start_time = latest_timestamp

                        if data.get("item_id"):
                            last_item_id = data["item_id"]
                            await send_mark(websocket, stream_sid)

                    elif data.get("type") == "input_audio_buffer.speech_started":
                        await handle_speech_interrupt(openai_ws, websocket, stream_sid, last_item_id, latest_timestamp, response_start_time)
                        last_item_id = None
                        response_start_time = None
            except Exception as e:
                print("Error:", e)

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def setup_openai_session(ws):
    """Initialize the OpenAI realtime session with fast response settings."""
    session_config = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_MESSAGE,
            "voice": VOICE,
            "modalities": ["text", "audio"],
            "temperature": 0.7,
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {"type": "server_vad"}
        }
    }
    await ws.send(json.dumps(session_config))

    greeting = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Tjena, har du en minut?"}]
        }
    }
    await ws.send(json.dumps(greeting))
    await ws.send(json.dumps({"type": "response.create"}))

async def send_mark(connection, sid):
    mark = {
        "event": "mark",
        "streamSid": sid,
        "mark": {"name": "responsePart"}
    }
    await connection.send_json(mark)

async def handle_speech_interrupt(openai_ws, websocket, sid, item_id, latest_ts, start_ts):
    if not item_id or not start_ts:
        return
    elapsed = latest_ts - start_ts
    await openai_ws.send(json.dumps({
        "type": "conversation.item.truncate",
        "item_id": item_id,
        "content_index": 0,
        "audio_end_ms": elapsed
    }))
    await websocket.send_json({
        "event": "clear",
        "streamSid": sid
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
