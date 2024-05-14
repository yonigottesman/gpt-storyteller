import asyncio
import io
import json
import os
import tempfile

import ffmpeg
from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI

app = FastAPI()


templates = Jinja2Templates(directory="templates")


SYSTEM_PROMPT = "You make up cool stories in hebrew"
MODEL = "gpt-4o"


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


async def send_texts(websocket: WebSocket, client: AsyncOpenAI, title: str, gist: str):
    prompt = """ Here is the title and gist of a story in hebrew. You need to write/finish the story,
            make it about 20-50 lines long.
            title: {title}
            gist: {gist}
            """
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt.format(title=title, gist=gist),
            }
        ],
        temperature=0,
        stream=True,
    )
    async for chunk in response:
        text = chunk.choices[0].delta.content
        await websocket.send_json({"type": "text", "value": text})


async def send_image_url(websocket: WebSocket, client: AsyncOpenAI, title: str, gist: str, audio_text: str):
    prompt = """
    generate an image for this short story. The image should contain the scene from the story. no text.:
    title: {title}
    story_gist: {gist}
    requested_story: {audio_text}
    """
    dalle_response = await client.images.generate(
        model="dall-e-3",
        prompt=prompt.format(title=title, gist=gist, audio_text=audio_text),
        size="1024x1024",
        quality="standard",
        n=1,
    )
    await websocket.send_json({"type": "image", "value": dalle_response.data[0].url})


async def get_text_from_audio(client: AsyncOpenAI, audio: UploadFile, language: str):

    if audio.content_type == "audio/webm;codecs=opus":
        data = await audio.read()
        f = io.BytesIO(data)
        f.name = "audio.webm"
        whisper_output = await client.audio.transcriptions.create(model="whisper-1", file=f, language="he")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_input = os.path.join(tmpdir, audio.filename)
            with open(mp4_input, "wb") as f:
                f.write(await audio.read())
            webm_output = os.path.join(tmpdir, "audio.webm")
            ffmpeg.input(mp4_input).output(
                webm_output,
                **{"lossless": 1, "vcodec": "libvpx-vp9", "acodec": "libopus", "crf": 30, "b:v": 0, "b:a": "192k"},
            ).run()

            with open(webm_output, "rb") as f:
                whisper_output = await client.audio.transcriptions.create(model="whisper-1", file=f, language="he")

    print(f"whisper output: {whisper_output.text[:100]}")
    return whisper_output


async def speech2text(client: AsyncOpenAI, audio_data: bytes, mimetype: str):
    print(f"mimetype: {mimetype}")
    if mimetype in ["audio/webm;codecs=opus", ""]:
        f = io.BytesIO(audio_data)
        f.name = "audio.webm"
        whisper_output = await client.audio.transcriptions.create(model="whisper-1", file=f, language="he")
    else:
        print("Converting audio")
        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_input = os.path.join(tmpdir, "file.name")
            with open(mp4_input, "wb") as f:
                f.write(audio_data)
            webm_output = os.path.join(tmpdir, "audio.webm")
            ffmpeg.input(mp4_input).output(
                webm_output,
                **{"lossless": 1, "vcodec": "libvpx-vp9", "acodec": "libopus", "crf": 30, "b:v": 0, "b:a": "192k"},
            ).run()

            with open(webm_output, "rb") as f:
                whisper_output = await client.audio.transcriptions.create(model="whisper-1", file=f, language="he")
    print(f"whisper output: {whisper_output.text}")
    return whisper_output.text


async def story_title_gist(client: AsyncOpenAI, audio_text: str):
    prompt = """
    here is a user request in hebrew for a short story. return a json with 'title' and 'snippet'.
    the snippet should be about 30 words long and only contain the main idea and scenery of the story. It will be used to create an
    image of the story so should contain any visual aspects of the story.
    USER_REQUEST: {text}
    """
    response = await client.chat.completions.create(
        response_format={"type": "json_object"},
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt.format(text=audio_text),
            },
        ],
    )
    title_gist = json.loads(response.choices[0].message.content)
    return title_gist


@app.websocket("/websocket_endpoint")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = AsyncOpenAI()
    try:
        while True:
            audio_mime = await websocket.receive_text()
            audio_data = await websocket.receive_bytes()
            audio_text = await speech2text(client, audio_data, audio_mime)
            await websocket.send_json({"type": "audio_text", "value": audio_text})
            title_gist = await story_title_gist(client, audio_text)
            await websocket.send_json({"type": "title", "value": title_gist["title"]})
            text_task = asyncio.create_task(send_texts(websocket, client, title_gist["title"], title_gist["snippet"]))
            image_url_task = asyncio.create_task(
                send_image_url(websocket, client, title_gist["title"], title_gist["snippet"], audio_text)
            )
            await asyncio.wait([text_task, image_url_task])
            await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        print("Client disconnected")
