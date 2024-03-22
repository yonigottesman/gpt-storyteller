import io
import json
import os
import tempfile

import ffmpeg
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import AsyncOpenAI

app = FastAPI()


templates = Jinja2Templates(directory="templates")


SYSTEM_PROMPT = "You make up cool and educational stories for children aged 7-16 in hebrew. "
MODEL = "gpt-4-turbo-preview"

USER_PREFIX_PROMPT = """ Here is the beginning or the main idea of a story in hebrew. You need to write/finish the story,
make it about 20-50 lines long. return a json with 'title' and 'text'.
text: {text}
"""


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


@app.post("/audio/")
async def audio(audioFile: UploadFile = File(...)):
    client = AsyncOpenAI()
    text = await get_text_from_audio(client, audioFile, "he")

    response = await client.chat.completions.create(
        response_format={"type": "json_object"},
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PREFIX_PROMPT.format(text=text),
            },
        ],
    )
    story = json.loads(response.choices[0].message.content)
    dalle_response = await client.images.generate(
        model="dall-e-3",
        prompt=f"generate an image for this short story:\n{story['text']}",
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return {
        "text": story["text"],
        "title": story["title"],
        "image_url": dalle_response.data[0].url,
    }


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
