import logging
import os
import tempfile

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from sentry_asgi import SentryMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from speech2text_api.settings import settings
from speech2text_api.whisper import Whisper

logger = logging.getLogger(__name__)


class HelloResponse(BaseModel):
    message: str


class TranscriptResponse(BaseModel):
    transcript: str


def get_app() -> FastAPI:
    app = FastAPI(
        title=settings.title,
        description=settings.description,
        debug=settings.debug_api,
        openapi_url=settings.openapi_route,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    app.add_middleware(SentryMiddleware)
    return app


app = get_app()

INDEX_HTML = """
<h1>Speech2Text API</h1>
<p>
<a href="docs">API documentation</a>
</p>
""".lstrip()


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


# speech2text_wrapper = Whisper('openai/whisper-medium')  # TODO: selected model is big, pick the model below for testing
speech2text_wrapper = Whisper(settings.model_id)


async def create_tmp_file(file: UploadFile, output_fname: str = "input.wav") -> str:
    tmp_dir = tempfile.mkdtemp()
    tmp_fname = os.path.join(tmp_dir, output_fname)  # librosa uses suffix of the tempfile to determine format :/
    # Note that we've seen that librosa causes trouble with some .mp3
    with open(tmp_fname, "wb") as wav_tmp_f:
        wav_tmp_f.write(file.file.read())
    return tmp_fname


@app.post("/transcribe/", response_model=TranscriptResponse)
async def transcribe_default(file: UploadFile) -> TranscriptResponse:
    tmp_fname = await create_tmp_file(file)

    transcript = speech2text_wrapper.transcribe_file(tmp_fname)

    return TranscriptResponse(transcript=transcript)


@app.post("/transcribe/{lang}/", response_model=TranscriptResponse)
async def transcribe_chosen_lang(lang: str, file: UploadFile) -> TranscriptResponse:
    tmp_fname = await create_tmp_file(file)

    transcript = speech2text_wrapper.transcribe_file(tmp_fname, lang=lang)

    return TranscriptResponse(transcript=transcript)
