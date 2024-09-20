from utils import test_record_paths

from speech2text_api.whisper import Whisper


def test_whisper_cs() -> None:
    wrapper = Whisper()
    transcript = wrapper.transcribe_file(test_record_paths["cs"], lang="cs")

    assert len(transcript)
    assert "testovací" in transcript


def test_whisper_en() -> None:
    wrapper = Whisper()
    transcript = wrapper.transcribe_file(test_record_paths["en"], lang="en")

    assert len(transcript)
    assert "evacuation" in transcript.lower()
