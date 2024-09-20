## Speech2Text API

This project contains API for Speech-to-text based on a [Whisper speech2text model](https://openai.com/blog/whisper/).


### Usage

1. Through the API:

```shell
curl 'localhost:5000/transcribe/' \
     -X 'POST' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@tests/res/test_cs_longer.wav'
```
returns a transcript in the default language (`cs`):
```json
{"transcript":" Já poruká si dáv za hrad písnicku, prasvý mančelku, mi chéli také proce rukestíku, také prasvě na davitka, také prasvě na to láška. Váte to otáte na pavátku,"}
```

```shell
curl 'localhost:5000/transcribe/{lang}' \
     -X 'POST' \
     -H 'accept: application/json' \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@tests/res/test_cs_longer.wav'
```
returns a transcript in a chosen langauge `{lang}`. See the list of [Whisper's supported languages (Appendix D2)](https://cdn.openai.com/papers/whisper.pdf) 
and the [encoding of language ids](https://huggingface.co/languages) by ISO 639. 

2. As a python library:

```python
from speech2text_api.whisper import Whisper
wrapper = Whisper(model_name_or_path='openai/whisper-medium')
transcript = wrapper.transcribe_file("tests/res/test_cs_longer.wav", lang="cs")

print(transcript)
```