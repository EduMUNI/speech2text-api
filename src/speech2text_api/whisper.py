import logging
from typing import Any, Dict, Iterator, List, Tuple, Union

import librosa
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    BatchEncoding,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
    WhisperTokenizer,
)
from transformers.pipelines import PIPELINE_REGISTRY

from speech2text_api.speech2text_abs import Speech2Text

logger = logging.getLogger()


class WhisperPipeline(AutomaticSpeechRecognitionPipeline):

    def __init__(self,
                 feature_extractor: Union["SequenceFeatureExtractor", str],  # type: ignore  # noqa: F821
                 *args: List[Any],
                 default_lang: str = "cs",
                 generation_kwargs: Dict[str, Any] = {},
                 **kwargs: Dict[str, Any]):
        hardcoded_params = {"feature_extractor": feature_extractor,
                            "tokenizer": WhisperTokenizer.from_pretrained(kwargs["model"].name_or_path)}
        super().__init__(*args, **{**hardcoded_params,
                                   **{k: v for k, v in kwargs.items() if k not in hardcoded_params}})
        self.generation_kwargs = generation_kwargs
        self.default_lang = default_lang

    def _sanitize_parameters(self, **kwargs: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
        # Override of default ASR Pipeline, that removes our "lang" parameter
        sanitize_params = {k: v for k, v in kwargs.items() if k not in ("lang", )}
        preproc_args, fwd_args, postproc_args = super()._sanitize_parameters(**sanitize_params)
        preproc_args = {**preproc_args, **kwargs}

        return preproc_args, fwd_args, postproc_args

    def preprocess(self,
                   inputs: Union[Dict[str, Any], BatchEncoding],
                   *args: List[Any],
                   **kwargs: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        # adds "lang" argument required by our Pipeline among the inputs.
        request_lang = kwargs.pop("lang") if "lang" in kwargs else self.default_lang
        inputs["lang"] = request_lang
        return super().preprocess(inputs, *args, **kwargs)  # type ignore: inferred type is incorrect

    def _get_forced_decoder_input_ids(self, target_lang: str) -> List[Tuple[int, int]]:
        # Our adjustment: extra generation args resolution:
        if target_lang is not None:
            lang_token_l = self.tokenizer('<|%s|>' % target_lang, add_special_tokens=False)

            # heuristic:
            # tokenization to more or less than one token means that the language id is not among the special symbols
            if len(lang_token_l["input_ids"]) != 1:
                raise ValueError("Given language <|%s|> not recognized. Pick one of: %s"
                                 % (target_lang, self.tokenizer.additional_special_tokens))

            # forcing the first token in generation assures that the model will stick to the desired language
            processor = WhisperProcessor(self.feature_extractor, self.tokenizer)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=target_lang, task="transcribe")
            return forced_decoder_ids

    def _forward(self, model_inputs: Union[Dict[str, Any], BatchEncoding]) -> Dict[str, str]:
        """
        Implementation of the main self.__call__() method.
        :param model_inputs: Model inputs must contain at least (1) "lang": str, (2) "input_features" or "input_values".

        """
        is_last = model_inputs.pop("is_last")
        # adjustment - specialization: removed condition: `if self.type == "seq2seq"`:  # Whisper is seq2seq

        # Consume values, so we can let extra information flow freely through
        # the pipeline (important for `partial` in microphone)
        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        # adjustment - specialization: removed condition if accepts_attention_mask:  # False for Whisper

        model_args = self.generation_kwargs.copy()
        # language resolution
        lang_id = model_inputs.pop("lang")
        model_args["forced_decoder_ids"] = self._get_forced_decoder_input_ids(lang_id)

        model_args["max_length"] = 384  # After transformers>=4.23.X, setting this to None breaks Whisper's pipeline

        tokens = self.model.generate(inputs, **model_args)

        out = {"tokens": tokens}

        # Leftover inputs
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}


class Whisper(Speech2Text):

    def __init__(self, model_name_or_path: str = 'openai/whisper-tiny'):
        # injects our own pipeline
        PIPELINE_REGISTRY.register_pipeline("asr-with-whisper",
                                            pipeline_class=WhisperPipeline,
                                            pt_model=WhisperForConditionalGeneration)
        if model_name_or_path == 'openai/whisper-tiny':
            logger.warning("You are using the default smallest whisper model, intended only for CI tests."
                           "Consider changing the default `model_name_or_path`, e.g., to `openai/whisper-medium`.")
        self.pipe = pipeline("asr-with-whisper", model_name_or_path)

    def transcribe_file_default(self, fpath: str, lang: str = "cs") -> str:
        # not used currently - works properly only for the <30sec files
        # kept here just for a demonstration
        prediction = self.pipe(fpath, lang=lang)
        return prediction["text"]

    def transcribe_file(self, fpath: str, lang: str = "cs") -> str:
        """
        Generates a transcription of the audio file and retrieves it as a string.
        :param fpath: A path to the persisted audio file to be transcribed.
        :param lang: Language of the generated transcript. ISO 639 code (https://huggingface.co/languages).
        :return a transcript of the referenced audio file.
        """
        np_seq, sr = librosa.load(fpath, sr=None)
        segment_size = self.pipe.feature_extractor.chunk_length  # 30 seconds is whisper's preprocessing default
        # TODO: do the chunking smarter: https://huggingface.co/blog/asr-chunking
        # segments split on each `segment_size` seconds
        segments = [np_seq[start:start + (segment_size * sr)] for start in range(0, len(np_seq), segment_size * sr)]

        predictions = []

        for sequence in segments:
            input_formatted = {"raw": sequence, "sampling_rate": sr}
            predictions.append(self.pipe(input_formatted, lang=lang)["text"])

        return " ".join(predictions)
