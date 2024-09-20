import abc


class Speech2Text(abc.ABC):

    @abc.abstractmethod
    def transcribe_file(self, fpath: str, lang: str) -> str:
        """
        Generates a transcription of the audio file and retrieves it as a string.
        :param fpath: A path to the persisted audio file to be transcribed.
        :param lang: Language of the generated transcript. ISO 639 code (https://huggingface.co/languages).
        :return a transcript of the referenced audio file.
        """
        pass
