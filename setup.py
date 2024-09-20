#!/usr/bin/env python3

from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="UTF-8") as f:
    long_description = f.read()

setup(
    name="speech2text-api",
    description="Speech2Text API",
    long_description=long_description,
    author="Michal Stefanik",
    author_email="stefanik.m@mail.muni.cz",
    packages=find_packages("src"),
    package_dir={"": "src"},
    use_scm_version={"write_to": ".version", "write_to_template": "{version}\n"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "click",
        "email-validator",
        "fastapi",
        "pydantic[dotenv]",
        "sentry-asgi",
        "sentry-sdk",
        "starlette",
        "statsd",
        "uvicorn",
        "torch",
        "transformers>=4.23.1",
        "python-multipart",
        "torchaudio",
        "librosa"
    ],
    entry_points={
        "console_scripts": ["speech2text-api=speech2text_api.__main__:main"]
    },
    package_data={"speech2text_api": ["py.typed"]},
    include_package_data=True,
    platforms=["platform-independent"],
)
