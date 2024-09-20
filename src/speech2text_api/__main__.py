import logging
from typing import Optional

import click
import sentry_sdk

from speech2text_api import __version__
from speech2text_api.settings import settings

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(__version__)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=settings.verbose,
    help="Enable verbose logging",
)
@click.option(
    "--sentry-dsn",
    default=settings.sentry_dsn,
    help="Send errors to Sentry",
)
def cli(verbose: bool, sentry_dsn: Optional[str]) -> None:
    settings.verbose = verbose
    settings.sentry_dsn = sentry_dsn

    sentry_sdk.init(dsn=settings.sentry_dsn)


@cli.command(help="Start the HTTP API server.")
@click.option("--host", default=settings.host)
@click.option("--port", default=settings.port)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Automatically reload the server if the source code changes (useful for development).",
)
@click.option(
    "--debug-api", is_flag=True, default=settings.debug_api, help="Debug flag"
)
@click.option(
    "--log-level-uvicorn",
    default=settings.log_level_uvicorn,
    type=click.Choice(
        ["critical", "error", "warning", "info", "debug", "trace"], case_sensitive=False
    ),
)
def httpapi(
    host: str,
    port: int,
    reload: bool,
    debug_api: bool,
    log_level_uvicorn: str,
) -> None:
    import uvicorn

    settings.host = host
    settings.port = port
    settings.debug_api = debug_api
    settings.log_level_uvicorn = log_level_uvicorn

    logger.info("settings main: %s", settings.json(sort_keys=True))

    uvicorn.run(
        "speech2text_api.httpapi:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        access_log=settings.debug_api,
        log_level=settings.log_level_uvicorn,
        reload=reload,
    )


def main() -> None:
    cli()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sentry_sdk.capture_exception(error)
        raise
