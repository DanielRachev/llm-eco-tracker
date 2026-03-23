import asyncio
import functools
import inspect
import logging
import threading


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def carbon_aware(max_delay_hours=2, location="NL", mock_csv=None):
    """
    A decorator that intercepts function execution until energy on the power grid is greener. # TODO: explain greener
    
    Args:
        max_delay_hours (int): Maximum time to wait for a greener grid.
        location (str): Region code. # TODO: decide how the region will be represented, maybe use an enum or an accepted format
        mock_csv (str | None): Optional CSV path for mocked forecast data during development.
    """
    def _log_intercept(func):
        logger.info(f"Intercepting call to '{func.__name__}'")
        logger.info(f"Target Location: {location}, Max Delay: {max_delay_hours}h")
        if mock_csv:
            logger.info(f"Using mock forecast data from: {mock_csv}")

    def _get_delay_seconds():
        # TODO: actually calculate the delay based on the grid energy
        return 5

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                _log_intercept(func)
                delay_seconds = _get_delay_seconds()
                logger.info(
                    f"Carbon-aware scheduler: Awaiting {delay_seconds} seconds to reach optimal grid energy..."
                )
                await asyncio.sleep(delay_seconds)
                logger.info("Optimal energy reached. Proceeding with execution.")
                return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            _log_intercept(func)
            delay_seconds = _get_delay_seconds()
            logger.info(
                f"Carbon-aware scheduler: Scheduling execution in {delay_seconds} seconds to reach optimal grid energy..."
            )

            done = threading.Event()
            outcome = {}

            def run_later():
                try:
                    logger.info("Optimal energy reached. Proceeding with execution.")
                    outcome["result"] = func(*args, **kwargs)
                except BaseException as exc:
                    outcome["exception"] = exc
                finally:
                    done.set()

            timer = threading.Timer(delay_seconds, run_later)
            timer.start()

            try:
                done.wait()
            except KeyboardInterrupt:
                timer.cancel()
                raise

            if "exception" in outcome:
                raise outcome["exception"]

            return outcome.get("result")

        return sync_wrapper
    return decorator
