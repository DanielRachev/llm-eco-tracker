import time
import functools
import logging


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
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Intercepting call to '{func.__name__}'")
            logger.info(f"Target Location: {location}, Max Delay: {max_delay_hours}h")
            if mock_csv:
                logger.info(f"Using mock forecast data from: {mock_csv}")
            
            # TODO: actually calculate the delay based on the grid energy
            delay_seconds = 5
            logger.info(f"Carbon-aware scheduler: Pausing for {delay_seconds} seconds to reach optimal grid energy...")
            time.sleep(delay_seconds)
            
            logger.info("Optimal energy reached. Proceeding with execution.")
            return func(*args, **kwargs)
        return wrapper
    return decorator
