# src/app/services/kagi_client.py
from models.kagi import KagiClient
from app.config import CONFIG
from app.logger import logger
from app.utils.browser import get_cookie_from_browser

_kagi_client = None


async def init_kagi_client() -> bool:
    """
    Initialize and set up the Kagi client based on the configuration.
    Returns True on success, False on failure.
    """
    global _kagi_client
    
    if not CONFIG.getboolean("EnabledAI", "kagi", fallback=False):
        logger.info("Kagi client is disabled in configuration.")
        return False
    
    try:
        kagi_session = CONFIG["Cookies"].get("kagi_session")
        kagi_proxy = CONFIG["Proxy"].get("http_proxy")
        
        if not kagi_session:
            cookies = get_cookie_from_browser("kagi")
            if cookies:
                kagi_session = cookies[0]
        
        if kagi_proxy == "":
            kagi_proxy = None
        
        if kagi_session:
            _kagi_client = KagiClient(session_token=kagi_session, proxy=kagi_proxy)
            await _kagi_client.init()
            return True
        else:
            logger.warning("Kagi session cookie not found. Kagi API will not be available.")
            return False
    
    except Exception as e:
        logger.error(f"Failed to initialize Kagi client: {e}", exc_info=True)
        _kagi_client = None
        return False


def get_kagi_client() -> KagiClient:
    """Returns the initialized Kagi client instance."""
    return _kagi_client

