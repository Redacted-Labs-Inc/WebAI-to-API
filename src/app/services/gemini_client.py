# src/app/services/gemini_client.py
from app.config import CONFIG
from app.logger import logger
from app.utils.browser import get_cookie_from_browser

# Try to import Gemini dependencies (may not be available on ARM)
try:
    from models.gemini import MyGeminiClient
    from gemini_webapi.exceptions import AuthError
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    MyGeminiClient = None
    AuthError = Exception

# Global variable to store the Gemini client instance
_gemini_client = None

async def init_gemini_client() -> bool:
    """
    Initialize and set up the Gemini client based on the configuration.
    Returns True on success, False on failure.
    """
    global _gemini_client
    
    if not GEMINI_AVAILABLE:
        logger.info("Gemini dependencies not installed. Gemini API will not be available.")
        return False
    
    if CONFIG.getboolean("EnabledAI", "gemini", fallback=True):
        try:
            gemini_cookie_1PSID = CONFIG["Cookies"].get("gemini_cookie_1PSID")
            gemini_cookie_1PSIDTS = CONFIG["Cookies"].get("gemini_cookie_1PSIDTS")
            gemini_proxy = CONFIG["Proxy"].get("http_proxy")
            if not gemini_cookie_1PSID or not gemini_cookie_1PSIDTS:
                cookies = get_cookie_from_browser("gemini")
                if cookies:
                    gemini_cookie_1PSID, gemini_cookie_1PSIDTS = cookies
            
            if gemini_proxy == "":
                gemini_proxy = None
            
            if gemini_cookie_1PSID and gemini_cookie_1PSIDTS:
                _gemini_client = MyGeminiClient(secure_1psid=gemini_cookie_1PSID, secure_1psidts=gemini_cookie_1PSIDTS, proxy=gemini_proxy)
                await _gemini_client.init()
                return True
            else:
                logger.warning("Gemini cookies not found. Gemini API will not be available.")
                return False

        except AuthError as e:
            logger.error(
                f"Gemini authentication or connection failed: {e}. "
                "This could be due to expired cookies or a temporary network issue with Google's servers."
            )
            _gemini_client = None
            return False
            
        except Exception as e:
            logger.error(f"An unexpected error occurred while initializing Gemini client: {e}", exc_info=True)
            _gemini_client = None
            return False
    else:
        logger.info("Gemini client is disabled.")
        return False


def get_gemini_client():
    """
    Returns the initialized Gemini client instance.
    """
    return _gemini_client

