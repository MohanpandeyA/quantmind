"""SSL fix for macOS LibreSSL compatibility with yfinance/curl_cffi.

macOS ships with LibreSSL 2.8.3 which is incompatible with
yfinance's curl_cffi library (SSL certificate verification fails).

The fix: monkey-patch curl_cffi.requests.Session.request() to
pass verify=False by default. This must be applied BEFORE yfinance
is imported anywhere.

This is a LOCAL DEVELOPMENT workaround only.
In production (Linux servers with OpenSSL), this is not needed.

Usage — import at the very top of api/main.py:
    from engine.ssl_fix import apply_ssl_fix
    apply_ssl_fix()
"""

from __future__ import annotations

import warnings

from config.logging_config import get_logger

logger = get_logger(__name__)

_ssl_fix_applied = False


def apply_ssl_fix() -> None:
    """Patch curl_cffi to disable SSL verification on macOS LibreSSL.

    Monkey-patches curl_cffi.requests.Session.request() to pass
    verify=False by default. This resolves the LibreSSL 2.8.3
    incompatibility without requiring any environment variables.

    Must be called BEFORE yfinance is imported.
    Safe to call multiple times — only applies once.
    """
    global _ssl_fix_applied
    if _ssl_fix_applied:
        return

    # Suppress urllib3 NotOpenSSLWarning
    try:
        from urllib3.exceptions import NotOpenSSLWarning
        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    except Exception:
        pass

    # Suppress all InsecureRequestWarnings
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass

    # Patch curl_cffi.requests.Session.request to disable SSL verification
    try:
        import curl_cffi.requests as curl_req

        _orig_request = curl_req.Session.request

        def _patched_request(self: object, method: str, url: str, **kwargs: object) -> object:
            """Patched request that disables SSL verification by default."""
            kwargs.setdefault("verify", False)
            return _orig_request(self, method, url, **kwargs)  # type: ignore[arg-type]

        curl_req.Session.request = _patched_request  # type: ignore[method-assign]
        logger.info(
            "SSL fix applied | curl_cffi.Session.request patched | "
            "verify=False by default | macOS LibreSSL 2.8.3 workaround"
        )

    except ImportError:
        logger.debug("curl_cffi not installed — SSL fix not needed")
    except Exception as e:
        logger.warning("SSL fix failed: %s", e)

    _ssl_fix_applied = True
