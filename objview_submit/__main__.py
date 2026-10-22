"""Run the local-development API. This CLI never binds an external interface."""
import argparse
import os
from pathlib import Path
import re
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-only', action='store_true', required=True,
                        help='Acknowledge local-only intake; no OAuth, builds or evaluation')
    parser.add_argument('--state-dir', type=Path, required=True, help='Private 0700 directory; parent must exist')
    parser.add_argument('--port', type=int, default=8090, help='127.0.0.1 port (default 8090)')
    args = parser.parse_args()
    token = os.environ.get('OBJVIEW_SUBMIT_ADMIN_TOKEN', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', token):
        parser.error('Set OBJVIEW_SUBMIT_ADMIN_TOKEN to a random URL-safe secret of at least 32 characters')
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    from .app import create_app
    from .store import Store
    import uvicorn
    try:
        store = Store(args.state_dir)
        app = create_app(store, token, development_mode=True)
    except (ValueError, OSError, sqlite3.Error):
        parser.error('State initialization failed; check private directory permissions and schema version')
    uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False, log_level='warning',
                proxy_headers=False, server_header=False, limit_concurrency=32, timeout_keep_alive=5)


if __name__ == '__main__':
    main()
