"""Development entry point.

    .venv/bin/python run.py

Port 8000, not 5000: macOS Monterey's AirPlay Receiver can claim 5000. The local
hostname is `wepr.localhost` — see `docs/local-domain.md`. `PORT` overrides it, so
a second copy can run alongside the first without editing this file.

Named `run.py` rather than `app.py` because the application package is `app/`, and
a module and a package of the same name in the same directory shadow each other
confusingly.
"""

import os

from app import create_app

application = create_app()

if __name__ == "__main__":
    application.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), debug=True)
