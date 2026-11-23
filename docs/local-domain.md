# Serving the project at a local domain

The app is configured to run at **`http://wepr.localhost:8000`**.

## Why not `wepr.dev`

`.dev` is a real top-level domain owned by Google, and **the entire TLD is on the HSTS
preload list** built into every major browser. Any `https`-capable browser will silently
rewrite `http://wepr.dev` to `https://wepr.dev` and refuse to load it without a valid
TLS certificate. You would need to generate and trust a local certificate authority
before it worked at all.

`.localhost` and `.test` are both reserved by [RFC 6761][rfc] for exactly this purpose,
carry no HSTS policy, and work over plain HTTP.

[rfc]: https://www.rfc-editor.org/rfc/rfc6761

| Hostname | HTTPS forced | Needs `/etc/hosts` | Verdict |
|---|---|---|---|
| `wepr.localhost` | no | only for Safari / curl | **chosen** |
| `wepr.test` | no | yes | fine alternative |
| `wepr.dev` | **yes** | yes | needs a local cert |

## Setup

Chrome and Firefox resolve any `*.localhost` name to `127.0.0.1` internally, so **in
those browsers it already works** — no setup at all.

macOS's system resolver does not, so Safari, `curl` and `ping` will fail until you add
one line to `/etc/hosts`. This needs your password, so run it yourself:

```bash
echo '127.0.0.1 wepr.localhost' | sudo tee -a /etc/hosts
```

Verify:

```bash
ping -c1 wepr.localhost
```

## Running the server

```bash
.venv/bin/python run.py
```

Then open <http://wepr.localhost:8000>.

Port **8000** rather than Flask's default 5000: macOS Monterey and later run the AirPlay
Receiver service on 5000, which intermittently steals the port and produces a confusing
"address already in use" failure.

## If you later want `wepr.dev` specifically

It is achievable with a locally trusted certificate:

```bash
brew install mkcert
mkcert -install
mkcert wepr.dev
echo '127.0.0.1 wepr.dev' | sudo tee -a /etc/hosts
```

Then pass the generated `.pem` pair to Flask via `ssl_context` and serve on 443. This is
extra moving parts for no functional gain locally, so it is not the default — but it is
the right approach if you ever want the local setup to mirror a production HTTPS
deployment.
