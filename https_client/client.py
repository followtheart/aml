#!/usr/bin/env python3
"""HTTPS experiment client; Python standard library only."""
import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward a Bearer token to a redirected destination.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://localhost:8443")
    parser.add_argument("--ca", help="Trusted PEM certificate (required for self-signed TLS)")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--user-id", help="Default: unique experiment user")
    parser.add_argument("--text", default="I started photography in March 2024.")
    parser.add_argument("--query", default="When did I start photography?")
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.url)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        parser.error("--url must be an https:// URL without credentials, query or fragment")
    token = os.environ.get("AML_API_KEY", "")
    if not args.health_only and not token:
        parser.error("Set AML_API_KEY to the server's API token")
    context = ssl.create_default_context(cafile=args.ca)
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context), NoRedirect())

    def request(path, body=None):
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            headers.update({"Content-Type": "application/json",
                            "Authorization": f"Bearer {token}"})
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(args.url.rstrip("/") + path, data=data, headers=headers)
        with opener.open(req, timeout=args.timeout) as response:
            result = json.load(response)
        print(path, json.dumps(result, ensure_ascii=False, indent=2))
        return result

    request("/health")
    if args.health_only:
        return
    run = uuid.uuid4().hex
    user = args.user_id or f"https-experiment:{run}"
    added = request("/add", {"request_id": f"https:{run}", "user_id": user,
                             "session_id": f"https:{run}",
                             "messages": [{"role": "user", "content": args.text}]})
    if added.get("success") is not True:
        raise ValueError("Add did not report success")
    found = request("/search", {"user_id": user, "query": args.query, "top_k": 10,
                                "min_revision": added["write_revision"]})
    if not found.get("data"):
        raise ValueError("Search returned no memories; experiment failed")
    print(f"PASS: HTTPS add/search completed; user_id={user}")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        sys.exit(1)
