#!/usr/bin/env python3
"""Check PROXY_CONFIG_FILE (.json or Webshare .txt): HTTPS GET through each proxy."""

import argparse
import os
import sys
from pathlib import Path

import requests

from desktop_env.providers.aws.proxy_pool import ProxyInfo, ProxyPool


def _default_config_path() -> str:
    if os.environ.get("PROXY_CONFIG_FILE"):
        return os.environ["PROXY_CONFIG_FILE"]
    gui_root = Path(__file__).resolve().parents[1]
    roll = gui_root.parent / "dart_rollouter" / "evaluation_examples" / "settings" / "proxy"
    for name in ("webshare.secrets.txt", "webshare.txt"):
        p = roll / name
        if p.is_file():
            return str(p)
    return str(gui_root / "evaluation_examples" / "settings" / "proxy" / "dataimpulse.json")


def _try_proxy(proxy: ProxyInfo, url: str, timeout: float) -> tuple[bool, str]:
    if proxy.username and proxy.password:
        proxy_url = f"{proxy.protocol}://{proxy.username}:{proxy.password}@{proxy.host}:{proxy.port}"
    else:
        proxy_url = f"{proxy.protocol}://{proxy.host}:{proxy.port}"
    proxies = {"http": proxy_url, "https": proxy_url}
    try:
        r = requests.get(url, proxies=proxies, timeout=timeout)
        if r.status_code == 200:
            return True, r.text[:120].replace("\n", " ")
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, repr(e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=None,
        help="Proxy file (JSON or Webshare .txt). Default: PROXY_CONFIG_FILE env, else dart_rollouter/.../webshare*.txt",
    )
    ap.add_argument(
        "--url",
        default="https://api.ipify.org?format=json",
        help="URL to fetch through each proxy (must allow GET, 200 = ok)",
    )
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()
    config_path = args.config or _default_config_path()

    if not os.path.isfile(config_path):
        print(f"error: file not found: {config_path}", file=sys.stderr)
        return 2

    pool = ProxyPool(config_path)
    if not pool.proxies:
        print("error: no proxies loaded (empty file or parse error)", file=sys.stderr)
        return 2

    print(f"config={config_path!r} proxies={len(pool.proxies)} url={args.url!r}\n")
    ok_n = 0
    for i, proxy in enumerate(pool.proxies, 1):
        ok, detail = _try_proxy(proxy, args.url, args.timeout)
        if ok:
            ok_n += 1
            print(f"[{i}] OK  {proxy.host}:{proxy.port}  {detail}")
        else:
            print(f"[{i}] FAIL {proxy.host}:{proxy.port}  {detail}")

    print(f"\n{ok_n}/{len(pool.proxies)} passed")
    return 0 if ok_n == len(pool.proxies) else 1


if __name__ == "__main__":
    sys.exit(main())
