"""Download the price history (and optionally follow the live bars) of any TradingView symbol to CSV. Python standard library only: nothing to install.

    python tv_fetch.py CME_MINI:NQ1! --interval 1D --bars 5000
    python tv_fetch.py BITSTAMP:BTCUSD --interval 60 --bars 2000 --out data
    python tv_fetch.py CME_MINI:NQ1! --interval 5 --live            # history, then every bar update until Ctrl+C
    python tv_fetch.py TVC:SPX --interval 1D --all                  # the S&P 500 index from 1871, but only monthly points before 1950 (futures only exist from 1997: CME_MINI:ES1! --all)
                                                                    # (this script saves what the feed gives; tv_prices.py, which the Lab uses, cuts those monthly bars out)

The symbol is what the chart shows in its top-left corner as EXCHANGE:SYMBOL (click the symbol name to see the exchange). Intervals: 1 3 5 15 30 45 60 120 180 240 (minutes), 1D 1W 1M.

How it works: TradingView's own charts get their bars from a WebSocket feed (wss://data.tradingview.com). This script speaks the same protocol: it opens the socket, creates a chart session,
resolves the symbol and asks for a series of bars, then writes what comes back. It uses the anonymous (logged-out) token, so you get what a logged-out visitor's chart gets: for many
futures and stocks that is DELAYED data and a limited history, and only the bars (time, open, high, low, close, volume), never the values of indicators or strategies. For those, use the chart's
own "Download chart data" menu (needs a signed-in account).

Be aware: this is TradingView's unofficial, undocumented feed. Their terms of use do not allow automated collection of their data, and the protocol can change without notice. Use it for
your own research, with moderate volume."""
import argparse
import base64
import csv
import datetime as dt
import json
import os
import random
import re
import socket
import ssl
import string
import struct
import sys
import time

HOST = "data.tradingview.com"
PATH = "/socket.io/websocket?type=chart"
ORIGIN = "https://www.tradingview.com"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
ANONYMOUS_TOKEN = "unauthorized_user_token"
INTERVALS = ["1", "3", "5", "15", "30", "45", "60", "120", "180", "240", "1D", "1W", "1M"]


# ------------------------------------------------------------------------------------------------------------------------------------ a minimal WebSocket client (RFC 6455)
class WebSocket:
    def __init__(self, host: str, path: str, origin: str, timeout: float = 30.0):
        raw = socket.create_connection((host, 443), timeout=timeout)
        self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           f"Sec-WebSocket-Version: 13\r\nOrigin: {origin}\r\nUser-Agent: {USER_AGENT}\r\n\r\n").encode())
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("the server closed the connection during the handshake")
            data += chunk
        head, _, self.buffer = data.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode(errors="replace")
        if " 101 " not in status:
            raise ConnectionError(f"WebSocket handshake refused: {status}")

    def _read(self, n: int) -> bytes:
        while len(self.buffer) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("the server closed the connection")
            self.buffer += chunk
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out

    def _send_frame(self, opcode: int, payload: bytes):
        n = len(payload)
        header = bytes([0x80 | opcode])
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)                                         # a client must mask what it sends
        self.sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def send_text(self, text: str):
        self._send_frame(1, text.encode("utf-8"))

    def recv_text(self) -> str:
        message = b""
        while True:
            b1, b2 = self._read(2)
            fin, opcode, n = b1 & 0x80, b1 & 0x0F, b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(n)
            if opcode == 8:
                raise ConnectionError("the server closed the connection")
            if opcode == 9:                                          # ping -> pong
                self._send_frame(10, payload)
                continue
            if opcode == 10:
                continue
            message += payload
            if fin:
                return message.decode("utf-8", errors="replace")

    def close(self):
        try:
            self._send_frame(8, b"")
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------------------------------------------ TradingView's message format: ~m~<length>~m~<payload>
def pack(payload: str) -> str:
    return f"~m~{len(payload)}~m~{payload}"


def unpack(text: str) -> list:
    out, pos, header = [], 0, re.compile(r"~m~(\d+)~m~")
    while pos < len(text):
        m = header.match(text, pos)
        if not m:
            break
        start = m.end()
        out.append(text[start:start + int(m.group(1))])
        pos = start + int(m.group(1))
    return out


def random_id(prefix: str) -> str:
    return prefix + "".join(random.choice(string.ascii_lowercase) for _ in range(12))


class Feed:
    """One chart session on TradingView's data socket."""

    def __init__(self, timeout: float = 30.0):
        self.ws = WebSocket(HOST, PATH, ORIGIN, timeout)
        self.session = random_id("cs_")

    def send(self, name: str, params: list):
        self.ws.send_text(pack(json.dumps({"m": name, "p": params}, separators=(",", ":"))))

    def events(self):
        """Yields every decoded message of the next network read ({'m': name, 'p': params}); answers the server's heartbeats."""
        for payload in unpack(self.ws.recv_text()):
            if payload.startswith("~h~"):
                self.ws.send_text(pack(payload))                     # the heartbeat is echoed back as it came
                continue
            try:
                message = json.loads(payload)
            except ValueError:
                continue
            if isinstance(message, dict) and "m" in message:
                yield message

    def open_series(self, symbol: str, interval: str, count: int, session: str):
        self.send("set_auth_token", [ANONYMOUS_TOKEN])
        self.send("chart_create_session", [self.session, ""])
        self.send("switch_timezone", [self.session, "Etc/UTC"])
        self.send("resolve_symbol", [self.session, "sds_sym_1", "=" + json.dumps({"symbol": symbol, "adjustment": "splits", "session": session})])
        self.send("create_series", [self.session, "sds_1", "s1", "sds_sym_1", interval, int(count), ""])

    def close(self):
        self.ws.close()


def bars_from(params: list) -> dict:
    """{time: (open, high, low, close, volume)} from the series payloads of a timescale_update / du message."""
    out = {}
    for block in params[1:]:
        if not isinstance(block, dict):
            continue
        for series in block.values():
            if isinstance(series, dict) and isinstance(series.get("s"), list):
                for point in series["s"]:
                    v = point.get("v") or []
                    if len(v) >= 5:
                        out[int(v[0])] = tuple(v[1:6]) if len(v) >= 6 else tuple(v[1:5]) + (None,)
    return out


def check_error(message: dict):
    if message["m"] in ("symbol_error", "series_error", "critical_error", "protocol_error"):
        detail = " ".join(str(x) for x in message["p"][1:]) or message["m"]
        raise RuntimeError(f"TradingView answered {message['m']}: {detail}  (is the symbol right? it must look like EXCHANGE:SYMBOL, e.g. CME_MINI:NQ1!)")


# ------------------------------------------------------------------------------------------------------------------------------------ history
def describe(info: dict) -> str:
    name = info.get("full_name") or info.get("name") or "?"
    text = f"Symbol: {name} - {info.get('description', '')} ({info.get('exchange', '')}, {info.get('type', '')})"
    delay = info.get("delay_seconds")
    return text + (f"; delayed {int(delay) // 60} min" if delay else "")


def fetch_history(symbol: str, interval: str, bars: int, session: str = "regular", timeout: float = 30.0, log=print):
    """Returns ({time: (o, h, l, c, v)}, info about the symbol, the open Feed). The caller closes the feed. Asks for the newest `bars` bars, then for older ones until `bars` are in or the server has no more."""
    feed = Feed(timeout)
    got, info = {}, {}
    feed.open_series(symbol, interval, min(max(int(bars), 1), 5000), session)
    rounds, size_before_round = 0, 0
    deadline = time.time() + max(60.0, 2 * timeout)
    while time.time() < deadline:
        try:
            events = list(feed.events())
        except socket.timeout:
            break
        for message in events:
            check_error(message)
            name, params = message["m"], message.get("p") or []
            if name == "symbol_resolved" and len(params) >= 3 and isinstance(params[2], dict):
                info = params[2]
                log(describe(info))
            elif name in ("timescale_update", "du"):
                got.update(bars_from(params))
            elif name == "series_completed":
                if len(got) >= bars or len(got) <= size_before_round or rounds >= 40:
                    return got, info, feed                    # enough bars, or the last request brought nothing new
                size_before_round = len(got)
                feed.send("request_more_data", [feed.session, "sds_1", min(5000, bars - len(got))])
                rounds += 1
                deadline = time.time() + max(30.0, timeout)
    return got, info, feed



EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def iso(t: int) -> str:
    return (EPOCH + dt.timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%SZ")        # not fromtimestamp(): on Windows it fails for dates before 1970


def write_csv(path: str, bars: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "unix", "open", "high", "low", "close", "volume"])
        for t in sorted(bars):
            o, h, l, c, v = bars[t]
            w.writerow([iso(t), t, o, h, l, c, "" if v is None else v])


def follow_live(feed: Feed, path: str, seconds: float, log=print):
    """Prints and appends every bar update that arrives (the last bar changes with each tick, a new bar appears when one closes) until Ctrl+C or `seconds`."""
    new_file = not os.path.exists(path)
    end = time.time() + seconds if seconds else None
    log(f"Following live bars into {path} (Ctrl+C to stop)...")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["received_at", "bar_time", "open", "high", "low", "close", "volume"])
        feed.ws.sock.settimeout(60.0)
        while end is None or time.time() < end:
            try:
                events = list(feed.events())
            except socket.timeout:
                continue
            for message in events:
                check_error(message)
                if message["m"] in ("timescale_update", "du"):
                    for t, (o, h, l, c, v) in sorted(bars_from(message.get("p") or []).items()):
                        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        w.writerow([now, iso(t), o, h, l, c, "" if v is None else v])
                        f.flush()
                        log(f"{now}  bar {iso(t)}  O {o}  H {h}  L {l}  C {c}  V {v}")


def main():
    p = argparse.ArgumentParser(description="Download TradingView price history to CSV (standard library only).")
    p.add_argument("symbol", help="EXCHANGE:SYMBOL as shown on the chart, e.g. CME_MINI:NQ1!  BITSTAMP:BTCUSD  NASDAQ:AAPL")
    p.add_argument("--interval", default="1D", help=f"one of {' '.join(INTERVALS)} (default 1D)")
    p.add_argument("--bars", type=int, default=5000, help="how many bars to ask for, newest first (default 5000; the server may give fewer)")
    p.add_argument("--all", action="store_true", help="everything the feed has (as far back as it goes), instead of --bars")
    p.add_argument("--session", choices=["regular", "extended"], default="regular", help="regular or extended trading hours (default regular)")
    p.add_argument("--out", default="tv_data", help="folder for the CSV (default ./tv_data)")
    p.add_argument("--live", action="store_true", help="after the history, keep the socket open and append every bar update to a second CSV until Ctrl+C")
    p.add_argument("--seconds", type=float, default=0, help="with --live: stop after this many seconds (default: until Ctrl+C)")
    a = p.parse_args()
    if a.all:
        a.bars = 1_000_000
    interval = {"D": "1D", "W": "1W", "M": "1M"}.get(a.interval.upper(), a.interval.upper())      # the chart's own names for day / week / month work too
    if interval not in INTERVALS:
        sys.exit(f"Unknown interval '{a.interval}'. Use one of: {' '.join(INTERVALS)}")
    os.makedirs(a.out, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", a.symbol)
    print(f"Connecting to {HOST} ...")
    try:
        bars, info, feed = fetch_history(a.symbol, interval, a.bars, a.session)
    except (OSError, ConnectionError, RuntimeError) as e:
        sys.exit(f"Failed: {e}")
    if not bars:
        feed.close()
        sys.exit("No bars came back (wrong symbol, or the feed refused the request).")
    path = os.path.join(a.out, f"{safe}_{interval}.csv")
    write_csv(path, bars)
    first, last = min(bars), max(bars)
    print(f"{len(bars)} bars, {iso(first)} to {iso(last)}  ->  {path}")
    try:
        if a.live:
            follow_live(feed, os.path.join(a.out, f"{safe}_{interval}_live.csv"), a.seconds)
    except KeyboardInterrupt:
        print("Stopped.")
    except (OSError, ConnectionError, RuntimeError) as e:
        print(f"Live feed ended: {e}")
    finally:
        feed.close()


if __name__ == "__main__":
    main()
