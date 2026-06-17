"""P2P Specialist Swarm — discovery and peer-to-peer specialist exchange.

Lightweight P2P layer on top of the existing specialist_network.py:
- UDP broadcast for peer discovery on LAN
- Peer registry with health monitoring
- Specialist checkpoint exchange via direct TCP transfer
- REST API integration with the dashboard

Architecture:
  Each Tabula Rasa instance runs a P2P daemon that:
  1. Broadcasts its presence on the LAN every 30s
  2. Listens for peer broadcasts
  3. Maintains a peer registry (hostname, IP, port, specialists available)
  4. Serves specialist checkpoints via HTTP for peer downloads
  5. Exposes a REST endpoint for the dashboard

Usage:
    python3 egefalos/p2p_daemon.py              # Start P2P daemon (port 9000)
    python3 egefalos/p2p_daemon.py --port 9001   # Custom port
    python3 egefalos/p2p_daemon.py --list        # List discovered peers
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
import sys
from pathlib import Path
from typing import Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import argparse


# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

MULTICAST_GROUP = '239.255.43.21'   # Randomly chosen multicast group
MULTICAST_PORT = 43210
BROADCAST_INTERVAL = 30  # seconds
PEER_TIMEOUT = 120        # seconds without hearing from peer = dead
HTTP_PORT = 9000          # Port for HTTP peer API and checkpoint serving

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECIALISTS_DIR = PROJECT_ROOT / 'specialists' / 'math'


# ══════════════════════════════════════════════════════════════
# Peer Registry
# ══════════════════════════════════════════════════════════════

class PeerRegistry:
    """Thread-safe registry of known P2P peers."""

    def __init__(self):
        self._peers: dict[str, dict] = {}  # hostname -> peer info
        self._lock = threading.Lock()

    def update(self, hostname: str, ip: str, port: int, specialists: list[str]):
        """Update or add a peer."""
        hostname = hostname.lower()
        with self._lock:
            self._peers[hostname] = {
                'hostname': hostname,
                'ip': ip,
                'port': port,
                'specialists': specialists or [],
                'last_seen': time.time(),
                'alive': True,
            }

    def get_peers(self) -> list[dict]:
        """Get list of alive peers (not timed out)."""
        now = time.time()
        with self._lock:
            alive = []
            for hostname, info in self._peers.items():
                if now - info['last_seen'] < PEER_TIMEOUT:
                    info['alive'] = True
                    alive.append(info)
                else:
                    info['alive'] = False
            return alive

    def remove_stale(self) -> None:
        """Remove peers that have timed out."""
        now = time.time()
        with self._lock:
            stale = [h for h, i in self._peers.items()
                     if now - i['last_seen'] >= PEER_TIMEOUT]
            for h in stale:
                del self._peers[h]

    def get_summary(self) -> dict:
        """Get diagnostic summary."""
        alive = self.get_peers()
        return {
            'total_peers': len(self._peers),
            'alive_peers': len(alive),
            'peers': alive,
        }


# ══════════════════════════════════════════════════════════════
# UDP Discovery Protocol
# ══════════════════════════════════════════════════════════════

def _get_specialist_list() -> list[str]:
    """List locally available specialist operations."""
    if not SPECIALISTS_DIR.exists():
        return []
    ops = []
    for d in sorted(SPECIALISTS_DIR.iterdir()):
        if d.is_dir() and (d / 'best.pt').exists():
            ops.append(d.name)
    return ops


class DiscoveryBroadcaster(threading.Thread):
    """Periodically broadcasts this node's presence via UDP multicast."""

    def __init__(self, registry: PeerRegistry, listen_port: int, daemon: bool = True):
        super().__init__(daemon=daemon)
        self.registry = registry
        self.listen_port = listen_port
        self._stop = threading.Event()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(5)

        while not self._stop.is_set():
            try:
                hostname = socket.gethostname()
                specialists = _get_specialist_list()
                payload = json.dumps({
                    'type': 'tabula_rasa_peer',
                    'hostname': hostname,
                    'port': self.listen_port,
                    'specialists': specialists,
                    'version': 1,
                }).encode('utf-8')

                sock.sendto(payload, (MULTICAST_GROUP, MULTICAST_PORT))
                print(f'  [P2P] Broadcast presence: {hostname} ({len(specialists)} specialists)')
            except Exception as e:
                print(f'  [P2P] Broadcast error: {e}')

            self._stop.wait(BROADCAST_INTERVAL)

        sock.close()

    def stop(self):
        self._stop.set()


class DiscoveryListener(threading.Thread):
    """Listens for UDP multicast peer announcements."""

    def __init__(self, registry: PeerRegistry, daemon: bool = True):
        super().__init__(daemon=daemon)
        self.registry = registry
        self._stop = threading.Event()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', MULTICAST_PORT))

        mreq = struct.pack('4sl', socket.inet_aton(MULTICAST_GROUP), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(2)

        print(f'  [P2P] Listening on multicast {MULTICAST_GROUP}:{MULTICAST_PORT}')

        while not self._stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
                msg = json.loads(data.decode('utf-8'))
                if msg.get('type') == 'tabula_rasa_peer':
                    self.registry.update(
                        hostname=msg['hostname'],
                        ip=addr[0],
                        port=msg.get('port', HTTP_PORT),
                        specialists=msg.get('specialists', []),
                    )
            except socket.timeout:
                continue
            except Exception as e:
                sys.stderr.write(f'  [P2P] Listener error: {e}\n')

        sock.close()

    def stop(self):
        self._stop.set()


# ══════════════════════════════════════════════════════════════
# HTTP Peer API — serve specialist checkpoints to peers
# ══════════════════════════════════════════════════════════════

class PeerHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for peer-to-peer specialist exchange.

    Endpoints:
      GET /health          — Peer status + specialist list
      GET /peers           — Known peers from registry
      GET /checkpoint/<op> — Download a specialist's best.pt
    """

    # Shared via class variable (set before server starts)
    registry: PeerRegistry = None  # type: ignore

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/health':
            self._json_response({
                'status': 'ok',
                'hostname': socket.gethostname(),
                'specialists': _get_specialist_list(),
                'peers': self.registry.get_summary(),
            })
        elif path == '/peers':
            self._json_response(self.registry.get_summary())
        elif path.startswith('/checkpoint/'):
            op = path.split('/checkpoint/')[1]
            ckpt_path = SPECIALISTS_DIR / op / 'best.pt'
            if ckpt_path.exists():
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('X-Operation', op)
                self.send_header('Content-Length', str(ckpt_path.stat().st_size))
                self.end_headers()
                with open(ckpt_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self._json_response({'error': f'No checkpoint for {op}'}, 404)
        else:
            self._json_response({'error': 'Not found'}, 404)

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Suppress HTTP log spam


# ══════════════════════════════════════════════════════════════
# P2P Daemon
# ══════════════════════════════════════════════════════════════

class P2PDaemon:
    """Full P2P service: discovery + HTTP peer API.

    Runs as a background thread alongside the main API server.
    Can also run standalone for testing.
    """

    def __init__(self, http_port: int = HTTP_PORT):
        self.http_port = http_port
        self.registry = PeerRegistry()
        self.broadcaster: Optional[DiscoveryBroadcaster] = None
        self.listener: Optional[DiscoveryListener] = None
        self.http_server: Optional[HTTPServer] = None
        self._threads: list[threading.Thread] = []

    def start(self):
        """Start all P2P services."""
        print(f'\n{"="*50}')
        print(f'  P2P Specialist Swarm Daemon')
        print(f'  HTTP API:  http://localhost:{self.http_port}')
        print(f'  Multicast: {MULTICAST_GROUP}:{MULTICAST_PORT}')
        print(f'  Interval:  {BROADCAST_INTERVAL}s broadcast / {PEER_TIMEOUT}s timeout')
        print(f'{"="*50}')

        # Start discovery listener
        self.listener = DiscoveryListener(self.registry)
        self.listener.start()
        self._threads.append(self.listener)

        # Start discovery broadcaster
        self.broadcaster = DiscoveryBroadcaster(self.registry, self.http_port)
        self.broadcaster.start()
        self._threads.append(self.broadcaster)

        # Start HTTP API
        PeerHTTPHandler.registry = self.registry
        self.http_server = HTTPServer(('0.0.0.0', self.http_port), PeerHTTPHandler)
        server_thread = threading.Thread(target=self.http_server.serve_forever,
                                         daemon=True)
        server_thread.start()
        self._threads.append(server_thread)

        print(f'  [P2P] Daemon running on port {self.http_port}')
        print(f'  [P2P] Endpoints:')
        print(f'    GET /health    — Peer status + specialists')
        print(f'    GET /peers     — Known peers')
        print(f'    GET /checkpoint/<op> — Download specialist')
        return self

    def stop(self):
        """Stop all P2P services."""
        if self.broadcaster:
            self.broadcaster.stop()
        if self.listener:
            self.listener.stop()
        if self.http_server:
            self.http_server.shutdown()

    def list_peers(self) -> list[dict]:
        """Return discovered peers."""
        return self.registry.get_peers()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='P2P Specialist Swarm Daemon')
    parser.add_argument('--port', type=int, default=HTTP_PORT,
                        help=f'HTTP API port (default: {HTTP_PORT})')
    parser.add_argument('--list', action='store_true',
                        help='List currently discovered peers and exit')
    parser.add_argument('--oneshot', action='store_true',
                        help='Listen for broadcasts briefly, list peers, then exit')
    args = parser.parse_args()

    daemon = P2PDaemon(http_port=args.port)

    if args.list:
        # Just start listener briefly, collect peers, show them
        reg = PeerRegistry()
        listener = DiscoveryListener(reg)
        listener.start()
        print('  [P2P] Listening for peers (5s)...')
        time.sleep(5)
        listener.stop()
        peers = reg.get_peers()
        if peers:
            print(f'\n  Discovered {len(peers)} peer(s):')
            for p in peers:
                print(f'    {p["hostname"]} ({p["ip"]}:{p["port"]}) — '
                      f'{len(p["specialists"])} specialists')
        else:
            print('  No peers discovered.')
        return

    if args.oneshot:
        daemon.start()
        print('  [P2P] Running. Press Ctrl+C to stop.')
        try:
            while True:
                time.sleep(10)
                peers = daemon.list_peers()
                if peers:
                    print(f'  [P2P] Peers: {", ".join(p["hostname"] for p in peers)}')
        except KeyboardInterrupt:
            print('\n  [P2P] Stopping...')
        daemon.stop()
        return

    # Full daemon mode
    daemon.start()
    print('\n  [P2P] Running. Press Ctrl+C to stop.\n')
    try:
        while True:
            time.sleep(10)
            peers = daemon.list_peers()
            if peers:
                print(f'  [P2P] Peers: {", ".join(p["hostname"] for p in peers)}')
    except KeyboardInterrupt:
        print('\n  [P2P] Shutting down...')
    daemon.stop()


if __name__ == '__main__':
    main()
