#!/usr/bin/env python3
"""Check that a ggml-rpc-server is actually SERVING, not just listening.

ggml-rpc-server handles one client at a time and never times out a read. If a
client's host dies without closing its socket (VM reboot, OOM kill), the server
blocks forever on that dead connection -- but the kernel keeps accepting new
connections into the listen backlog, so a plain TCP connect still succeeds.
A new llama-server pointed at it then hangs in the HELLO handshake and never
opens its HTTP port.

This sends the real HELLO and requires a reply within the timeout.

    rpc-hello.py HOST PORT [TIMEOUT_S]     exit 0 = serving, 1 = no reply

Wire format from src/llama.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp:
    request : cmd (1 byte) | size (u64 LE) | rpc_msg_hello_req (caps)
    response: size (u64 LE) | rpc_msg_hello_rsp (major, minor, patch, pad, caps)
The version is printed but not enforced: any answer proves the server is
serving, and llama-server does its own version check.
"""
import socket
import struct
import sys

RPC_CMD_HELLO = 14          # static_assert(RPC_CMD_HELLO == 14)
RPC_CONN_CAPS_SIZE = 24     # ggml-rpc/transport.h

host, port = sys.argv[1], int(sys.argv[2])
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed by server")
        buf += chunk
    return buf


try:
    with socket.create_connection((host, port), timeout=timeout) as s:
        # All-zero caps = no RDMA, so the server stays on plain TCP.
        s.sendall(struct.pack("<BQ", RPC_CMD_HELLO, RPC_CONN_CAPS_SIZE)
                  + bytes(RPC_CONN_CAPS_SIZE))
        (size,) = struct.unpack("<Q", recv_exact(s, 8))
        if size != 4 + RPC_CONN_CAPS_SIZE:
            raise ConnectionError("unexpected HELLO reply size %d" % size)
        major, minor, patch = recv_exact(s, size)[:3]
except OSError as e:
    print("%s:%d no HELLO reply (%s)" % (host, port, e or type(e).__name__))
    sys.exit(1)

print("%s:%d serving, rpc protocol %d.%d.%d" % (host, port, major, minor, patch))
