// env-ts/core/socket-out.ts: the bytes of a reply that a socket has not taken yet.
//
// A Bun socket writes only what the kernel takes and drops the rest unless it is written again on `drain`. A reply of
// one agent (about 10 KB) always fit; a batch reply of a hundred agents (about 1 MB) did not, and its reader waited
// for the end of a line that never came (18 November 2026, engine/batch.ts).

export interface OutSocket {
    write(data: Uint8Array): number;
    data: { out?: Uint8Array | null };
}

/** Write what the socket takes now and keep the rest for the next drain. */
export function flush(socket: OutSocket): void {
    const out = socket.data.out;
    if (!out) return;
    const n = socket.write(out);
    socket.data.out = n < out.length ? out.subarray(Math.max(n, 0)) : null;
}

/** Queue `bytes` behind anything still waiting, then write what the socket takes. */
export function queue(socket: OutSocket, bytes: Uint8Array): void {
    const waiting = socket.data.out;
    if (!waiting) {
        socket.data.out = bytes;
        flush(socket);
        return;
    }
    const both = new Uint8Array(waiting.length + bytes.length);
    both.set(waiting);
    both.set(bytes, waiting.length);
    socket.data.out = both;
}
