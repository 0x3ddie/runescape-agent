import { test, expect } from 'bun:test';
import { flush, queue } from './socket-out';

test('a reply larger than the socket takes at once arrives whole and in order over its drains', () => {
    const received: number[] = [];
    const socket = { data: {} as { out?: Uint8Array | null },
        write(data: Uint8Array) { const n = Math.min(4096, data.length); received.push(...data.subarray(0, n)); return n; } };
    const big = new Uint8Array(1_000_000).map((_, i) => i % 251), small = new Uint8Array([7, 8, 9]);
    queue(socket, big);
    queue(socket, small);                       // behind the rest of the big reply, not mixed into it
    expect(received.length).toBe(4096);
    while (socket.data.out) flush(socket);      // the drains
    expect(received.length).toBe(big.length + small.length);
    expect(received.slice(0, big.length)).toEqual([...big]);
    expect(received.slice(big.length)).toEqual([7, 8, 9]);
});
