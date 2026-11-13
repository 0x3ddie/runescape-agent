/** Explicit episode reset retires the old object before admitting its successor.
 * Natural death/respawn does not call this helper.
 */
export function replacePlayerForReset<T>(previous: T, pending: Set<T>,
    unregister: (player: T) => void, admit: () => T): T {
    pending.delete(previous);
    unregister(previous);
    const next = admit();
    if (next === previous) throw new Error('reset: admission reused the previous Player');
    return next;
}
