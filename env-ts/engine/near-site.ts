// env-ts/engine/near-site.ts: RL_NPC_SITES="x,z;x,z" with RL_NPC_KEEP_TILES=R (a speed prototype, never a training
// world): at boot the world removes every npc farther than R tiles (the larger of |dx| and |dz|, on any level) from
// every listed site, the way a script-spawned npc despawns (World.removeNpc with a DESPAWN life cycle: it leaves its
// zone, its collision and the npc list, and never respawns). The per-tick passes over the npc list (npc info, cleanup,
// hunt, the dormant turns of vendor patch 0009) then cover only the npcs near the sites.
//
// What it changes for an agent: every npc more than R tiles from its site is gone for good. The dormant-npc gate wakes
// every npc of a 64-tile chunk within RL_NPC_RADIUS (64) tiles of a player, so an agent that comes within 128 tiles of
// that edge can meet a world emptier than the game's (an npc that would have walked or hunted into its view).
const DESPAWN = 2; // EntityLifeCycle.DESPAWN (a const enum of the vendor tree)

export function parseSites(text: string): Array<[number, number]> {
    const sites = text.split(';').filter(Boolean).map(pair => pair.split(',').map(v => parseInt(v, 10)));
    if (!sites.length || sites.some(s => s.length !== 2 || !s.every(Number.isSafeInteger)))
        throw new Error(`RL_NPC_SITES holds x,z pairs separated by ';', not ${JSON.stringify(text)}`);
    return sites as Array<[number, number]>;
}

/** The npcs farther than `radius` tiles from every site, removed; returns the counts kept and removed. */
export function keepNpcsNear(world: any, sites: Array<[number, number]>, radius: number): { kept: number; removed: number } {
    if (!Number.isSafeInteger(radius) || radius < 0) throw new Error('RL_NPC_KEEP_TILES is a whole number of tiles');
    const far = [...world.npcs].filter((npc: any) =>
        sites.every(([x, z]) => Math.max(Math.abs(npc.x - x), Math.abs(npc.z - z)) > radius));
    for (const npc of far) {
        npc.lifecycle = DESPAWN;
        world.removeNpc(npc, -1);
    }
    return { kept: world.getTotalNpcs(), removed: far.length };
}

/** The prototype in `world` (the engine's World) when RL_NPC_SITES is set; null otherwise. */
export function installNearSite(world: any, env: Record<string, string | undefined> = process.env) {
    if (!env.RL_NPC_SITES) return null;
    const sites = parseSites(env.RL_NPC_SITES);
    const counts = keepNpcsNear(world, sites, parseInt(env.RL_NPC_KEEP_TILES ?? '', 10));
    return { sites, ...counts };
}
