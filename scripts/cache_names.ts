// Every loc, npc and obj in the game cache with a name: its examine text and options.
// Name vectors are computed from this list. The output is
// Jagex-derived data: write it under logs/, never into git.
//
//   bun scripts/cache_names.ts logs/names/cache-names.json
import fs from 'fs';
import path from 'path';

import LocType from '../vendor/rs-sdk/server/engine/src/cache/config/LocType.js';
import NpcType from '../vendor/rs-sdk/server/engine/src/cache/config/NpcType.js';
import ObjType from '../vendor/rs-sdk/server/engine/src/cache/config/ObjType.js';

const out = process.argv[2];
if (!out) throw new Error('usage: bun scripts/cache_names.ts OUTPUT.json');
const pack = path.join(import.meta.dir, '../vendor/rs-sdk/server/engine/data/pack');
LocType.load(pack);
NpcType.load(pack);
ObjType.load(pack);

const rows: { kind: string; id: number; name: string; desc: string | null; options: string[] }[] = [];
const add = (kind: string, configs: { id: number; name: string | null; desc: string | null }[],
             options: (c: any) => (string | null)[]) => {
    for (const c of configs) {
        if (!c?.name || c.name === 'null') continue;
        rows.push({ kind, id: c.id, name: c.name, desc: c.desc, options: options(c).filter((o): o is string => !!o) });
    }
};
add('loc', LocType.configs, c => c.op ?? []);
add('npc', NpcType.configs, c => c.op ?? []);
add('obj', ObjType.configs, c => [...(c.op ?? []), ...(c.iop ?? [])]);
if (!rows.length) throw new Error(`no cache configs under ${pack}`);
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(rows));
const names = new Set(rows.map(r => `${r.kind}:${r.name.toLowerCase()}`));
console.log(`${rows.length} named configs, ${names.size} distinct kind:name pairs -> ${out}`);
