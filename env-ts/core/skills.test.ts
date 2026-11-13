import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { SKILL_NAMES, sdkSkillName } from './skills';

test('native SDK aliases produce the same 21 named slots as engine and Python', () => {
    const native = [...SKILL_NAMES.slice(0, 18), 'Stat18', 'Stat19', 'Runecraft'];
    expect(native.map(sdkSkillName)).toEqual([...SKILL_NAMES]);
    const python = readFileSync(new URL('../../rl/obs.py', import.meta.url), 'utf8');
    const declaration = python.match(/SKILL_ORDER\s*=\s*\(([\s\S]*?)\)/)![1];
    expect([...declaration.matchAll(/"([^"]+)"/g)].map(m => m[1])).toEqual([...SKILL_NAMES]);
});

test('canonical names and unexpected names remain unchanged; missing rows stay missing', () => {
    expect(SKILL_NAMES.map(sdkSkillName)).toEqual([...SKILL_NAMES]);
    expect(sdkSkillName('Unknown')).toBe('Unknown');
    expect(sdkSkillName('Stat20')).toBe('Stat20');
    expect([].map(sdkSkillName)).toEqual([]);
});
