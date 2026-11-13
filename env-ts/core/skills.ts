// Match rl/obs.py SKILL_ORDER. Slots 18/19 are unused native stats in this
// content revision; their historical contract labels do not imply playable skills.
export const SKILL_NAMES = [
    'Attack', 'Defence', 'Strength', 'Hitpoints', 'Ranged', 'Prayer', 'Magic',
    'Cooking', 'Woodcutting', 'Fletching', 'Fishing', 'Firemaking', 'Crafting',
    'Smithing', 'Mining', 'Herblore', 'Agility', 'Thieving', 'Slayer', 'Farming',
    'Runecrafting'
] as const;

export function sdkSkillName(name: string): string {
    // Normalize only the three native SDK aliases. Never invent absent rows or XP.
    if (name === 'Stat18') return SKILL_NAMES[18];
    if (name === 'Stat19') return SKILL_NAMES[19];
    if (name === 'Runecraft') return SKILL_NAMES[20];
    return name;
}
