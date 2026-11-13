import { expect, test } from 'bun:test';
import { A_ACT0, A_CLICK_HIGH, A_CLICK_LOW } from './actions';
import { CLICK_OFFSETS } from './clicks';
import { TextCapture, describeAction, feedback, type OptionEnd, type Snapshot } from './feedback';

const state = (tick: number, extra: Partial<Snapshot> = {}): Snapshot => ({ tick, x: 10, z: 10, level: 0, hp: 10, xp: 0,
    items: new Map([[1351, 1]]), bankOpen: false, dialogOpen: false, ...extra });

test('dialog text set before the dialog opens is read through the open interface', () => {
    // ~mesbox sets its line, then opens the chat interface; component 1001 belongs to interface 1000.
    const text = new TextCapture();
    text.onText(1001, 'You need a net to catch these fish.', 5);
    text.onText(2001, 'Bank of RuneScape', 5); // another interface's text
    text.onChat('You swing your axe at the tree.', 3);
    const inside = (com: number) => Math.floor(com / 1000) === 1;
    expect(text.interfaceText(inside)).toBe('You need a net to catch these fish.');
    expect(text.publish(6)).toEqual([{ text: 'You swing your axe at the tree.', age: 3 }]);
    expect(text.publish(7)).toEqual([]);  // published once
    text.clear(); // a reset: nothing carries into the next episode
    expect(text.interfaceText(inside)).toBe('');
});

test('each chat line is published by exactly one observation, whatever tick it carries', () => {
    const text = new TextCapture();
    text.onChat('Before the reset.', 9);
    text.clear();                                   // the reset observation at tick 10 covers everything so far
    text.onChat('You need a Woodcutting level of 15 to chop down this tree.', 10); // the first cycle after it
    expect(text.publish(12)).toEqual([{ text: 'You need a Woodcutting level of 15 to chop down this tree.', age: 2 }]);
    text.onChat('You eat the bread.', 12);          // same tick as the observation just taken
    text.onChat('It heals some health.', 12);
    expect(text.publish(14)).toEqual([{ text: 'You eat the bread.', age: 2 }, { text: 'It heals some health.', age: 2 }]);
    expect(text.publish(16)).toEqual([]);
});

test('an action keeps the meaning it had when chosen, and an option end is reported once', () => {
    const chosen = { actSel: [{ kind: 'npc', name: 'Fishing spot', option: 'Net' }, { kind: 'loc', name: 'Tree', option: 'Chop down' }] };
    const meaning = describeAction(chosen, A_ACT0);
    expect(meaning).toMatchObject({ family: 'act', kind: 'npc', name: 'Fishing spot', option: 'Net' });
    // a click means its offset (feedback-v2); one chosen from no observation keeps only its family
    const clicks = { clicks: CLICK_OFFSETS.map(([dx, dz]) => ({ dx, dz, inside: true, open: true })) };
    expect(describeAction(clicks, A_CLICK_LOW + 2)).toMatchObject({ family: 'click', option: '4,0' });
    expect(describeAction(clicks, A_CLICK_HIGH + 4)).toMatchObject({ family: 'click', option: '0,16' });
    expect(describeAction(null, A_CLICK_LOW)).toMatchObject({ family: 'click', option: '' });
    const end: OptionEnd = { action: A_ACT0, reason: 'done', refused: true, reported: false };
    const dialog = { text: 'You need a net to catch these fish.', choices: [] };
    const first = feedback(meaning, state(10), state(16, { dialogOpen: true }), false, end, [], dialog);
    expect(first).toMatchObject({ action: meaning, ended: { reason: 'done', refused: true }, ticks: 6, gained: 0,
                                  lost: 0, xp: 0, dialogOpened: true });
    expect(feedback(meaning, state(16), state(18), false, end, [], dialog).ended).toBeNull();
    const caught = feedback(meaning, state(20), state(30, { xp: 10, items: new Map([[1351, 1], [317, 1]]) }), true, null, [],
                            { text: '', choices: [] });
    expect(caught).toMatchObject({ gained: 1, xp: 10, running: true, ended: null });
});
