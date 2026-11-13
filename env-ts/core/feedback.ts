// env-ts/core/feedback.ts: feedback-v2: what the last decision asked for and what followed.
//
// PURE. Until October 2026 the policy saw the option still running and nothing else: no
// game text, no note of a finished or refused option, so a failed "Net" and a successful
// one looked alike. This publishes evidence, never a verdict:
// the action's meaning taken when it was dispatched (a slot may hold another thing next
// time), the changes since that decision, how a running option ended, and the text the
// game showed. Nothing here concludes "missing tool"; the learner has to. feedback-v2 (native-click, act-v3)
// has one click family, whose meaning is the click's offset ("dx,dz"), in place of place, retreat and move.
import {
    A_ACT0, A_CLOSE_IF, A_DEPOSIT0, A_DIALOG_CHOICE0, A_DIALOG_CONTINUE, A_DROP0, A_HELD0, A_INV_PREV,
    A_STYLE0, A_USE0, A_WITHDRAW0, N_ACT, N_HELD, N_INV_SEL
} from './actions';
import { clickSlot } from './clicks';

export const FEEDBACK_CONTRACT = 'feedback-v2';
/** Row families of the action table, in rl/feedback_contract.py's order. */
export const FAMILIES = ['wait', 'act', 'click', 'held', 'drop', 'deposit', 'withdraw', 'close', 'dialog', 'page',
    'style', 'use'] as const;
export const END_REASONS = ['done', 'stalled', 'expired', 'failed', 'superseded'] as const;
export type EndReason = typeof END_REASONS[number];
/** Chat lines kept between decisions; a decision publishes only those that arrived since the last. */
export const CHAT_KEEP = 16;

export function familyOf(action: number): typeof FAMILIES[number] {
    const within = (start: number, n: number) => action >= start && action < start + n;
    if (action === 0) return 'wait';
    if (within(A_ACT0, N_ACT)) return 'act';
    if (clickSlot(action) !== null) return 'click';
    if (within(A_HELD0, N_HELD)) return 'held';
    if (within(A_DROP0, N_INV_SEL)) return 'drop';
    if (within(A_DEPOSIT0, N_INV_SEL)) return 'deposit';
    if (within(A_WITHDRAW0, N_INV_SEL)) return 'withdraw';
    if (action === A_CLOSE_IF) return 'close';
    if (action === A_DIALOG_CONTINUE || within(A_DIALOG_CHOICE0, 4)) return 'dialog';
    if (within(A_INV_PREV, 4)) return 'page';
    if (within(A_STYLE0, 4)) return 'style';
    if (within(A_USE0, N_INV_SEL)) return 'use';
    throw new Error(`no family for action ${action}`);
}

export type ActionMeaning = { action: number; family: string; kind: string; name: string; option: string; item: string };

/** What `action` meant in the observation the policy chose it from. */
export function describeAction(obs: Record<string, any> | null, action: number): ActionMeaning {
    const family = familyOf(action);
    const meaning: ActionMeaning = { action, family, kind: '', name: '', option: '', item: obs?.using?.name ?? '' };
    const slot = (list: any[] | undefined, start: number) => list?.[action - start];
    if (family === 'act') {
        const a = slot(obs?.actSel, A_ACT0);
        if (a) Object.assign(meaning, { kind: a.kind, name: a.name, option: a.option });
    } else if (family === 'click') {
        const c = obs?.clicks?.[clickSlot(action)!];
        if (c) meaning.option = `${c.dx},${c.dz}`;
    } else if (family === 'held') {
        const h = slot(obs?.heldSel, A_HELD0);
        if (h) Object.assign(meaning, { kind: 'obj', name: h.name, option: obs?.using ? 'Use' : h.option });
    } else if (family === 'drop' || family === 'deposit' || family === 'use') {
        const i = slot(obs?.invSel, family === 'drop' ? A_DROP0 : family === 'deposit' ? A_DEPOSIT0 : A_USE0);
        if (i && i.slot >= 0) Object.assign(meaning, { kind: 'obj', name: i.name,
            option: family === 'drop' ? 'Drop' : family === 'deposit' ? 'Deposit' : 'Use', item: '' });
    } else if (family === 'withdraw') {
        const i = slot(obs?.bankSel, A_WITHDRAW0);
        if (i && i.slot >= 0) Object.assign(meaning, { kind: 'obj', name: i.name, option: 'Withdraw' });
    }
    return meaning;
}

/** What the agent can see of itself at a decision. */
export type Snapshot = { tick: number; x: number; z: number; level: number; hp: number; xp: number;
    items: Map<number, number>; bankOpen: boolean; dialogOpen: boolean };

export function changes(before: Snapshot, after: Snapshot) {
    let gained = 0, lost = 0;
    for (const id of new Set([...before.items.keys(), ...after.items.keys()])) {
        const delta = (after.items.get(id) ?? 0) - (before.items.get(id) ?? 0);
        if (delta > 0) gained += delta; else lost -= delta;
    }
    return { ticks: after.tick - before.tick, dx: after.x - before.x, dz: after.z - before.z,
             levelChanged: after.level !== before.level, xp: after.xp - before.xp, hp: after.hp - before.hp,
             gained, lost, bankOpened: after.bankOpen && !before.bankOpen, dialogOpened: after.dialogOpen && !before.dialogOpen };
}

/**
 * Game text as the player would read it. Chat arrives as lines; dialog text is set on
 * interface components, often BEFORE the dialog opens (~mesbox sets its lines, then opens),
 * so the latest text of every component is kept and read through the interface that is
 * open now. A reset makes a new Player; clear() runs with it.
 */
export class TextCapture {
    private texts = new Map<number, { text: string; tick: number }>();
    private chat: { text: string; tick: number; sequence: number }[] = [];
    // Each chat line gets the next sequence number on arrival; an observation publishes exactly the
    // lines above the last sequence an earlier one covered. Ticks cannot decide it: a line sent in the
    // first cycle after an observation carries that observation's tick.
    private sequence = 0;
    private covered = 0;

    /** A reset: no earlier text belongs to the new episode, and its first observation covers it all. */
    clear(): void { this.texts.clear(); this.chat = []; this.covered = this.sequence; }

    onChat(text: string, tick: number): void {
        this.chat.push({ text, tick, sequence: ++this.sequence });
        if (this.chat.length > CHAT_KEEP) this.chat.splice(0, this.chat.length - CHAT_KEEP);
    }

    onText(component: number, text: string, tick: number): void { this.texts.set(component, { text, tick }); }

    /** The lines no earlier observation published, with their age at `now`; they are covered from now on. */
    publish(now: number): { text: string; age: number }[] {
        const lines = this.chat.filter(c => c.sequence > this.covered).map(c => ({ text: c.text, age: now - c.tick }));
        this.covered = this.sequence;
        return lines;
    }

    /** Non-empty text of the components `inside` accepts, in component order, joined. */
    interfaceText(inside: (component: number) => boolean): string {
        return [...this.texts.entries()].filter(([com, t]) => t.text.trim() && inside(com)).sort(([a], [b]) => a - b)
            .map(([, t]) => t.text.trim()).join(' ');
    }

    textOf(component: number): string { return this.texts.get(component)?.text.trim() ?? ''; }
}

/** How an option ended; `reported` once an observation has carried it, so it is reported once. */
export type OptionEnd = { action: number; reason: EndReason; refused: boolean; reported: boolean };

/** The feedback field of an observation. `action` and `before` are null at an episode's first observation. */
export function feedback(action: ActionMeaning | null, before: Snapshot | null, after: Snapshot, running: boolean,
                         end: OptionEnd | null, chat: { text: string; age: number }[],
                         dialog: { text: string; choices: string[] }) {
    const none = { ticks: 0, dx: 0, dz: 0, levelChanged: false, xp: 0, hp: 0, gained: 0, lost: 0, bankOpened: false,
                   dialogOpened: false };
    let ended = null;
    if (end && !end.reported) {
        ended = { action: end.action, reason: end.reason, refused: end.refused };
        end.reported = true;
    }
    return { version: FEEDBACK_CONTRACT, action, running, ended, ...(before ? changes(before, after) : none), chat, dialog };
}
