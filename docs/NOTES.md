# Notes

How the project got where it is, from the first agent in the real engine to goal-free play, and then
[the results of the 12 November release in detail](#results-in-detail) with how far to trust each number.

## Starting in the real engine

We started at the end of July 2026 by hooking an agent straight into the engine. The first
agent had one policy for woodcutting, mining, fishing and combat, and it was paid in XP. Two
things went wrong. The engine's player has no "is dead" field, so for a while our agent could
die and we never saw it. XP was also the wrong thing to pay for, since an agent can earn XP,
die, lose its axe and logs, and still look good. So in late September we switched to paying
only for items the agent made and still holds at the end, with nothing if it dies. Kills count
too now, and that's the reward we still use.

## Copying a scripted teacher

Next we tried imitation. We wrote a scripted player, had the agent copy it, then fine-tuned
with PPO. Copying worked at Draynor, but PPO fine-tuning made it worse, until it kept almost
no logs. Our best guess was the critic: copying only trains the actor, so PPO starts with a
random critic. The copied policy had also learned Draynor rather than woodcutting. At two
villages it had never seen, it got 0 logs. Training on four villages fixed one held-out bank
(15 of 16 trials at Varrock west) but not the other (0 of 16 at Varrock east). Through early
October we kept adding labels and tricks, with mixed results.

## Learning from outcomes only

On October 13 we dropped the teacher and trained PPO from scratch on short tasks at Draynor,
paid only by the outcome. It learned woodcutting in all 3 seeds and fishing in 1 (the other two
just chopped trees instead). Since then everything here has been trained this way, with no
teacher and no demos.

## The engine is too slow

The catch was speed. The real engine gave about 45 decisions a second with one world on a Mac,
and four worlds together were no faster, since each world eats a whole CPU core. Each of those
fresh PPO runs took about an hour for 65,536 decisions. At that pace, one of our later 4.2
million decision runs would take almost three days.

## Our own simulator: Python, then JAX

So on October 15 we wrote our own simulator: a Python copy of the mechanics our tasks need,
ported from the engine's source and checked against recorded games from the real engine. Its
path finder matched the engine's on all 6,000 test queries. Python tops out around 1,000
decisions a second per core, though.

We also built a JAX version that runs a thousand worlds at once on a GPU. On a rented A100 it trained at 654 decisions a second against 549 for plain Python worker
processes on the same machine, and it spent about 20 minutes compiling before it could take a
step.

## Why the GPU didn't help, and the move to C

Then we looked at how the fast projects do it. [Craftax](https://arxiv.org/abs/2402.16801) puts
the game, the rollouts and PPO in one JAX program and reports about 267,000 steps a second on
one RTX 4090. [Neural MMO 3](https://github.com/PufferAI/PufferLib) in PufferLib is written in
plain C and says it runs millions of steps a second per CPU core. What mattered for us is that every
Craftax step costs about the same, and ours don't. One of our decisions can last 1 tick or over
100. On a GPU all the worlds step together, so the batch waits for the slowest world, and only
about 8% of the work was useful. On CPU cores each world runs on its own clock.

We followed Neural MMO 3 (none of its code is used here) and rewrote the simulator in C,
about 12,000 lines, on October 26 and 27. The Python version stays the reference, and the C
port has to match it decision for decision. Its release check compared 6.3 million decisions
with zero differences. One core now steps about 100,000 random decisions a second, about 100
times the Python version. With a faster PPO update on top, a phase 1 run trains at about 28,000
decisions a second on one GPU, about two and a half minutes a run.

## Detours that didn't pan out

Before the C port, we spent a week trying to teach one policy new skills without it forgetting
the old ones: a novelty bonus (random network distillation), continual backpropagation,
self-distillation and PCGrad. None were good enough to keep. The most useful thing from that
week was about testing. Taking the policy's top move every time (greedy) can give a very
different answer from sampling moves the way it trained. One saved policy scored 23 of 200
greedy and 139 of 200 sampled, so now we only judge policies by sampled play.

## Phase 1: one skill at a time

With fast training we went back to basics: one policy per skill, five seeds each, and pass
marks fixed before training. Wood passed everywhere, in the simulator and in the real engine.
Melee passed in the simulator, but we couldn't build a fair real-engine test for it (no real
spawn puts a mugger near goblins, and nothing resets hitpoints). Fish and mine passed in the
simulator in every seed and mostly failed in the real engine. Real sites are full of things the
training world never had. At Entrana, a second fishing spot has the same name as the net spot
but needs a rod, and most failed fish trials loop on it, though we haven't proven that's the
cause.

We also let the wood agent play the real game for a full hour. It fills its bag within the first 12
minutes and then mostly stands around. Our guess is that training episodes only last 300 ticks
(about 3 minutes), so it never learned what to do with a full bag.

## Clicking instead of walking to named places

The phase 1 controls walk to named places up to 32 tiles away. That's not how a player moves,
and the list of places gives away where things are. In early November we switched to clicking
one of 24 tiles around the player. The click-trained wood policy still passes its real-engine
check, but it's slower to find the next grove and fails the scarce-trees test in two of five
seeds. We kept clicks anyway, as a choice. Mine and melee with clicks learned in the simulator
(one seed each), but mining didn't carry over to the real engine.

## A bug that hid for three weeks

In the game, when something hits you, you hit back automatically. Ours never did, because the
damage code cleared the queued hit-back in the same pass. The Python reference and the C port
had the same bug, so the check that they agree passed anyway. It was there from mid-October, so
phase 1 trained without it. We found it in November while digging into why agents got stuck.

After the fix we replayed the phase 1 policies. The melee ones still pass in every seed (758 to
782 of 800), and the shared four-skill policy now passes melee in all five seeds instead of two.
That doesn't tell us what training with the fix would do, since those weights learned without
it. In the real engine, an idle player hit back in 16 of 16 trials, which is the rule the fix
follows.

## Goal-free play (12 to 21 November)

Every task up to phase 1 told the agent what to do. The next question was whether it would
find things to do by itself, so we dropped the goal. One policy now plays whole lives and is
paid the gain in its own levels, across every stat in the game.

We also moved training into the real engine. The simulator only has the mechanics we ported,
so the agent never sees anything it leaves out, and training in the real server closes that
gap. Many agents of one policy share each world of the real server. The world stops at every tick where an agent needs a new
action and waits while the learner trains. The C simulator got the same goal-free task, still
has to match the Python reference bit for bit, and stays as the fast twin.

The agent picked up four skills on its own: woodcutting, firemaking, fighting and burying
bones. Then it stopped. It never ate when hurt: its actions reach only the 8 pack slots of the
page on screen, and it drops all its food early in a life to free those slots for logs. A
charge for time spent hurt, a pay for first uses (which brought the first cooked food, but got
farmed), lives started at a long life's levels and a phase of extra exploration all failed to
add a skill. Across four tests the agent moved between the skills it knew and never tried a
new one, even with a rock one step away.

What was missing was exploration aimed at what the agent hadn't tried. So we added a small pay
for each decision that gives XP in a stat, worth 10 / (10 + R), where R counts the recent
decisions with XP in that stat. It is big for rare kinds of XP, fades as they become common,
and names no skill. On lives that started at a long life's levels, the share that mined five
ores or more went from 0.7% and 0.0% to 15.1% and 13.6% (two seeds). After we switched the
pay off for 25 more updates, mining stayed and grew, to 15.1% and 47.7%, and those lives
gained more levels than lives that were never paid. The pay gave the first tries, and the
game's own levels kept them.

Limits: two seeds, the long-life starts only (lives at their drawn levels still rarely mine),
and deaths rose on small counts. The agent still doesn't fish, cook or eat when hurt.

## Results in detail

These are the results of the 12 November release, the one-skill-at-a-time policies.

### How to read the numbers

A success is counted on records (start tile, levels, kit) drawn before training and never trained
on: "172 of 200" means 172 successes on 200 such records. Sampled play draws each action from the
policy, as in training; a verdict needs two sampled passes, and both must pass. Greedy play always
takes the top action and is only reported beside ([why](#greedy-and-sampled-play)). "Native" means
the real engine. The native check is 32 trials, 16 at each of two sites, and needs 26. Training,
gates and native checks run at 25 times the normal experience rate. Points are percentage points.

Wood is 300 ticks among three groves; success is 8 logs and staying alive. In `ample` the start
grove is as drawn; in `scarce` one normal tree is left, so the agent has to find another grove.
Fish and mine are a 240-tick task: get 2 of the asked-for item first, keep them, stay alive. Melee
is 5 kills in 1,200 ticks against goblins and muggers that hit back, with health and food carried
between fights, on 800 records in four start groups of 200 (healthy, hurt with food, hurt without
food, a mugger nearby). Code: [sim/search.py](../sim/search.py), [sim/combat.py](../sim/combat.py).

### Phase 1 numbers

One specialist per skill and one shared four-goal policy, five seeds each, 4,194,304 decisions a
run, with the older controls: compass moves plus walks to named places up to 32 tiles away. A skill
is supported when its specialist passes the simulator gate in all five seeds and the native check
in at least four, both sampled passes each time.

| Skill | Simulator, sampled | Native check, sampled | Greedy, reported only | Verdict |
| --- | --- | --- | --- | --- |
| Wood | Every seed passes: `ample` 197-200, `scarce` 184-195 of 200 (gate 160) | Every seed passes: 31-32 of 32 | - | Supported |
| Fish | Every seed passes (fish and mine 191-200 of 200) | Every seed fails: 16-25 of 32 | Native 29-32 of 32 | Not supported |
| Mine | Every seed passes | 19-30 of 32; under 26 in a pass in 4 of 5 seeds | Native 25-32 of 32 | Not supported |
| Melee | Every seed passes: 755-774 of 800, each group at least 176 of 200 | None exists | Simulator 656-775 of 800 | Supported, simulator only |

Melee has no native check: no native spawn puts a mugger 6-12 tiles from goblins, nothing spawns
npcs, and no reset sets hitpoints. The fish and mine short tasks are near saturation in the
simulator; the native check is where they fail. Limits: 32 native trials is a small sample (near 26
the binomial standard deviation is about 2.3, and a repeat of the check moves about 2), fish's 32
native records are 16 starts on two engine builds, and melee trained before the
[auto-retaliate fix](#a-bug-that-hid-for-three-weeks).

### One policy for four skills

In the simulator the shared policy passed wood, fish and mine in all five seeds and melee in two
(the misses were all the mugger group in one pass, 135-139 of 200 against 140). Natively it failed
wood (9-26 of 32) and fish (9-21) in all five seeds and passed mine in two (26-28). Interference at
equal experience per skill (a specialist at a quarter of its run against the shared final) was not
shown either way. At equal total compute the melee specialist beat the shared policy by 8.75 to
13.88 points in every seed and pass (each lower bound at least +6.12); for the other skills,
specialist minus shared ran from -4.00 to +5.00.

### Why fish and mine fail in the real engine

Open. This is an association in 1,920 saved native trials (replayed exactly), not tested by training.

- Native sites hold many things the training world never shows (about 8 of the 12 listed places).
  At Entrana a second fishing spot has the net spot's name but offers Lure and Bait (both need a rod).
  The policies put 0.14-0.23 of their probability on such things at the goal site (0.18-0.26 on the
  goal action), and sampled play takes them 5.5-5.9 times a trial, greedy 2.5.
- Fish trials with 5 or fewer such actions succeeded 174 of 200; with 6 or more, 34 of 120 (mine:
  171 of 195 against 85 of 125). 75 of 112 sampled fish failures loop at the Lure/Bait spot,
  against 8 of 208 successes. But 33 of those 75 never had the Net option in range, so a loop that
  causes a failure cannot be told from a failing policy that loops. Had every loop succeeded, three
  seeds would reach 26 in both passes; four are needed.
- Adding native-like content to a simulator world, with fixed policies, raised off-goal actions to
  native levels in only 3 of 5 seeds, and the fish drop showed in no seed: a failed manipulation.

### Clicks instead of place walks

The current controls walk by clicking one of 24 tiles around the player (4, 9 and 16 tiles out in
eight directions; [rl/act_contract.py](../rl/act_contract.py)) and act on things within 15 tiles.

**Wood, five seeds.** The click-trained wood policy passes its native check (28-32 of 32) and
`ample` (189-196 of 200) in every seed, but misses `scarce` in two: 147 and 152 in seed 101, 159 in
one pass of seed 202, against 160. So it fails the wood rule; the phase 1 specialists pass the same
records (`scarce` 191-196). Against them it loses 32.7 of 200 on `scarce` on average (SD 7.5 over
seeds) and 1.3 of 32 natively. It did learn to leave an empty grove (it went on to logs elsewhere in
194-200 of 194-200 such trials per seed and pass) but runs out of time between groves: a median
56-57 ticks to the next log, against 41-43 for phase 1; 84-88% of its failures ended with 5-7 logs.

Our guess was that it lost the place list's knowledge of where the next tree stands. For scripts that
holds: on 100 `scarce` records, a click script with no tree knowledge made 77, one that knows trees
within 25 tiles 94, within 32 tiles 97, an oracle 99; the click policy made 71-82, phase 1 94-98. But
14.6 of the policy's 20.2-record gap lies on records the blind script already wins, so this does not
show knowledge is what the policy lacks. The native check asks for little walking (0.6-3.5 clicks a
trial, against 6.4-8.2 in `scarce`). Clicks became the default anyway: a choice, not a pass.

**Mine and melee, one seed each,** trained with the same recipe on a Mac, 4,194,304 decisions
each, on the simulator after the retaliate fix:

| Read | Sampled, two passes | Greedy | Untrained start | Line |
| --- | --- | --- | --- | --- |
| Simulator melee, of 800 | 729, 713 | 663 | 15, 19 | 640 |
| Simulator mine, of 200 | 172, 176 | 148 | 22, 28 | 160 |
| Native mine check, of 32 | 17, 16 | 21 | - | 26 |

Both learned in the simulator; native mining did not transfer. At Rimmington the starts lie 8-14
tiles from the nearest rock, and 33 of 48 sampled trials never asked to mine. Once a click or action
left no rock to mine in the list, one came back within 12 decisions in 4 of 37 cases, against 166 of
184 in training; why is not shown. At Varrock south-east it made 13 and 14 of 16. The native combat tests
gave 1 of 16 in both cells but are not the trained task: one has chickens behind a closed gate
(training had none), the other starts 25-40 tiles from goblins, past the 15-tile list. Limits: one
seed each, and the Mac's update path is not the GPU path.

**Retraining mine on three seeds.** The idea: the small training world (8 layouts of 48 x 48 tiles,
rocks east of 69% of their start tiles) causes the mine gap. At fixed weights, on 64 new Rimmington
starts per side, the one-seed policy made 32 and 35 with rocks to the west and 49 and 46 with rocks
to the east, and its clicks went east 1.7-1.9 times as often as west (greedy play did not show the
side difference; a script made 16 of 16). The rule, set before training: retrain the same recipe on
a cloud GPU at three seeds; if two hold the rocks-west cell at 52 of 64 in both passes, the gap
belongs to that one run, and the bigger-world run (128 layouts) does not train. Seeds 101 (54, 58)
and 303 (57, 53) held it, 202 (48, 55) did not, with simulator mine at 177-186 of 200. So the study
stopped, and the small-world idea was never tested. Limits: one site, the one that suggested the
idea; thin margins (seed 303 held by one trial); with one Mac run, the Mac's update path cannot be
told apart from run-to-run noise.

### Long real-engine play

The phase 1 wood specialist and the shared policy (fixed weights, five seeds) played hour-long native
episodes at Barbarian Village, three passes (two sampled, one greedy) of 40 hours each. In the first
1,200 ticks they made 21.4-24.5 and 18.9-24.6 logs, then 0.00-0.75 and 0.00-2.69 in each later
1,200-tick window. The specialist filled its inventory in the first window in 39, 40 and 40 of 40
hours, then mostly stalled. It is not mainly deaths: per 1,000 ticks alive it made 0.31, 0.33 and
0.14 logs after the first window, against 19.7, 19.3 and 18.4 in it. It died in 26, 25 and 11 of 40
hours, most often to an aggressive npc named Tree that cannot be chopped; the shared policy died in
30, 31 and 20, mostly after asking to attack its killer's kind. Untested guesses: training lasts 300
ticks, but the inventory filled at a median of 435-467 ticks; and the training reward stops at 20
logs. Limits: one site, fixed weights, older controls.

### Speed

- **C simulator** ([sim/native/](../sim/native/rs.h)). Its release check compared 429 cases,
  6,305,640 decisions and 61,669,351 ticks against the Python reference with 0 differences. On a
  Mac one worker steps 104,055-106,448 random decisions a second against 1,020 for Python, about 100
  times faster. The agreement check ([sim/native_harness.py](../sim/native_harness.py)) cannot
  catch a mistake both share.
- **Learner.** One 65,536-decision update at minibatch 256 on the GPU below: 41.5 s with stock
  Stable-Baselines3, 12.2-13.1 s on our earlier path, 9.6 s with the new update (bit-equal to stock
  on CPU), 5.5 s with CUDA graphs.
- **Training rate.** On a cloud machine with one NVIDIA RTX PRO 6000 Blackwell GPU and 48 vCPUs,
  with 16 C workers: 36,174 decisions a second at minibatch 1,024, against 2,424-2,510 with the
  Python simulator (which capped it near 6,200). The phase 1 runs trained at 27,597-29,286. The
  real engine, for comparison, ran about 39-41 decisions a second on a Mac (one short sample).

### Greedy and sampled play

The policy is trained by sampling, and taking the top action instead gives a different policy. On
the same 200 hidden-tree search records, one saved policy (a seed's starting policy in the search
study) made 23 successes greedy and 139 sampled: 58 percentage points apart (116 of 200 records).
Two of three earlier retention losses disappeared under sampling; one persisted (-13.5 and -15.0
points). Greedy play can sit in confident loops by its start tile, repeating WAIT, inventory pages
or style changes. Two sampled passes also differ, by 6-11 of 200. Now sampled play decides gates.

### Before phase 1, and methods beside plain PPO

Earlier studies used three seeds, mostly with greedy gates. Four-goal short-task policies passed
all 12 seed-goal conditions on 200 new layouts (164-200), but natively 17 of 27. No seed passed
every sustained-combat gate. Plain PPO learned hidden-tree search (greedy 196, 189 and 197 of 200)
and 300-tick wood (178-197 per variant), losing some combat skill. Tried beside it, none adopted:

- **Random network distillation.** About 2.2 times as many decisions per episode, a shift to short
  interface actions, fewer kills, and 21.5 to 41 points of combat lost against the start in every
  seed; search was not better. Likely reason, not tested: the bonus is paid per decision.
- **Continual backpropagation,** with that bonus and then alone. Alone it showed no combat benefit.
- **Self-distillation** kept slightly more of the earlier
  skills and blocked the new learning.
- **Targeted self-distillation** learned the new task, but greedy search fell to 141 and 139 of 200
  in two seeds (sampled 188 and 192), and it missed the combat floor and the recovery bounds.
- **PCGrad** met every rule in two seeds and lost combat in the third: 583 and 568 of 800 against
  its start's 659 and 654. A follow-up gradient check was inconclusive. The code for the bonus,
  continual backpropagation, the distillation and PCGrad is not in this repository.

### What isn't done yet

- No training run has reached a bank, and bank actions pay nothing; a script showed the click
  controls can bank (deposit, withdraw, walk back, chop again) in 16 of 16. No action reaches a
  shop's buy or sell buttons: Trade opened the shop in 16 of 16, then only WAIT and close were legal.
- Mine and melee under clicks are one-seed runs (three for the mine retrain), fish has no click
  policy, melee has no faithful native check, and every one of these tasks gives the agent its goal.
