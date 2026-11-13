from hypothesis import given, strategies as st
import pytest

from sim import rs2


class Ctx:
    """Commands a drop table uses, with one fixed draw."""

    def __init__(self, draw):
        self.draw, self.added = draw, []

    def unsupported(self, what):
        raise NotImplementedError(what)

    def constant(self, name):
        return dict(lootdrop_duration=200, true=1, false=0)[name]

    def varp(self, name):
        return 0

    def name(self, name):
        return name

    def native(self, name):
        return None

    def command(self, name, args):
        yield from ()
        if name == 'random':
            return self.draw
        if name == 'obj_add':
            self.added.append(tuple(args[1:3]))
            return 0
        self.unsupported(name)


def table(bounds):
    branches = ' else '.join(f'if ($r < {b}) {{ obj_add(npc_coord, item{k}, {k + 1}, ^lootdrop_duration); }}'
                             for k, b in enumerate(bounds))
    return f'def_int $r = random(128);\n{branches}\n'


def run(text, ctx):
    scripts = {'[ai_queue3,x]': dict(file=None, params=[], body=rs2.Parser(rs2.tokens(text)).block())}
    for _ in rs2.run(scripts, '[ai_queue3,x]', ctx):
        pass


@given(st.lists(st.integers(1, 127), min_size=1, max_size=8, unique=True).map(sorted), st.integers(0, 127))
def test_a_threshold_chain_adds_the_first_row_the_draw_is_under(bounds, draw):
    ctx = Ctx(draw)
    run(table(bounds), ctx)
    rows = [k for k, b in enumerate(bounds) if draw < b]
    assert ctx.added == ([(f'item{rows[0]}', rows[0] + 1)] if rows else [])


def test_an_unparsed_statement_fails_only_when_reached():
    text = 'def_int $r = random(128);\nif ($r < 10) { .npc_findall(npc_coord, duck, 1); } else { obj_add(npc_coord, bones, 1, 200); }'
    ctx = Ctx(50)
    run(text, ctx)
    assert ctx.added == [('bones', 1)]
    with pytest.raises(NotImplementedError):
        run(text, Ctx(5))
