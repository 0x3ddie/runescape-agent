"""The part of RuneScript that npc death scripts and their drop tables use.

sim.build parses the [ai_queue3,...] scripts of the area's npcs, and the labels and procs
they reach, from the content scripts into the private archive; the simulator runs them
with `run`, so no drop table lives in this repository. The subset is statements (if/else
chains, def_*/assignment, return, gosub, @label, calls) and expressions (literals,
^constants, $locals, %varps, calls, comparisons, & and |, calc arithmetic). A statement or
call outside it becomes an 'unsupported' node: parsing never fails, but reaching one raises
Unsupported, which the caller records as a coverage failure.
"""
import re

TOKEN = re.compile(r'\s*(?:(//[^\n]*)|("(?:[^"\\]|\\.)*")|(<=|>=|[(){},;=!<>&|+\-*/])|([~@^$%.]?[A-Za-z_][A-Za-z0-9_]*)'
                   r'|(-?\d+))')
HEADER = re.compile(r'^\[(\w+),([^\]]+)\](\([^)]*\))?(\([^)]*\))?', re.M)


def tokens(text):
    out, i = [], 0
    while i < len(text):
        m = TOKEN.match(text, i)
        if not m or m.end() == i:
            if text[i:].strip():
                out.append(('bad', text[i]))
                i += 1
                continue
            break
        i = m.end()
        comment, string, punct, name, number = m.groups()
        if comment:
            continue
        if string:
            out.append(('str', string[1:-1]))
        elif punct:
            out.append(('p', punct))
        elif name:
            out.append(('name', name))
        elif number:
            out.append(('num', int(number)))
    return out


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else ('eof', None)

    def take(self, value=None):
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise SyntaxError(f'expected {value!r}, got {tok}')
        self.i += 1
        return tok

    def block(self):
        out = []
        while self.peek()[0] != 'eof' and self.peek()[1] != '}':
            out.append(self.statement())
        return out

    def statement(self):
        start = self.i
        try:
            return self._statement()
        except SyntaxError:
            # skip to the end of this statement and keep the text for the coverage report
            self.i = start
            depth, text = 0, []
            while self.peek()[0] != 'eof':
                kind, value = self.take()
                text.append(str(value))
                depth += {'{': 1, '}': -1}.get(value, 0) if kind == 'p' else 0
                if depth <= 0 and value in (';', '}') and kind == 'p':
                    break
            return ['unsupported', ' '.join(text)]

    def _statement(self):
        kind, value = self.peek()
        if value == 'if':
            branches, otherwise = [], None
            while True:
                self.take('if')
                self.take('(')
                cond = self.expr()
                self.take(')')
                branches.append([cond, self.body()])
                if self.peek()[1] != 'else':
                    break
                self.take('else')
                if self.peek()[1] != 'if':
                    otherwise = self.body()
                    break
            return ['if', branches, otherwise or []]
        if value == 'return':
            self.take()
            values = []
            if self.peek()[1] == '(':
                self.take('(')
                values = self.args(')')
            self.take(';')
            return ['return', values]
        if kind == 'name' and value.startswith('def_'):
            self.take()
            var = self.take()[1]
            expr = ['num', 0]
            if self.peek()[1] == '=':
                self.take('=')
                expr = self.expr()
            self.take(';')
            return ['set', var, expr]
        if kind == 'name' and value.startswith(('$', '%')) and self.peek(1)[1] == '=':
            self.take()
            self.take('=')
            expr = self.expr()
            self.take(';')
            return ['set', value, expr] if value.startswith('$') else ['setvarp', value[1:], expr]
        if kind == 'name' and value.startswith('@'):
            self.take()
            self.take(';')
            return ['jump', value[1:]]
        if value == 'gosub':
            self.take()
            self.take('(')
            name = self.take()[1]
            self.take(')')
            self.take(';')
            return ['gosub', name]
        if kind == 'name' and not value.startswith(('$', '%', '^', '.')) and value not in ('while', 'switch_int'):
            expr = self.atom()
            self.take(';')
            if expr[0] == 'name':  # a command without arguments, e.g. if_close;
                expr = ['call', expr[1], []]
            if expr[0] != 'call':
                raise SyntaxError('expression statement')
            return ['do', expr]
        raise SyntaxError(f'statement {value}')

    def body(self):
        if self.peek()[1] == '{':
            self.take('{')
            out = self.block()
            self.take('}')
            return out
        return [self.statement()]

    def args(self, close):
        out = []
        while self.peek()[1] != close:
            out.append(self.expr())
            if self.peek()[1] == ',':
                self.take(',')
        self.take(close)
        return out

    # precedence: | < & < comparison < + - < * / < atom
    def expr(self):
        left = self.conj()
        while self.peek()[1] == '|':
            self.take()
            left = ['or', left, self.conj()]
        return left

    def conj(self):
        left = self.compare()
        while self.peek()[1] == '&':
            self.take()
            left = ['and', left, self.compare()]
        return left

    def compare(self):
        left = self.sum()
        if self.peek()[1] in ('=', '!', '<', '>', '<=', '>='):
            op = self.take()[1]
            left = ['cmp', op, left, self.sum()]
        return left

    def sum(self):
        left = self.product()
        while self.peek()[1] in ('+', '-'):
            op = self.take()[1]
            left = ['arith', op, left, self.product()]
        return left

    def product(self):
        left = self.atom()
        while self.peek()[1] in ('*', '/'):
            op = self.take()[1]
            left = ['arith', op, left, self.atom()]
        return left

    def atom(self):
        kind, value = self.take()
        if kind == 'num':
            return ['num', value]
        if kind == 'str':
            return ['str', value]
        if value == '(':
            inner = self.expr()
            self.take(')')
            return inner
        if kind != 'name' or value.startswith('.'):
            raise SyntaxError(f'atom {value}')
        if value.startswith('^'):
            return ['const', value[1:]]
        if value.startswith('$'):
            return ['local', value]
        if value.startswith('%'):
            return ['varp', value[1:]]
        if self.peek()[1] == '(':
            self.take('(')
            return ['call', value, self.args(')')]
        if value.startswith('~'):
            return ['call', value, []]
        return ['name', value]


def script_index(root):
    """Every script header in the content tree: (trigger, subject) -> (file, body, params)."""
    index = {}
    for path in sorted(root.rglob('*.rs2')):
        text = path.read_text()
        heads = list(HEADER.finditer(text))
        for k, m in enumerate(heads):
            end = heads[k + 1].start() if k + 1 < len(heads) else len(text)
            params = re.findall(r'\$(\w+)', m.group(3) or '')
            index.setdefault((m.group(1), m.group(2)), (str(path.relative_to(root)), text[m.end():end],
                                                        ['$' + p for p in params]))
    return index


def parse_scripts(index, roots, native):
    """The parsed scripts reachable from `roots` (script names like '[ai_queue3,goblin]'),
    following gosub, @label and ~proc; names in `native` are run by the simulator itself."""
    out, todo = {}, list(roots)
    while todo:
        name = todo.pop()
        if name in out:
            continue
        trigger, subject = name.strip('[]').split(',', 1)
        found = index.get((trigger, subject))
        if found is None:
            out[name] = dict(file=None, params=[], body=[['unsupported', f'no script {name}']])
            continue
        file, text, params = found
        body = Parser(tokens(text)).block()
        out[name] = dict(file=file, params=params, body=body)
        for ref in references(body):
            if ref not in native:
                todo.append(ref)
    return out


def references(node):
    if isinstance(node, list):
        if node and node[0] == 'gosub':
            yield f'[proc,{node[1]}]'
        elif node and node[0] == 'jump':
            yield f'[label,{node[1]}]'
        elif node and node[0] == 'call' and node[1].startswith('~'):
            yield f'[proc,{node[1][1:]}]'
        for child in node:
            yield from references(child)


def constants(root):
    """^name = value from every .constant file; numbers as ints, the rest as text."""
    out = {}
    for path in sorted(root.rglob('*.constant')):
        for line in path.read_text().splitlines():
            m = re.match(r'\^(\w+)\s*=\s*(.+?)\s*(//.*)?$', line)
            if m:
                value = m.group(2)
                out[m.group(1)] = int(value) if re.fullmatch(r'-?\d+', value) else value
    return out


# ---- evaluation ----------------------------------------------------------------------------
class Return(Exception):
    def __init__(self, values):
        self.values = values


def run(scripts, name, ctx, args=()):
    """Run a parsed script as a generator; it yields the suspensions of native calls."""
    script = scripts.get(name)
    if script is None:
        ctx.unsupported(f'script {name} was not extracted')
    local = dict(zip(script['params'], args))
    try:
        yield from _block(scripts, script['body'], ctx, local)
    except Return as r:
        return r.values
    return []


def _block(scripts, body, ctx, local):
    for stmt in body:
        op = stmt[0]
        if op == 'set':
            local[stmt[1]] = _single((yield from _expr(scripts, stmt[2], ctx, local)))
        elif op == 'setvarp':
            ctx.set_varp(stmt[1], _single((yield from _expr(scripts, stmt[2], ctx, local))))
        elif op == 'if':
            for cond, block in stmt[1]:
                if _single((yield from _expr(scripts, cond, ctx, local))):
                    yield from _block(scripts, block, ctx, local)
                    break
            else:
                yield from _block(scripts, stmt[2], ctx, local)
        elif op == 'return':
            values = []
            for e in stmt[1]:
                values.extend(_flat((yield from _expr(scripts, e, ctx, local))))
            raise Return(values)
        elif op == 'gosub':
            yield from _call(scripts, f'~{stmt[1]}', [], ctx)
        elif op == 'jump':
            yield from run(scripts, f'[label,{stmt[1]}]', ctx)
            raise Return([])
        elif op == 'do':
            yield from _expr(scripts, stmt[1], ctx, local)
        else:
            ctx.unsupported(f"statement '{stmt[1]}'")


def _expr(scripts, e, ctx, local):
    op = e[0]
    if op == 'num' or op == 'str':
        return e[1]
    if op == 'const':
        return ctx.constant(e[1])
    if op == 'local':
        if e[1] not in local:
            ctx.unsupported(f'unset local {e[1]}')
        return local[e[1]]
    if op == 'varp':
        return ctx.varp(e[1])
    if op == 'name':
        return ctx.name(e[1])
    if op == 'and':
        return int(bool(_single((yield from _expr(scripts, e[1], ctx, local))))
                   and bool(_single((yield from _expr(scripts, e[2], ctx, local)))))
    if op == 'or':
        return int(bool(_single((yield from _expr(scripts, e[1], ctx, local))))
                   or bool(_single((yield from _expr(scripts, e[2], ctx, local)))))
    if op == 'cmp':
        a = _single((yield from _expr(scripts, e[2], ctx, local)))
        b = _single((yield from _expr(scripts, e[3], ctx, local)))
        return int({'=': a == b, '!': a != b, '<': a < b, '>': a > b, '<=': a <= b, '>=': a >= b}[e[1]])
    if op == 'arith':
        a = _single((yield from _expr(scripts, e[2], ctx, local)))
        b = _single((yield from _expr(scripts, e[3], ctx, local)))
        return {'+': a + b, '-': a - b, '*': a * b, '/': int(a / b)}[e[1]]
    if op == 'call':
        args = []
        for a in e[2]:
            args.extend(_flat((yield from _expr(scripts, a, ctx, local))))
        return (yield from _call(scripts, e[1], args, ctx))
    ctx.unsupported(f'expression {e}')


def _call(scripts, name, args, ctx):
    if name.startswith('~'):
        native = ctx.native(name[1:])
        if native is not None:
            return (yield from native(*args))
        values = yield from run(scripts, f'[proc,{name[1:]}]', ctx, args)
        return values[0] if len(values) == 1 else tuple(values)
    if name == 'calc':
        return args[0]
    return (yield from ctx.command(name, args))


def _flat(value):
    return list(value) if isinstance(value, tuple) else [value]


def _single(value):
    if isinstance(value, tuple):
        raise TypeError('a pair where one value is expected')
    return value
