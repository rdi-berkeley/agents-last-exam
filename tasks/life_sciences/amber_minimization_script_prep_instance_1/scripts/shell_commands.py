"""Trace explicit pmemd invocations through a bounded, static subset of Bash.

Branches with unknown conditions must agree on the invocation sequence.
Candidate shell code is parsed, never executed.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field, replace

import bashlex


@dataclass
class _State:
    variables: dict[str, str] = field(default_factory=dict)
    functions: dict = field(default_factory=dict)
    calls: tuple = ()
    status: bool | None = None
    stopped: bool = False
    returning: bool = False

    def fork(self):
        return replace(self, variables=self.variables.copy(), functions=self.functions.copy())


def _word(raw, variables):
    if raw in {"[", "]", "[[", "]]"}:
        return raw
    pieces = []
    quote = None
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == "'" and quote != '"':
            quote = None if quote == "'" else "'"
            i += 1
            continue
        if char == '"' and quote != "'":
            quote = None if quote == '"' else '"'
            i += 1
            continue
        if char == "\\" and quote != "'" and i + 1 < len(raw):
            following = raw[i + 1]
            if quote == '"' and following not in '$`"\\\n':
                pieces.append(char)
                i += 1
                continue
            pieces.append(following)
            i += 2
            continue
        if quote != "'" and (raw.startswith("$(", i) or char == "`"):
            raise ValueError("dynamic shell expansion")
        if char == "$" and quote != "'":
            match = re.match(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)", raw[i:])
            if match:
                value = variables.get(match[1] or match[2], match[0])
                if quote is None and (re.search(r"\s", value) or any(c in value for c in "*?[")):
                    raise ValueError("unquoted field splitting or glob")
                pieces.append(value)
                i += len(match[0])
                continue
        if quote is None and char in "*?[":
            raise ValueError("unexpanded glob")
        pieces.append(char)
        i += 1
    if quote:
        raise ValueError("unterminated quote")
    return "".join(pieces)


class _Trace:
    def __init__(self, script):
        self.script = script
        self.steps = 0

    def run(self, nodes, states, depth=0):
        if depth > 16:
            raise ValueError("recursive shell function")
        operator = ";"
        for node in nodes:
            self.steps += len(states)
            if self.steps > 10000 or len(states) > 64:
                raise ValueError("shell analysis limit")
            if node.kind == "operator":
                operator = node.op
                if operator not in {";", "\n", "&&", "||"}:
                    raise ValueError("unsupported shell operator")
                continue
            updated = []
            for state in states:
                if state.stopped or state.returning:
                    updated.append(state)
                    continue
                desired = operator == "&&"
                if operator in {"&&", "||"}:
                    if state.status is not None and state.status != desired:
                        updated.append(state)
                        continue
                    if state.status is None:
                        skipped = state.fork()
                        skipped.status = not desired
                        updated.append(skipped)
                updated.extend(self.node(node, state.fork(), depth))
            states = updated
            operator = ";"
        return states

    def node(self, node, state, depth):
        if node.kind == "list":
            return self.run(node.parts, [state], depth)
        if node.kind == "compound":
            if getattr(node, "redirects", []):
                raise ValueError("redirected compound command")
            return self.run(node.list, [state], depth)
        if node.kind == "reservedword":
            if node.word not in {"{", "}"}:
                raise ValueError("unsupported compound scope")
            return [state]
        if node.kind == "function":
            state.functions[node.parts[0].word] = node.parts[-1]
            state.status = True
            return [state]
        if node.kind == "if":
            pending, completed = [state], []
            parts, index = node.parts, 0
            while index < len(parts):
                marker = parts[index].word
                if marker in {"if", "elif"}:
                    checked = self.run([parts[index + 1]], pending, depth)
                    pending = []
                    for branch in checked:
                        if branch.stopped or branch.returning:
                            completed.append(branch)
                            continue
                        if branch.status is not False:
                            completed.extend(self.run([parts[index + 3]], [branch.fork()], depth))
                        if branch.status is not True:
                            pending.append(branch.fork())
                    index += 4
                elif marker == "else":
                    completed.extend(self.run([parts[index + 1]], pending, depth))
                    pending = []
                    index += 2
                elif marker == "fi":
                    for branch in pending:
                        branch.status = True
                    return completed + pending
                else:
                    raise ValueError("unsupported conditional")
        if node.kind != "command":
            raise ValueError("unsupported shell construct: " + node.kind)
        assignments = [p for p in node.parts if p.kind == "assignment"]
        words = [p for p in node.parts if p.kind == "word"]
        if assignments and words:
            raise ValueError("command-local assignments")
        for part in assignments:
            raw = self.script[slice(*part.pos)]
            name, value = raw.split("=", 1)
            try:
                state.variables[name] = _word(value, state.variables)
            except ValueError:
                state.variables.pop(name, None)
                directory_expression = (
                    "$(pwd)" in value or '$(dirname "${BASH_SOURCE[0]}")' in value
                )
                if not directory_expression or "pmemd.cuda" in value:
                    raise
        args = [_word(self.script[slice(*p.pos)], state.variables) for p in words]
        if not args:
            state.status = True
            return [state]
        for part in node.parts:
            if part.kind == "redirect":
                if part.type.startswith("<<"):
                    raise ValueError("unsupported heredoc")
                if hasattr(part.output, "pos"):
                    _word(self.script[slice(*part.output.pos)], state.variables)
        executable = args[0]
        if executable == "exec":
            args = args[1:]
            if not args:
                raise ValueError("empty exec")
            executable = args[0]
            state.stopped = True
        if executable in state.functions:
            if len(args) != 1 or state.stopped:
                raise ValueError("unsupported function arguments")
            results = self.run([state.functions[executable]], [state], depth + 1)
            for result in results:
                result.returning = False
            return results
        if executable == "return" and depth == 0:
            raise ValueError("return outside a function")
        if executable in {"exit", "return"}:
            state.stopped = executable == "exit"
            state.returning = executable == "return"
            state.status = len(args) == 1 or args[1] == "0"
        elif executable == "set":
            if any(arg.startswith("-") and "n" in arg for arg in args[1:]):
                state.stopped = True
            state.status = True
        elif executable == "export":
            for part in words[1:]:
                raw = self.script[slice(*part.pos)]
                if "=" in raw:
                    name, value = raw.split("=", 1)
                    state.variables[name] = _word(value, state.variables)
            state.status = True
        elif executable in {"eval", "source", ".", "local", "unset", "read", "trap"}:
            raise ValueError("unsupported shell state mutation")
        elif posixpath.basename(executable) == "pmemd.cuda":
            state.calls += (tuple(args[1:]),)
            state.status = None
        elif executable in {"true", ":"}:
            state.status = True
        elif executable == "false":
            state.status = False
        else:
            state.status = None
        return [state]


def minimization_commands(script):
    try:
        script = script.replace("\\\n", "")
        states = _Trace(script).run(bashlex.parse(script), [_State()])
        sequences = {state.calls for state in states}
        if len(sequences) != 1:
            return []
        return [list(args) for args in sequences.pop()]
    except (ValueError, NotImplementedError, RecursionError, bashlex.errors.ParsingError):
        return []
