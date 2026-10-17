"""Static readers for the two Amber workflow contracts; no submitted code is run."""

from __future__ import annotations

import copy
import math
import posixpath
import re
import shlex
import subprocess
from dataclasses import dataclass

import pyparsing as pp


SHELL_CONTRACT = """Supported shell contract (these artifacts are checked statically, not executed):
- Use bash with literal command names/paths and standalone scalar assignments. Single/double quotes, $NAME, ${NAME}, ${NAME:-literal_default}, and backslash continuations are supported. Keep each quoted shell word on one physical line.
- For script-relative paths, use $0 or ${BASH_SOURCE[0]}, dirname, pwd, or $(cd <resolved-path> && pwd). These are the only supported command substitutions. Other computed paths, arithmetic/array/glob expansion, eval, sourced scripts, and nested shell execution are not supported.
- Use sequential commands, &&/||, if/then/else/fi, and groups. Conditions may use true/false, command -v, [ ... ], [[ ... ]], or test with !, -f/-s/-e/-d/-x, and literal =/==/!= comparisons. File-existence guards must build when either topology or coordinates are missing and skip when both exist.
- Simple called helper functions, positional arguments, quoted "$@", local assignments, shift, and return are supported. Do not use loops, case/elif, traps, aliases, pipelines, background jobs, or recursive helpers. Use set only for -e, -u and -o pipefail.
- Supported commands are module load/add/purge, mkdir, mv, cp, rm, cat, printf, echo, ls, pwd, nvidia-smi, and the required Amber tools. Amber tools may be invoked directly, through command/exec/env, or through the provided run_ambertools.sh <tool> wrapper. Diagnostic commands do not establish workflow steps.
- Generate control text with cat heredocs or printf '%s\\n'/'%s' and > or >>. Heredocs may use quoted delimiters (literal content), or unquoted delimiters with simple $NAME/${NAME} expansion. Input redirection with < is supported. Do not redirect file descriptors or use process substitution.
- Put #SBATCH long-option directives before all executable statements. Preserve filename and variable-name case. Keep any required MMGBSA control text inside the submitted script; do not rely on undeclared external control files.
"""


class ContractError(ValueError):
    pass


def number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value) if math.isfinite(value) else None


def namelists(text: str, *, mmpbsa: bool = False) -> dict[str, dict | list[dict]]:
    """Read singleton groups and ordered Amber &wt records ending in TYPE='END'."""
    identifier = pp.Word(pp.alphas + "_", pp.alphanums + "_")
    exponent = "eE" if mmpbsa else "eEdD"
    numeric = pp.Regex(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[" + exponent + r"][+-]?\d+)?")
    numeric.set_parse_action(lambda tokens: float(tokens[0].lower().replace("d", "e")))
    value = (
        numeric
        | pp.QuotedString("'", esc_quote="''", multiline=True)
        | pp.QuotedString('"', esc_quote='""', multiline=True)
    )
    if mmpbsa:
        value |= pp.Regex(r"[!:][^,\s/'\"=]*")
    assignment = pp.Group(identifier + pp.Suppress("=") + value)
    if mmpbsa:
        body = pp.Optional(pp.DelimitedList(assignment, allow_trailing_delim=True))
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    else:
        body = pp.ZeroOrMore(assignment + pp.Optional(pp.Suppress(",")))
        body.ignore(pp.Regex(r"![^\n]*"))
    groups = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line.startswith(("&", "$")):
            continue
        header = re.match(r"[&$]([A-Za-z_][A-Za-z0-9_]*)(.*)", line)
        if not header:
            raise ContractError("malformed namelist header")
        name, rest = header.groups()
        name = name.lower()
        repeated_weight = name == "wt" and not mmpbsa
        if name == "end" or name in groups and not repeated_weight:
            raise ContractError("duplicate or unexpected namelist")
        if name == "wt" and mmpbsa:
            raise ContractError("&wt is not an MMPBSA namelist")
        chunks = [rest]
        terminator = (
            pp.Suppress("/")
            | pp.CaselessLiteral("&end").suppress()
            | pp.CaselessLiteral("$end").suppress()
        )
        grammar = body + terminator
        if not mmpbsa:
            grammar.ignore(pp.Regex(r"![^\n]*"))
        while True:
            try:
                parsed = grammar.parse_string("\n".join(chunks), parse_all=True)
                break
            except pp.ParseBaseException:
                if (
                    index >= len(lines)
                    or lines[index].lstrip().startswith(("&", "$"))
                    and not lines[index].strip().lower().startswith(("&end", "$end"))
                ):
                    raise ContractError(f"malformed or unterminated &{name}") from None
                chunks.append(lines[index])
                index += 1
        params = {}
        for key, value in parsed:
            key = key.lower()
            if key in params:
                raise ContractError(f"duplicate {name}.{key}")
            params[key] = value
        if repeated_weight:
            kind = params.get("type")
            if not isinstance(kind, str) or not kind.strip():
                raise ContractError("&wt requires a nonempty quoted TYPE")
            weights = groups.setdefault("wt", [])
            if weights and weights[-1]["type"].rstrip() == "END":
                raise ContractError("unexpected &wt after TYPE='END'")
            weights.append(params)
        else:
            groups[name] = params
    if "wt" in groups and groups["wt"][-1]["type"].rstrip() != "END":
        raise ContractError("&wt records must finish with TYPE='END'")
    return groups


def leap_commands(text: str) -> list[list[str]]:
    lexer = shlex.shlex(text, posix=True, punctuation_chars="=;\n")
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    commands, current = [], []
    try:
        for token in lexer:
            if not token.strip(";\n"):
                if current:
                    commands.append(current)
                current = []
            else:
                current.append(token)
        if current:
            commands.append(current)
    except ValueError as error:
        raise ContractError(f"malformed LEaP: {error}") from error
    return commands


def path(value: str, cwd: str = "/script") -> str:
    """Resolve paths using the Linux sandbox's single-root semantics."""
    if "__UNKNOWN__" in value or "$" in value or any(char in value for char in "*?\n"):
        raise ContractError(f"unresolved path: {value}")
    return posixpath.normpath("/" + posixpath.join(cwd, value).lstrip("/"))


def options(argv: list[str], flags: set[str] | None = None) -> dict[str, str]:
    if flags is None:
        flags = {"-O", "-s"} if posixpath.basename(argv[0]) == "tleap" else {"-O"}
    result = {}
    index = 1
    while index < len(argv):
        key = argv[index]
        attached = None
        if key.startswith("--") and "=" in key:
            key, attached = key.split("=", 1)
        if not key.startswith("-") or key in result:
            raise ContractError(f"unexpected or duplicate command option: {key}")
        index += 1
        if attached is not None:
            result[key] = attached
        elif key in flags:
            result[key] = ""
        else:
            if index == len(argv) or argv[index].startswith("-"):
                raise ContractError(f"missing value for {key}")
            result[key] = argv[index]
            index += 1
    return result


_single = pp.QuotedString("'", multiline=True, unquote_results=False)
_double = pp.Forward()
_substitution = pp.Forward()
_escape = pp.Regex(r"\\[\s\S]")
_parameter = pp.Regex(r"\$\{[^}\n]+\}|\$[A-Za-z_][A-Za-z0-9_]*|\$[0-9@?#]")
_double <<= pp.Combine(
    '"'
    + pp.ZeroOrMore(
        _substitution | _escape | pp.CharsNotIn('"\\$', min=1) | _parameter | pp.Literal("$")
    )
    + '"',
    adjacent=True,
)
_substitution <<= pp.Combine(
    "$(" + pp.ZeroOrMore(_single | _double | _substitution | pp.CharsNotIn("()'\"", min=1)) + ")",
    adjacent=True,
)
_piece = (
    _single
    | _double
    | _substitution
    | _parameter
    | _escape
    | pp.Regex(r"[^\s$'\"\\;&|<>()]+")
    | pp.Literal("$")
)
_word = pp.Combine(pp.OneOrMore(_piece), adjacent=True)
_operator = pp.one_of("<<- << >> && || ; & | < > ( )")
_tokens = pp.ZeroOrMore(_word | _operator).ignore(pp.python_style_comment)
_tokens.set_whitespace_chars(" \t\r")


@dataclass
class Command:
    argv: list[str]
    cwd: str
    stdin: str | None
    files: dict[str, str]
    guards: tuple[str, ...]
    products: dict[str, str]


class Shell:
    """Resolve literal workflow data and reachable commands in a bounded Bash subset."""

    def __init__(self, script: str, files: dict[str, str] | None = None):
        checked = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-n"],
            input=script,
            text=True,
            capture_output=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin"},
        )
        if checked.returncode or "here-document" in checked.stderr:
            raise ContractError("invalid bash syntax")
        self.env = {
            "0": "/script/submit.sh",
            "BASH_SOURCE[0]": "/script/submit.sh",
            "BASH_SOURCE": "/script/submit.sh",
            "AMBERHOME": "/amber",
            "SLURM_SUBMIT_DIR": "/script",
        }
        self.cwd = "/script"
        self.files = {path(name): content for name, content in (files or {}).items()}
        self.products: dict[str, str] = {}
        self.directories = {"/script", "/amber"}
        self.commands: list[Command] = []
        self.functions = {}
        self.guards: tuple[str, ...] = ()
        self.stopped = False
        self.arguments = []
        self.local_scopes = []
        self.returning = False
        self.errexit = False
        self.budget = 2000
        self.heredocs = {}
        tokens = []
        lines = iter(script.replace("\\\n", "").splitlines())
        for line in lines:
            try:
                words = _tokens.parse_string(line, parse_all=True).as_list()
            except pp.ParseBaseException as error:
                raise ContractError(f"unsupported shell syntax: {line}") from error
            for index, token in enumerate(words):
                if token not in {"<<", "<<-"}:
                    continue
                raw = words[index + 1]
                delimiter = shlex.split(raw)[0]
                content = []
                for body in lines:
                    if token == "<<-":
                        body = body.lstrip("\t")
                    if body == delimiter:
                        break
                    content.append(body)
                else:
                    raise ContractError("unterminated heredoc")
                key = f"__HEREDOC_{len(self.heredocs)}__"
                self.heredocs[key] = ("\n".join(content) + "\n", raw != delimiter)
                words[index + 1] = key
            tokens.extend(words + [";"])
        self.tokens = tokens
        self.index = 0
        tree = self._sequence(set())
        self._run(tree)

    def _sequence(self, endings: set[str]) -> list:
        result = []
        while self.index < len(self.tokens):
            token = self.tokens[self.index]
            if token in endings:
                break
            if token == ";":
                self.index += 1
                continue
            if token == "if":
                self.index += 1
                condition = self._sequence({"then"})
                self.index += 1
                yes = self._sequence({"else", "fi"})
                no = []
                if self.index < len(self.tokens) and self.tokens[self.index] == "else":
                    self.index += 1
                    no = self._sequence({"fi"})
                self.index += 1
                node = ("if", condition, yes, no)
            elif token in {"(", "{"}:
                self.index += 1
                nested = self._sequence({")" if token == "(" else "}"})
                self.index += 1
                node = ("group", nested, token == "(")
            elif self.tokens[self.index + 1 : self.index + 4] == ["(", ")", "{"]:
                self.index += 4
                body = self._sequence({"}"})
                self.index += 1
                node = ("function", token, body)
            else:
                if token in {"for", "while", "until", "case", "elif", "select", "function"}:
                    raise ContractError(f"unsupported shell control flow: {token}")
                words = []
                while (
                    self.index < len(self.tokens)
                    and self.tokens[self.index] not in {";", "&&", "||", "|", "&"} | endings
                ):
                    if self.tokens[self.index] == "[[":
                        while self.index < len(self.tokens) and self.tokens[self.index] != "]]":
                            if self.tokens[self.index] != ";":
                                words.append(self.tokens[self.index])
                            self.index += 1
                    words.append(self.tokens[self.index])
                    self.index += 1
                if not words:
                    raise ContractError("unsupported shell operator")
                node = ("command", words)
            connector = ";"
            if self.index < len(self.tokens) and self.tokens[self.index] in {"&&", "||", "|", "&"}:
                connector = self.tokens[self.index]
                self.index += 1
                if connector in {"|", "&"}:
                    raise ContractError("pipeline/background workflow requires explicit commands")
            result.append((node, connector))
        return result

    def expand(self, raw: str) -> str:
        def substitute(source: str) -> str:
            try:
                tokens = _tokens.parse_string(source, parse_all=True).as_list()
                values = [self.expand(token) for token in tokens]
            except (pp.ParseBaseException, ContractError):
                return "__UNKNOWN__"
            if values == ["pwd"] or values == ["pwd", "-P"]:
                return self.cwd
            if len(values) == 2 and values[0] == "dirname":
                return posixpath.dirname(values[1])
            if values[:1] == ["cd"] and values[-2:] in (["&&", "pwd"], [";", "pwd"]):
                target = [item for item in values[1:-2] if item != "--"]
                if len(target) == 1:
                    return path(target[0], self.cwd)
            return "__UNKNOWN__"

        def expand_parts(source: str, quoted: bool = False) -> str:
            parts = (
                _double
                | _single
                | _substitution
                | _parameter
                | _escape
                | pp.Regex(r"[^$'\"\\]+")
                | pp.Literal("$")
            ).leave_whitespace()
            try:
                pieces = pp.OneOrMore(parts).leave_whitespace().parse_string(source, parse_all=True)
            except pp.ParseBaseException as error:
                raise ContractError("unsupported shell expansion") from error
            result = ""
            for piece in pieces:
                if piece.startswith("$("):
                    result += substitute(piece[2:-1])
                elif piece.startswith("${") or piece.startswith("$") and len(piece) > 1:
                    key = piece[2:-1] if piece.startswith("${") else piece[1:]
                    if ":-" in key:
                        key, default = key.split(":-", 1)
                        result += self.env.get(key) or expand_parts(default)
                    else:
                        result += self.env.get(key, "__UNKNOWN__")
                elif piece.startswith("'") and not quoted:
                    result += piece[1:-1]
                elif piece.startswith('"') and not quoted:
                    result += expand_parts(piece[1:-1], True) if piece[1:-1] else ""
                elif piece.startswith("\\"):
                    result += piece[1:] if not quoted or piece[1:] in '$`"\\\n' else piece
                else:
                    result += piece
            return result

        return expand_parts(raw) if raw else ""

    def _run(self, tree: list, *, conditional: bool = False) -> bool:
        status = True
        previous = ";"
        for node, connector in tree:
            self.budget -= 1
            if self.budget < 0:
                raise ContractError("shell analysis limit")
            if self.stopped:
                break
            skip = previous == "&&" and not status or previous == "||" and status
            previous = connector
            if skip:
                continue
            if node[0] == "command":
                status = self._command(node[1])
            elif node[0] == "function":
                self.functions[node[1]] = node[2]
            elif node[0] == "group":
                saved = (self.cwd, self.env.copy(), self.stopped)
                status = self._run(node[1])
                if node[2]:
                    self.cwd, self.env, self.stopped = saved
            else:
                saved_guards = self.guards
                condition = self._run(node[1], conditional=True)
                self.guards += tuple(
                    " ".join(item[0][1]) for item in node[1] if item[0][0] == "command"
                )
                status = self._run(node[2] if condition else node[3])
                self.guards = saved_guards
            if not status and self.errexit and connector == ";" and not conditional:
                self.stopped = True
        return status

    def _command(self, raw: list[str]) -> bool:
        if (
            raw
            and "=" in raw[0]
            and len(raw) > 1
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", raw[0], re.S)
        ):
            if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word, re.S) for word in raw):
                raise ContractError("use standalone shell assignments before workflow commands")
        while raw and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", raw[0], re.S):
            key, value = raw[0].split("=", 1)
            self.env[key] = self.expand(value)
            raw = raw[1:]
        if not raw:
            return True
        if raw[0] in {"export", "readonly", "local"}:
            for item in raw[1:]:
                if "=" in item:
                    key, value = item.split("=", 1)
                    if raw[0] == "local":
                        if not self.local_scopes:
                            raise ContractError("local outside a function")
                        self.local_scopes[-1].setdefault(key, self.env.get(key))
                    self.env[key] = self.expand(value)
            return True
        argv, stdin, output, append = [], None, None, False
        index = 0
        while index < len(raw):
            token = raw[index]
            if token in {'"$@"', '"${@}"'}:
                argv.extend(self.arguments)
                index += 1
                continue
            if token in {"<", ">", ">>", "<<", "<<-"}:
                target = raw[index + 1]
                index += 2
                if token.startswith("<<"):
                    stdin, quoted = self.heredocs[target]
                    if not quoted:
                        stdin = re.sub(
                            r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)",
                            lambda match: self.env.get(match[1] or match[2], "__UNKNOWN__"),
                            stdin,
                        )
                        if "$(" in stdin or "`" in stdin:
                            raise ContractError("dynamic heredoc content")
                elif token == "<":
                    stdin = self.files.get(path(self.expand(target), self.cwd))
                else:
                    output = path(self.expand(target), self.cwd)
                    append = token == ">>"
                continue
            argv.append(self.expand(token))
            index += 1
        if output:
            self.products.pop(output, None)
            if not append:
                self.files[output] = ""
        if not argv:
            if output:
                self.files[output] = ""
            return True
        executable = posixpath.basename(argv[0])
        if executable in {"command", "exec", "env"}:
            if executable == "exec":
                self.stopped = True
            if executable == "command" and argv[1:2] == ["-v"]:
                return True
            argv = argv[1:]
            while argv and "=" in argv[0]:
                argv = argv[1:]
            if not argv:
                return True
            executable = posixpath.basename(argv[0])
        if executable == "run_ambertools.sh":
            argv = argv[1:]
            if not argv:
                raise ContractError("missing AmberTools tool")
            executable = argv[0]
        if executable in self.functions:
            if len(self.local_scopes) >= 20:
                raise ContractError("function recursion limit")
            saved = (
                {key: value for key, value in self.env.items() if key.isdigit()},
                self.arguments,
            )
            self.local_scopes.append({})
            self.arguments = argv[1:]
            self.env.update({str(index): value for index, value in enumerate(argv[1:], 1)})
            result = self._run(self.functions[executable])
            for key, value in self.local_scopes.pop().items():
                if value is None:
                    self.env.pop(key, None)
                else:
                    self.env[key] = value
            for key in list(self.env):
                if key.isdigit():
                    del self.env[key]
            positional, self.arguments = saved
            self.env.update(positional)
            if self.returning:
                self.returning = self.stopped = False
            return result
        if executable == "shift":
            count = int(argv[1]) if len(argv) == 2 else 1
            self.arguments = self.arguments[count:]
            for key in list(self.env):
                if key.isdigit() and key != "0":
                    del self.env[key]
            self.env.update({str(index): value for index, value in enumerate(self.arguments, 1)})
            return True
        if executable == "set":
            for argument in argv[1:]:
                if argument == "pipefail":
                    continue
                if not argument.startswith(("-", "+")) or any(
                    letter not in "euo" for letter in argument[1:]
                ):
                    raise ContractError("unsupported set option")
                if "e" in argument:
                    self.errexit = argument.startswith("-")
            return True
        if executable in {"false", "true", ":"}:
            return executable != "false"
        if executable in {"exit", "return"}:
            self.stopped = True
            self.returning = executable == "return"
            if self.returning and not self.local_scopes:
                raise ContractError("return outside a function")
            return argv[1:] in ([], ["0"])
        if executable == "!":
            return not self._command(raw[1:])
        if executable in {"[", "[[", "test"}:
            args = argv[1:-1] if executable != "test" else argv[1:]
            for operator in ("||", "&&"):
                if operator in args:
                    split = args.index(operator)
                    left = self._command(["test", *args[:split]])
                    right = self._command(["test", *args[split + 1 :]])
                    return left or right if operator == "||" else left and right
            negate = args[:1] == ["!"]
            if negate:
                args = args[1:]
            if len(args) == 2 and args[0] in {"-f", "-s", "-e", "-d", "-x"}:
                target = path(args[1], self.cwd)
                result = bool(self.files.get(target)) if args[0] == "-s" else target in self.files
                result = result or target in self.products
                if args[0] == "-d":
                    result = target in self.directories
                elif args[0] == "-e":
                    result = result or target in self.directories
                if args[0] == "-x":
                    result = posixpath.basename(target) in {
                        "run_ambertools.sh",
                        "tleap",
                        "pmemd.cuda",
                        "MMPBSA.py",
                    }
            elif len(args) == 3 and args[1] in {"=", "==", "!="}:
                result = (args[0] == args[2]) != (args[1] == "!=")
            else:
                raise ContractError("unsupported shell condition")
            return not result if negate else result
        if executable == "cd":
            target = [value for value in argv[1:] if value != "--"]
            if len(target) != 1:
                raise ContractError("unresolved working directory")
            self.cwd = path(target[0], self.cwd)
            return True
        if executable in {"eval", "bash", "sh", "source", "."}:
            raise ContractError("dynamic shell execution is not statically verifiable")
        if executable not in {
            "pmemd.cuda",
            "tleap",
            "MMPBSA.py",
            "ante-MMPBSA.py",
            "cpptraj",
            "module",
            "mkdir",
            "mv",
            "cp",
            "rm",
            "cat",
            "printf",
            "echo",
            "set",
            "ls",
            "pwd",
            "nvidia-smi",
        }:
            raise ContractError(f"unsupported shell command: {executable}")
        if executable in {"pmemd.cuda", "tleap", "MMPBSA.py", "ante-MMPBSA.py", "cpptraj"}:
            path(argv[0], self.cwd)
        self.commands.append(
            Command(argv, self.cwd, stdin, copy.copy(self.files), self.guards, self.products.copy())
        )
        if output:
            self.products.pop(output, None)
        if executable == "mkdir":
            self.directories.update(
                path(value, self.cwd) for value in argv[1:] if not value.startswith("-")
            )
        elif executable == "cat" and output:
            if stdin is None:
                try:
                    stdin = "".join(self.files[path(value, self.cwd)] for value in argv[1:])
                except KeyError as error:
                    raise ContractError("unresolved cat input") from error
            self.files[output] = (self.files.get(output, "") if append else "") + stdin
        elif executable == "printf" and output:
            if len(argv) < 3 or argv[1] not in {"%s\\n", "%s"}:
                raise ContractError("unsupported printf format")
            content = ("\n" if argv[1] == "%s\\n" else "").join(argv[2:])
            if argv[1] == "%s\\n":
                content += "\n"
            self.files[output] = (self.files.get(output, "") if append else "") + content
        elif output:
            content = " ".join(argv[1:]) + "\n" if executable == "echo" else "__UNKNOWN__"
            self.files[output] = (self.files.get(output, "") if append else "") + content
        elif executable in {"cp", "mv"}:
            args = [value for value in argv[1:] if not value.startswith("-")]
            if len(args) >= 2:
                target = path(args[-1], self.cwd)
                for value in args[:-1]:
                    source = path(value, self.cwd)
                    destination = (
                        posixpath.join(target, posixpath.basename(source))
                        if target in self.directories or args[-1].endswith("/")
                        else target
                    )
                    if source == destination:
                        continue
                    self.files.pop(destination, None)
                    self.products.pop(destination, None)
                    if source in self.files:
                        self.files[destination] = self.files[source]
                    if source in self.products:
                        self.products[destination] = self.products[source]
                    if executable == "mv":
                        self.files.pop(source, None)
                        self.products.pop(source, None)
        elif executable == "rm":
            for value in argv[1:]:
                if not value.startswith("-"):
                    self.files.pop(path(value, self.cwd), None)
                    self.products.pop(path(value, self.cwd), None)
        if executable in {"tleap", "pmemd.cuda", "ante-MMPBSA.py", "cpptraj", "MMPBSA.py"}:
            flags = options(argv)
            outputs = {}
            if executable == "tleap":
                content = self.files.get(path(flags.get("-f", ""), self.cwd), stdin)
                units = set()
                for words in leap_commands(content or ""):
                    if words[0].lower() == "quit":
                        break
                    if len(words) == 4 and words[1] == "=" and words[2].lower() == "loadpdb":
                        units.add(words[0])
                    if (
                        len(words) == 4
                        and words[0].lower() == "saveamberparm"
                        and words[1] in units
                    ):
                        outputs.update({words[2]: "tleap:topology", words[3]: "tleap:coordinates"})
                    if len(words) == 3 and words[0].lower() == "savepdb" and words[1] in units:
                        outputs[words[2]] = "tleap:pdb"
            elif executable == "cpptraj":
                content = stdin or self.files.get(path(flags.get("-i", ""), self.cwd), "")
                for line in content.splitlines():
                    words = shlex.split(line, comments=True)
                    if words[:2] == ["parmwrite", "out"] and len(words) == 3:
                        outputs[words[2]] = "cpptraj:topology"
            else:
                output_flags = {
                    "pmemd.cuda": {"-o": "log", "-r": "restart", "-x": "trajectory"},
                    "ante-MMPBSA.py": {
                        "-c": "topology",
                        "-r": "topology",
                        "-l": "topology",
                        "--complex-prmtop": "topology",
                        "--receptor-prmtop": "topology",
                        "--ligand-prmtop": "topology",
                    },
                    "MMPBSA.py": {"-o": "report"},
                }[executable]
                outputs = {
                    flags[key]: f"{executable}:{kind}"
                    for key, kind in output_flags.items()
                    if key in flags
                }
            for filename, kind in outputs.items():
                target = path(filename, self.cwd)
                self.files.pop(target, None)
                self.products[target] = kind
        return True
