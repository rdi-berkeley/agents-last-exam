"""Conservative static proof of cross-stage Amber topology wiring.

No candidate code is executed. Unsupported dynamic shell constructs cannot
provide evidence for a newly accepted split.
"""
from __future__ import annotations

import posixpath
import re
import shlex


def _expand(text, variables):
    pattern = r"'[^']*'|\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)"
    return re.sub(pattern, lambda m: variables.get(m[1] or m[2], m[0]) if m[1] or m[2] else m[0], text)


def _statements(script):
    lines = iter(script.replace('\\\n', '').splitlines())
    for line in lines:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        body = None
        marker = re.search(r"<<(-?)\s*(['\"]?)(\w+)\2\s*$", line)
        if marker:
            chunks = []
            for following in lines:
                if following.lstrip('\t') == marker[3]:
                    break
                chunks.append(following.lstrip('\t') if marker[1] else following)
            else:
                raise ValueError('unterminated heredoc')
            body = ('\n'.join(chunks), not bool(marker[2]))
            line = line[:marker.start()]
        while True:
            try:
                shlex.split(line, comments=True)
                break
            except ValueError:
                try:
                    following = next(lines)
                except StopIteration as exc:
                    raise ValueError('unterminated shell quote') from exc
                line += '\n' + following
        yield line.strip(), body


def _path(value, cwd):
    if '$' in value or '`' in value:
        raise ValueError('unresolved path')
    return posixpath.normpath(value if value.startswith('/') else cwd + '/' + value)


def _option(args, flag):
    if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
        raise ValueError('missing or duplicate argument ' + flag)
    return args[args.index(flag) + 1]


def _scan(script, task_dir, input_dir, output_dir):
    variables = {'SLURM_SUBMIT_DIR': task_dir, 'PWD': task_dir, 'AMBERHOME': '/AMBER'}
    cwd = task_dir
    guards, commands, heredocs = [], [], {}
    for raw, body in _statements(script):
        if re.search(r'^\w+\s*\(\)|^function\b|^(?:while|until|case|eval|source)\b', raw):
            raise ValueError('unsupported shell control flow')
        if raw.startswith('if '):
            guards.append([_expand(raw[3:], variables), False])
            continue
        if raw == 'else':
            guards[-1][1] = True
            continue
        if raw.startswith('elif '):
            raise ValueError('unsupported elif')
        if raw == 'fi':
            guards.pop()
            continue
        if raw.startswith('for '):
            guards.append(['loop', False])
            continue
        if raw == 'done':
            guards.pop()
            continue
        assignment = re.fullmatch(r'(?:export\s+)?([A-Za-z_]\w*)=(.*)', raw, re.DOTALL)
        if assignment:
            name, value = assignment.groups()
            if guards:
                # The provided trajectory exists; a missing-input fallback is inactive.
                inactive = True
                for condition, alternate in guards:
                    match = re.fullmatch(r'\[\[\s*!\s+-[fs]\s+(.*?)\s*\]\];\s*then', condition)
                    if alternate or not match:
                        inactive = False
                        break
                    arguments = shlex.split(match[1])
                    if len(arguments) != 1 or _path(arguments[0], cwd) != input_dir + '/prod.mdcrd':
                        inactive = False
                        break
                if inactive:
                    continue
                raise ValueError('conditional variable assignment')
            if '$(dirname "${BASH_SOURCE[0]}")' in value:
                if value != '"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"':
                    raise ValueError('unsupported script directory expression')
                variables[name] = output_dir
                continue
            value = value.replace('$(pwd)', cwd)
            value = re.sub(r'\$\(cd "\$\{(\w+)\}/\.\." && pwd\)',
                           lambda m: posixpath.dirname(variables.get(m[1], '${unknown}')), value)
            for _ in range(3):
                value = _expand(value, variables)
                value = re.sub(r'\$\{([A-Za-z_]\w*):-([^{}]*)\}',
                               lambda m: variables.get(m[1], m[2]), value)
            words = shlex.split(value, comments=True)
            if len(words) != 1 or '$' in words[0] or '`' in words[0]:
                raise ValueError('unsupported variable value')
            variables[name] = words[0]
            continue
        expanded = _expand(raw, variables)
        if re.fullmatch(r'module\s+load\s+(?:amber/22|cuda/11\.6\.2)\s+2>/dev/null\s*\|\|\s*true', expanded):
            continue
        if '$(' in expanded.replace('$(date)', '') or '`' in expanded:
            raise ValueError('unsupported command substitution')
        logfile = re.search(r'\s+>\s*([\"\']?)([^\s\"\']+\.log)\1\s+2>&1\s*$', expanded)
        if logfile:
            expanded = expanded[:logfile.start()]
        words = shlex.split(expanded, comments=True)
        if not words:
            continue
        if (len(words) == 7 and words[:3] == ['grep', '-E', '^ATOM']
                and _path(words[3], cwd) == input_dir + '/complex_structure.pdb'
                and words[4:6] == ['|', 'awk']
                and not re.search(r'\b(?:system|getline|close|exit)\b|[>|]', words[6])):
            continue
        lexer = shlex.shlex(re.sub(r'>\s*&2\s*$', '', expanded), posix=True, punctuation_chars=';&|')
        lexer.whitespace_split = True
        if any(token and set(token) <= set(';&|') for token in lexer):
            raise ValueError('unsupported compound command')
        executable = posixpath.basename(words[0])
        if executable == 'set' and words[1:] not in (['-euo', 'pipefail'], ['-e'], ['-eu'], ['-u']):
            raise ValueError('unsupported shell execution options')
        if executable in {'echo', 'printf', 'module', 'set', 'mkdir', 'ls', 'grep'}:
            lexer = shlex.shlex(expanded, posix=True, punctuation_chars='<>')
            lexer.whitespace_split = True
            tokens = list(lexer)
            for index, token in enumerate(tokens):
                if token in {'>', '>>'} and (index + 1 == len(tokens) or tokens[index + 1] not in {'&2', '/dev/null'}):
                    raise ValueError('unverified output redirection')
            continue
        if executable in {'exit', 'return'}:
            if not guards:
                raise ValueError('unconditional termination')
            for condition, alternate in guards:
                if alternate:
                    raise ValueError('unsupported conditional termination')
                if condition == 'loop':
                    continue
                if not (re.fullmatch(r'\[\[\s*!\s+-[fs]\s+.+\]\];\s*then', condition)
                        or re.fullmatch(r'\[\[\s*-z\s+"\$\{AMBERHOME:-\}"\s*\]\];\s*then', condition)):
                    raise ValueError('unsupported conditional termination')
            continue
        if executable == 'cd':
            if guards:
                raise ValueError('conditional working directory')
            cwd = _path(words[1], cwd)
            continue
        if executable == 'cat' and body is not None:
            if words.count('>') != 1:
                raise ValueError('unsupported heredoc target')
            destination = _path(_option(words, '>'), cwd)
            text = _expand(body[0], variables) if body[1] else body[0]
            if destination in heredocs:
                raise ValueError('overwritten heredoc')
            heredocs[destination] = (text, [g[:] for g in guards])
            continue
        if executable == 'run_ambertools.sh':
            if not words[0].endswith('/software/run_ambertools.sh'):
                raise ValueError('unknown wrapper')
            words = words[1:]
            executable = words[0]
        if executable in {'pmemd.cuda', 'sander'}:
            for flag in ['-o', '-r', '-x', '-inf']:
                if flag in words and _option(words, flag).endswith(('.prmtop', '.pdb', '.leap', '.in')):
                    raise ValueError('simulation overwrites an input')
        if executable == 'rm':
            if not guards or any(not value.endswith('.prmtop') for value in words[1:] if not value.startswith('-')):
                raise ValueError('unsupported removal')
        elif executable == 'awk' and '>' not in words:
            if re.search(r'system\s*\(|getline|(?:print|printf)[^\n]*>', words[1]):
                raise ValueError('unverified AWK side effects')
            continue  # Inspection prints do not supply topology evidence.
        elif executable not in {'tleap', 'ante-MMPBSA.py', 'MMPBSA.py', 'awk', 'pmemd.cuda', 'sander'}:
            raise ValueError('unsupported command: ' + executable)
        if '$' in ' '.join(words) and executable != 'awk':
            raise ValueError('unresolved command argument')
        commands.append((executable, words[1:], cwd, [g[:] for g in guards]))
    if guards:
        raise ValueError('unclosed shell control flow')
    return commands, heredocs


def _build_guard(guards, outputs, cwd):
    for condition, alternate in guards:
        match = re.fullmatch(r'\[\[\s*(.*?)\s*\]\];\s*then', condition)
        if not match:
            return False
        expressions = re.split(r'\s+(\|\||&&)\s+', match[1])
        expected_operator = '&&' if alternate else '||'
        if any(op != expected_operator for op in expressions[1::2]):
            return False
        checked = []
        for expression in expressions[::2]:
            words = shlex.split(expression)
            if not alternate and words[:1] == ['!']:
                words = words[1:]
            elif not alternate:
                return False
            if len(words) != 2 or words[0] not in {'-f', '-s'}:
                return False
            checked.append(_path(words[1], cwd))
        if not checked or not set(checked).intersection(outputs):
            return False
    return True


def _chains(pdb_text):
    residues, seen = [], set()
    for line in pdb_text.splitlines():
        if not line.startswith('ATOM') or len(line) < 27:
            continue
        identity = (line[21], line[22:27])
        if identity not in seen:
            seen.add(identity)
            residues.append(line[21])
    if set(residues) != {'A', 'B', 'C'}:
        raise ValueError('input must contain chains A, B and C')
    return residues


def _mask_indices(mask, chains):
    if mask in {':%A', ':%B,C', ':%B|:%C'}:
        wanted = {'A'} if mask == ':%A' else {'B', 'C'}
        return {i + 1 for i, chain in enumerate(chains) if chain in wanted}
    if not re.fullmatch(r':\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', mask):
        raise ValueError('unsupported residue mask')
    selected = set()
    for part in mask[1:].split(','):
        bounds = [int(value) for value in part.split('-')]
        low, high = bounds[0], bounds[-1]
        if low < 1 or high < low or high > len(chains):
            raise ValueError('invalid residue range')
        selected.update(range(low, high + 1))
    return selected


def _awk_chains(program):
    compact = re.sub(r'\s+', '', program)
    single = re.fullmatch(r'/\^ATOM/&&substr\(\$0,22,1\)=="([A-Z])"\{print\}END\{print"TER";print"END"\}', compact)
    if single:
        return {single[1]}
    double = re.fullmatch(
        r'/\^ATOM/&&\(substr\(\$0,22,1\)=="([A-Z])"\|\|substr\(\$0,22,1\)=="([A-Z])"\)'
        r'\{ch=substr\(\$0,22,1\)if\(prev!=""&&ch!=prev\)print"TER"printprev=ch\}'
        r'END\{print"TER";print"END"\}', compact)
    if double:
        return {double[1], double[2]}
    raise ValueError('unsupported AWK record transformation')


def _nonmutating_leap(tokens, units, cwd, task_dir):
    if tokens in [['source', 'leaprc.protein.ff14SB'], ['set', 'default', 'PBRadii', 'mbondi3'], ['quit']]:
        return True
    if len(tokens) == 2 and tokens[0].lower() == 'check' and tokens[1] in units:
        return True
    return (len(tokens) == 3 and tokens[0].lower() == 'savepdb' and tokens[1] in units
            and _path(tokens[2], cwd).startswith(task_dir + '/params/')
            and _path(tokens[2], cwd).endswith('.pdb'))


def verify_cross_stage(files, pdb_text, *, task_dir, input_dir=None, output_dir=None):
    """Prove a static first-stage split and the consuming MMGBSA invocation."""
    try:
        if not task_dir or not task_dir.startswith('/'):
            raise ValueError('absolute task directory required')
        task_dir = posixpath.normpath(task_dir)
        input_dir = _path(input_dir or 'input', task_dir)
        output_dir = _path(output_dir or 'output', task_dir)
        if not pdb_text:
            raise ValueError('input PDB unavailable')
        chains = _chains(pdb_text)
        receptor = {i + 1 for i, chain in enumerate(chains) if chain == 'A'}
        ligand = set(range(1, len(chains) + 1)) - receptor
        results = files.get('FINAL_RESULTS_MMGBSA.dat', '')
        frames = re.findall(r'Calculations performed using\s+([\d.]+)\s+complex frames', results)
        if len(frames) != 1 or float(frames[0]) != 250:
            raise ValueError('result does not contain all 250 provided trajectory frames')
        for label, expected in [('Receptor', receptor), ('Ligand', ligand)]:
            masks = re.findall(label + r' mask\s*:?\s*[\"\']?(:[^\s\"\']+)', results)
            if len(masks) != 1 or _mask_indices(masks[0], chains) != expected:
                raise ValueError('incorrect result ' + label.lower() + ' mask')
        first, first_docs = _scan(files['submit_min.sh'], task_dir, input_dir, output_dir)
        second, second_docs = _scan(files['submit_prod.sh'], task_dir, input_dir, output_dir)
        third, third_docs = _scan(files['submit_mmgbsa.sh'], task_dir, input_dir, output_dir)
        if any(path.endswith(('.prmtop', '.pdb')) for path in list(first_docs) + list(second_docs) + list(third_docs)):
            raise ValueError('unverified topology or structure overwrite')
        md_first = [c for c in first if c[0] in {'pmemd.cuda', 'sander'}]
        md_second = [c for c in second if c[0] in {'pmemd.cuda', 'sander'}]
        if len(md_first) < 2 or len(md_second) != 1 or any(c[3] for c in md_first + md_second):
            raise ValueError('unverified minimization, equilibration or production calls')
        if any(c[0] not in {'pmemd.cuda', 'sander'} for c in second):
            raise ValueError('production stage modifies topologies')
        _, prod_args, prod_at, _ = md_second[0]
        restarts = {_path(_option(c[1], '-r'), c[2]) for c in md_first}
        if _path(_option(prod_args, '-c'), prod_at) not in restarts:
            raise ValueError('production restart not produced by stage one')
        for flag, filename in [('-o', 'prod.out'), ('-r', 'prod.rst'), ('-x', 'prod.mdcrd')]:
            if posixpath.basename(_option(prod_args, flag)) != filename:
                raise ValueError('incorrect production output wiring')
        mm_calls = [c for c in third if c[0] == 'MMPBSA.py']
        if len(mm_calls) != 1 or mm_calls[0][3]:
            raise ValueError('need one unconditional MMGBSA call')
        _, args, cwd, _ = mm_calls[0]
        roles = {flag: _path(_option(args, flag), cwd) for flag in ['-cp', '-rp', '-lp', '-y', '-i', '-o']}
        if len({roles[k] for k in ['-cp', '-rp', '-lp']}) != 3:
            raise ValueError('topology roles overlap')
        for command in md_first + md_second:
            if '-p' in command[1] and _path(_option(command[1], '-p'), command[2]) != roles['-cp']:
                raise ValueError('simulation uses a different complex topology')
        if roles['-y'] != input_dir + '/prod.mdcrd':
            raise ValueError('MMGBSA must consume provided input trajectory')
        if roles['-o'] != output_dir + '/FINAL_RESULTS_MMGBSA.dat':
            raise ValueError('wrong result filename')
        mdin, mdin_guards = third_docs[roles['-i']]
        if mdin_guards:
            raise ValueError('conditional MMGBSA namelist')
        params = {key.lower(): value.strip() for key, value in re.findall(r'(\w+)\s*=\s*([^,\n/]+)', mdin)}
        if (float(params.get('igb', 'nan')) != 8 or float(params.get('saltcon', 'nan')) != .150
                or int(params.get('startframe', '1')) != 1 or int(params.get('interval', '1')) != 1
                or int(params.get('endframe', '9999999')) < 250):
            raise ValueError('incorrect MMGBSA settings or incomplete frame range')
        split_calls = [c for c in first if c[0] == 'ante-MMPBSA.py']
        if not split_calls:
            pdb_roles = {input_dir + '/complex_structure.pdb': {'A', 'B', 'C'}}
            tops = {}
            for executable, arguments, at, guards in first:
                if executable == 'rm':
                    raise ValueError('unsupported removal in PDB splitting workflow')
                if executable == 'awk':
                    if guards or len(arguments) != 4 or arguments[2] != '>':
                        raise ValueError('unsupported conditional AWK split')
                    if _path(arguments[1], at) != input_dir + '/complex_structure.pdb':
                        raise ValueError('wrong input structure')
                    target = _path(arguments[3], at)
                    if target in pdb_roles:
                        raise ValueError('overwritten structure')
                    pdb_roles[target] = _awk_chains(arguments[0])
                elif executable == 'tleap':
                    doc, doc_guards = first_docs[_path(_option(arguments, '-f'), at)]
                    units = {}
                    for line in doc.splitlines():
                        tokens = shlex.split(line, comments=True)
                        if len(tokens) == 4 and tokens[1] == '=' and tokens[2].lower() == 'loadpdb':
                            units[tokens[0]] = pdb_roles[_path(tokens[3], at)]
                        elif tokens and tokens[0].lower() == 'saveamberparm' and len(tokens) == 4:
                            top = _path(tokens[2], at)
                            if top in tops or not _build_guard(guards + doc_guards, {top, _path(tokens[3], at)}, at):
                                raise ValueError('overwritten or conditional topology')
                            tops[top] = units[tokens[1]]
                        elif tokens and not _nonmutating_leap(tokens, units, at, task_dir):
                            raise ValueError('unsupported tleap modification')
            if any(tops.get(roles[k]) != required for k, required in [('-cp', {'A', 'B', 'C'}), ('-rp', {'A'}), ('-lp', {'B', 'C'})]):
                raise ValueError('unverified PDB split topology roles')
            if any(c[0] in {'rm', 'awk', 'tleap', 'ante-MMPBSA.py'} for c in third):
                raise ValueError('stage three modifies proven topologies')
            return {'verified': True, 'roles': roles, 'receptor_residues': len(receptor), 'ligand_residues': len(ligand), 'method': 'awk_tleap'}
        if len(split_calls) != 1:
            raise ValueError('need one supported first-stage topology split')
        _, split, split_cwd, split_guards = split_calls[0]
        paths = {flag: _path(_option(split, flag), split_cwd) for flag in ['-p', '-r', '-l']}
        if any(paths[a] != roles[b] for a, b in [('-p', '-cp'), ('-r', '-rp'), ('-l', '-lp')]):
            raise ValueError('cross-stage topology mismatch')
        if not _build_guard(split_guards, {paths['-r'], paths['-l']}, split_cwd):
            raise ValueError('split hidden by unsupported condition')
        mask_flags = [flag for flag in ['-m', '-n'] if flag in split]
        if len(mask_flags) != 1 or any(flag in split for flag in ['-s', '--strip-mask', '-c']):
            raise ValueError('ambiguous split mask or extra stripping')
        flag = mask_flags[0]
        if _mask_indices(_option(split, flag), chains) != (receptor if flag == '-m' else ligand):
            raise ValueError('wrong receptor/ligand mask')
        builds = []
        for command in first:
            if command[0] != 'tleap':
                continue
            _, call, at, guards = command
            doc, doc_guards = first_docs[_path(_option(call, '-f'), at)]
            loaded = {}
            for line in doc.splitlines():
                tokens = shlex.split(line, comments=True)
                if len(tokens) == 4 and tokens[1] == '=' and tokens[2].lower() == 'loadpdb':
                    loaded[tokens[0]] = _path(tokens[3], at)
                elif tokens and tokens[0].lower() == 'saveamberparm' and len(tokens) == 4:
                    top = _path(tokens[2], at)
                    if top == roles['-cp']:
                        valid = loaded.get(tokens[1]) == input_dir + '/complex_structure.pdb'
                        valid &= _build_guard(guards + doc_guards, {top, _path(tokens[3], at)}, at)
                        builds.append(bool(valid))
                    else:
                        raise ValueError('unexpected topology output before split')
                elif tokens and not _nonmutating_leap(tokens, loaded, at, task_dir):
                    raise ValueError('unsupported tleap modification')
        if builds != [True]:
            raise ValueError('unverified complex topology provenance')
        if first.index(split_calls[0]) <= next(i for i, c in enumerate(first) if c[0] == 'tleap'):
            raise ValueError('split precedes complex build')
        if any(c[0] in {'rm', 'tleap', 'awk'} for c in first[first.index(split_calls[0]) + 1:]):
            raise ValueError('modification after topology split')
        if any(c[0] in {'rm', 'awk', 'tleap', 'ante-MMPBSA.py'} for c in third):
            raise ValueError('stage three modifies proven topologies')
        return {'verified': True, 'roles': roles, 'receptor_residues': len(receptor), 'ligand_residues': len(ligand)}
    except (ValueError, KeyError, IndexError, StopIteration, TypeError) as exc:
        return {'verified': False, 'reason': str(exc)}
