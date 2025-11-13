
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef
from typing import List, Tuple, Dict, Set
import re

EX = Namespace('http://example.org/reasoner#')

# --- Parsing: map simple sentences to facts (predicate, subject, object) ---
PATTERNS = [
    (re.compile(r"^(.+?) is the parent of (.+?)\.?$", re.I), 'parentOf'),
    (re.compile(r"^(.+?) is the mother of (.+?)\.?$", re.I), 'parentOf'),
    (re.compile(r"^(.+?) is the father of (.+?)\.?$", re.I), 'parentOf'),
    (re.compile(r"^(.+?) works at (.+?)\.?$", re.I), 'worksAt'),
    (re.compile(r"^(.+?) manages (.+?)\.?$", re.I), 'manages'),
    (re.compile(r"^(.+?) is a colleague of (.+?)\.?$", re.I), 'colleagueOf'),
    (re.compile(r"^(.+?) is located in (.+?)\.?$", re.I), 'locatedIn'),
    (re.compile(r"^(.+?) is in (.+?)\.?$", re.I), 'locatedIn'),
    (re.compile(r"^(.+?) is part of (.+?)\.?$", re.I), 'locatedIn'),
    (re.compile(r"^(.+?) is (an?|the) (.+?)\.?$", re.I), 'isa'),
]

PRED_TO_URI = {
    'parentOf': EX.parentOf,
    'worksAt': EX.worksAt,
    'manages': EX.manages,
    'colleagueOf': EX.colleagueOf,
    'locatedIn': EX.locatedIn,
    'ancestorOf': EX.ancestorOf,
    'isa': RDF.type,
}

BUILTINS = { 'neq' }  # supported built-in guard

# --- utils ---

def canonicalize_entity(name: str) -> str:
    name = name.strip().rstrip('.')
    name = re.sub(r"\s+", "_", name)
    return name

# --- sentence parsing ---

def parse_sentence_to_fact(sentence: str) -> List[Tuple[str, str, str]]:
    s = sentence.strip()
    if not s:
        return []
    for rx, pred in PATTERNS:
        m = rx.match(s)
        if m:
            if pred == 'isa':
                subj = canonicalize_entity(m.group(1))
                cls = canonicalize_entity(m.group(3))
                return [(pred, subj, cls)]
            subj = canonicalize_entity(m.group(1))
            obj = canonicalize_entity(m.group(2))
            return [(pred, subj, obj)]
    return []

def parse_sentences_block(text: str) -> List[Tuple[str, str, str]]:
    facts: List[Tuple[str, str, str]] = []
    for line in text.splitlines():
        facts.extend(parse_sentence_to_fact(line))
    return facts

# --- Graph build & helpers ---

def build_graph() -> Graph:
    g = Graph()
    g.bind('ex', EX)
    for p in ['parentOf','worksAt','manages','colleagueOf','locatedIn','ancestorOf']:
        g.add((PRED_TO_URI[p], RDF.type, OWL.ObjectProperty))
        g.add((PRED_TO_URI[p], RDFS.label, Literal(p)))
    return g

def add_fact(g: Graph, pred: str, subj: str, obj: str):
    if pred == 'isa':
        g.add((EX[subj], RDF.type, EX[obj]))
    else:
        g.add((EX[subj], PRED_TO_URI[pred], EX[obj]))

def graph_facts(g: Graph) -> List[Tuple[str,str,str]]:
    rows: List[Tuple[str,str,str]] = []
    for s,_,o in g.triples((None, RDF.type, None)):
        if str(s).startswith(str(EX)) and str(o).startswith(str(EX)):
            rows.append(('isa', str(s).split('#')[-1], str(o).split('#')[-1]))
    for pred in ['parentOf','worksAt','manages','colleagueOf','locatedIn','ancestorOf']:
        for s,_,o in g.triples((None, PRED_TO_URI[pred], None)):
            rows.append((pred, str(s).split('#')[-1], str(o).split('#')[-1]))
    return rows

# --- Built-in (fixed) rule engine ---

def apply_reasoning(g: Graph) -> Dict[Tuple[str,str,str], Tuple[str, List[Tuple[str,str,str]]]]:
    derivations: Dict[Tuple[str,str,str], Tuple[str, List[Tuple[str,str,str]]]] = {}
    base = set(graph_facts(g))
    inferred = set()

    def add_inferred(triple, rule, sources):
        if triple not in base and triple not in inferred:
            inferred.add(triple)
            derivations[triple] = (rule, sources)
            p,s,o = triple
            add_fact(g, p, s, o)
            base.add(triple)
            return True
        return False

    changed = True
    while changed:
        changed = False
        facts = list(base)
        by_pred: Dict[str, List[Tuple[str,str]]] = {}
        for p,s,o in facts:
            by_pred.setdefault(p, []).append((s,o))
        # R1: parent -> ancestor
        for s,o in by_pred.get('parentOf', []):
            changed |= add_inferred(('ancestorOf', s, o), 'R1: parent -> ancestor', [('parentOf', s, o)])
        # R2: parent(X,Y) & ancestor(Y,Z) -> ancestor(X,Z)
        po = by_pred.get('parentOf', [])
        an = by_pred.get('ancestorOf', [])
        if po and an:
            an_by_y: Dict[str, Set[str]] = {}
            for y,z in an:
                an_by_y.setdefault(y, set()).add(z)
            for x,y in po:
                for z in an_by_y.get(y, set()):
                    changed |= add_inferred(('ancestorOf', x, z), 'R2: ancestor transitive', [('parentOf', x, y), ('ancestorOf', y, z)])
        # R3: manages -> colleague
        for s,o in by_pred.get('manages', []):
            changed |= add_inferred(('colleagueOf', s, o), 'R3: manager -> colleague', [('manages', s, o)])
        # R4: colleague symmetric
        for a,b in by_pred.get('colleagueOf', []):
            changed |= add_inferred(('colleagueOf', b, a), 'R4: colleague symmetric', [('colleagueOf', a, b)])
        # R5: same org -> colleagues
        wa = by_pred.get('worksAt', [])
        if wa:
            by_org: Dict[str, Set[str]] = {}
            for emp,org in wa:
                by_org.setdefault(org, set()).add(emp)
            for org, employees in by_org.items():
                emps = list(employees)
                for i in range(len(emps)):
                    for j in range(i+1, len(emps)):
                        x,y = emps[i], emps[j]
                        changed |= add_inferred(('colleagueOf', x, y), 'R5: same org -> colleagues', [('worksAt', x, org), ('worksAt', y, org)])
                        changed |= add_inferred(('colleagueOf', y, x), 'R4: colleague symmetric', [('colleagueOf', x, y)])
        # R6: locatedIn transitive
        li = by_pred.get('locatedIn', [])
        if li:
            mid_to_z: Dict[str, Set[str]] = {}
            src_to_mid: Dict[str, Set[str]] = {}
            for m,z in li:
                mid_to_z.setdefault(m, set()).add(z)
            for x,m in li:
                src_to_mid.setdefault(x, set()).add(m)
            for x, mids in src_to_mid.items():
                for m in mids:
                    for z in mid_to_z.get(m, set()):
                        if z != x:
                            changed |= add_inferred(('locatedIn', x, z), 'R6: location transitive', [('locatedIn', x, m), ('locatedIn', m, z)])
    return derivations

# --- Custom rule DSL ---

class Atom:
    def __init__(self, pred: str, args: Tuple[str,str]):
        self.pred = pred.strip()
        self.args = (args[0].strip(), args[1].strip())
    def __repr__(self):
        return f"{self.pred}({self.args[0]},{self.args[1]})"

class Rule:
    def __init__(self, name: str, antecedents: List[Atom], consequent: Atom):
        self.name = name.strip() or "rule"
        self.antecedents = antecedents
        self.consequent = consequent
    def __repr__(self):
        ants = ' AND '.join([repr(a) for a in self.antecedents])
        return f"{self.name}: IF {ants} THEN {repr(self.consequent)}"

VAR_RX = re.compile(r"^\?[a-zA-Z_][a-zA-Z0-9_]*$")
ATOM_RX = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^,]+)\s*,\s*([^\)]+)\s*\)\s*$")


def _parse_atom(txt: str) -> Atom:
    m = ATOM_RX.match(txt)
    if not m:
        raise ValueError(f"Invalid atom syntax: {txt}")
    pred = m.group(1)
    a1 = m.group(2)
    a2 = m.group(3)
    return Atom(pred, (a1, a2))


def _parse_rule_statement(stmt: str) -> Rule:
    head = ''
    body = stmt
    if ':' in stmt.split('IF',1)[0]:
        head, body = stmt.split(':', 1)
    name = head.replace('rule', '').strip() if head else ''

    if 'IF' not in body or 'THEN' not in body:
        raise ValueError(f"Missing IF/THEN in rule: {stmt}")
    before, after = body.split('THEN', 1)
    _, ants_txt = before.split('IF', 1)
    ants = [a.strip() for a in ants_txt.split('AND') if a.strip()]
    antecedents = [_parse_atom(a) for a in ants]
    cons = _parse_atom(after.strip())
    return Rule(name or cons.pred, antecedents, cons)


def parse_rules(rule_text: str) -> List[Rule]:
    rules: List[Rule] = []
    lines = [ln.strip() for ln in rule_text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    statement = ''
    for ln in lines:
        if 'THEN' in ln and 'IF' in ln:
            rules.append(_parse_rule_statement(ln))
        else:
            statement += (' ' + ln) if statement else ln
            if 'THEN' in statement and 'IF' in statement:
                rules.append(_parse_rule_statement(statement))
                statement = ''
    if statement:
        rules.append(_parse_rule_statement(statement))
    return rules

# --- Custom rules forward chaining ---

def apply_custom_rules(g: Graph, rules: List[Rule]) -> Dict[Tuple[str,str,str], Tuple[str, List[Tuple[str,str,str]]]]:
    derivations: Dict[Tuple[str,str,str], Tuple[str, List[Tuple[str,str,str]]]] = {}
    base = set(graph_facts(g))

    def index_facts(facts: Set[Tuple[str,str,str]]):
        idx: Dict[str, List[Tuple[str,str]]] = {}
        for p,s,o in facts:
            idx.setdefault(p, []).append((s,o))
        return idx

    def is_var(x: str) -> bool:
        return bool(VAR_RX.match(x.strip()))

    def unify_atom(atom: Atom, facts_idx: Dict[str, List[Tuple[str,str]]], envs: List[Dict[str,str]]):
        # Builtin guard
        if atom.pred == 'neq':
            a1, a2 = atom.args
            out = []
            for env in envs:
                v1 = env.get(a1, a1)
                v2 = env.get(a2, a2)
                if is_var(v1) or is_var(v2):
                    out.append(env)
                else:
                    if v1 != v2:
                        out.append(env)
            return out
        # Regular predicate
        cand = facts_idx.get(atom.pred, [])
        out_envs: List[Dict[str,str]] = []
        for env in envs:
            for (s,o) in cand:
                a1, a2 = atom.args
                t1 = env.get(a1, a1)
                t2 = env.get(a2, a2)
                env2 = env.copy()
                ok = True
                if is_var(t1):
                    env2[t1] = s
                else:
                    if canonicalize_entity(t1) != s:
                        ok = False
                if ok:
                    if is_var(t2):
                        env2[t2] = o
                    else:
                        if canonicalize_entity(t2) != o:
                            ok = False
                if ok:
                    out_envs.append(env2)
        return out_envs

    changed = True
    while changed:
        changed = False
        facts_idx = index_facts(base)
        for rule in rules:
            envs: List[Dict[str,str]] = [ {} ]
            for atom in rule.antecedents:
                envs = unify_atom(atom, facts_idx, envs)
                if not envs:
                    break
            if not envs:
                continue
            cpred = rule.consequent.pred
            a1, a2 = rule.consequent.args
            for env in envs:
                s = env.get(a1, a1)
                o = env.get(a2, a2)
                if is_var(s) or is_var(o):
                    continue
                s_c = canonicalize_entity(s)
                o_c = canonicalize_entity(o)
                triple = (cpred, s_c, o_c)
                if triple not in base:
                    # collect sources (skip builtins)
                    sources: List[Tuple[str,str,str]] = []
                    for at in rule.antecedents:
                        if at.pred == 'neq':
                            continue
                        s_at = env.get(at.args[0], at.args[0])
                        o_at = env.get(at.args[1], at.args[1])
                        sources.append((at.pred, canonicalize_entity(s_at), canonicalize_entity(o_at)))
                    add_fact(g, cpred, s_c, o_c)
                    base.add(triple)
                    derivations[triple] = (rule.name, sources)
                    changed = True
    return derivations

# --- Output helpers ---

def serialize_turtle(g: Graph) -> bytes:
    return g.serialize(format='turtle')


def proof_to_text(triple: Tuple[str,str,str], derivations: Dict[Tuple[str,str,str], Tuple[str, List[Tuple[str,str,str]]]]) -> str:
    seen = set()
    lines = []
    def rec(t, indent=0):
        if t in seen:
            lines.append('  '*indent + f"{t} (already shown)")
            return
        seen.add(t)
        if t not in derivations:
            lines.append('  '*indent + f"{t} (given)")
            return
        rule, sources = derivations[t]
        lines.append('  '*indent + f"{t} ⟵ {rule}")
        for s in sources:
            rec(s, indent+1)
    rec(triple)
    return ''.join(lines)
