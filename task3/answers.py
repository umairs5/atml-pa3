"""Answer normalization, grouping and grading. Groups are formed without looking at the gold answer;
the same equivalence is then used to grade each group.

MATH uses math-verify. MathVista follows the benchmark's rules: option letters for multiple choice,
int(float(x)) for integer answers, round(float(x), precision) for float answers, and exact lists.
"""
import multiprocessing as mp
import re

TIMEOUT_S = 5
timeouts = {"count": 0, "answers": []}


def _worker(conn):
    """Runs math-verify in its own process: some answers (huge powers and factorials) make sympy run away,
    and only a process can be stopped from outside. math-verify's own timeouts are turned off."""
    from math_verify import parse, verify
    parsed = {}

    def get(ans):
        if ans not in parsed:
            parsed[ans] = parse(f"${ans}$", parsing_timeout=None, raise_on_error=False)
        return parsed[ans]

    while True:
        try:
            a, b = conn.recv()
        except EOFError:  # the main process has finished
            return
        if b is None:
            conn.send(bool(get(a)))
            continue
        pa, pb = get(a), get(b)
        same = bool(pa and pb) and (verify(pa, pb, timeout_seconds=None, raise_on_error=False)
                                    or verify(pb, pa, timeout_seconds=None, raise_on_error=False))
        conn.send(bool(same))


def _squash(ans):
    return re.sub(r"\s+", "", re.sub(r"\\text\{([^{}]*)\}", r"\1", ans))


class MathScorer:
    """math-verify equivalence (either direction), with whitespace-free string equality for answers that do not
    parse. An answer or a comparison that runs past the time limit counts as not equivalent and is recorded."""

    def __init__(self):
        self.pairs, self.bad = {}, set()
        self.proc = self.conn = None

    def _start(self):
        self.conn, child = mp.Pipe()
        self.proc = mp.Process(target=_worker, args=(child,), daemon=True)
        self.proc.start()

    def _ask(self, a, b=None):
        if self.proc is None:
            self._start()
            self.warm = False
        self.conn.send((a, b))
        # the first call in a new worker also has to import sympy and the parser
        if self.conn.poll(TIMEOUT_S if self.warm else TIMEOUT_S + 60):
            self.warm = True
            return self.conn.recv()
        self.proc.kill()
        self.proc.join()
        self.proc = None
        timeouts["count"] += 1
        timeouts["answers"].append([a, b])
        return None

    def equivalent(self, a, b):
        if _squash(a) == _squash(b):
            return True
        key = (a, b) if a < b else (b, a)
        if key not in self.pairs:
            for ans in key:
                if ans not in self.bad and self._ask(ans) is None:
                    self.bad.add(ans)
            self.pairs[key] = False if self.bad & set(key) else bool(self._ask(a, b))
        return self.pairs[key]

    def prepare(self, ans, sample_id):
        return ans.strip()

    def prepare_gold(self, gold, sample_id):
        return gold.strip()


def _strip_units(s):
    def keep_number(m):
        return m.group(1) if re.fullmatch(r"\s*[-+\d.,]+\s*", m.group(1)) else " "
    s = re.sub(r"\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}", keep_number, s)
    for token in ["\\$", "\\,", "\\ ", "\\!", "\\;", "~", "$", "\\%", "%"]:
        s = s.replace(token, " ")
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|°", " ", s)
    return s.strip()


def parse_number(s):
    """Plain, comma-grouped or scientific-notation number with an optional trailing unit word; None otherwise."""
    s = _strip_units(s)
    m = re.fullmatch(r"([-+]?(?:\d[\d,]*\.?\d*|\.\d+))\s*(?:(?:\\times|\\cdot|x|\*)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?)?\s*([a-zA-Z][a-zA-Z/^\d\-\s]*)?", s)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return v * 10 ** int(m.group(2)) if m.group(2) else v


class VistaScorer:
    """An answer that cannot be read under its question's type becomes its own 'unparsed:' group: it still
    votes (the extraction was not missing), but it can never be graded correct."""

    def __init__(self, meta):
        self.meta = meta

    def prepare(self, ans, sample_id):
        m = self.meta[sample_id]
        ans = ans.strip()
        if m["question_type"] == "multi_choice":
            return self._letter(ans, m["choices"])
        if m["answer_type"] == "list":
            return "[" + ",".join(p.strip() for p in ans.strip("[]() ").split(",")) + "]"
        v = parse_number(ans)
        if v is None:
            return f"unparsed:{_squash(ans)}"
        return str(int(v)) if m["answer_type"] == "integer" else str(round(v, int(m["precision"])))

    @staticmethod
    def _letter(ans, choices):
        options = [chr(ord("A") + i) for i in range(len(choices))]
        hit = re.fullmatch(r"\(?([A-Za-z])\)?\.?", ans)
        if hit and hit.group(1).upper() in options:
            return hit.group(1).upper()
        for opt, choice in zip(options, choices):
            if _squash(str(choice)).lower() == _squash(ans).lower():
                return opt
        return f"unparsed:{_squash(ans)}"

    def prepare_gold(self, gold, sample_id):
        # multiple-choice gold answers are already option letters in the export
        if self.meta[sample_id]["question_type"] == "multi_choice":
            return gold.strip()
        return self.prepare(gold, sample_id)

    def equivalent(self, a, b):
        return a == b


def group_answers(extracted, scorer, sample_id):
    """Group one question's answers in rollout-id order. Returns (group id per rollout, representative per
    group); missing extractions all get id -1, the NO_ANSWER group."""
    reps, ids, seen = [], [], {}
    for ans in extracted:
        if ans is None or not str(ans).strip():
            ids.append(-1)
            continue
        prepared = scorer.prepare(str(ans), sample_id)
        if prepared not in seen:
            match = next((i for i, r in enumerate(reps) if scorer.equivalent(r, prepared)), None)
            if match is None:
                reps.append(prepared)
                match = len(reps) - 1
            seen[prepared] = match
        ids.append(seen[prepared])
    return ids, reps


def grade_groups(reps, gold, scorer, sample_id):
    target = scorer.prepare_gold(gold, sample_id)
    return [not r.startswith("unparsed:") and scorer.equivalent(r, target) for r in reps]
