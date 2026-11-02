"""Answer equivalence checks for task 3, including an answer that makes sympy run away.

PYTHONPATH=. python tests/test_task3_answers.py
"""
import time

from task3.answers import MathScorer, VistaScorer, group_answers, grade_groups, parse_number, timeouts


def main():
    m = MathScorer()
    same = [(r"\frac{1}{2}", "0.5"), (r"\$40", "40"), ("x^2-5", "x^2 - 5"), ("(10,14,61)", "(10, 14, 61)"), ("2.00", "2"),
            (r"\text{even}", "even"), (r"x\in[2,3]", "[2,3]"), (r"\frac{3}{4}", r"\dfrac34"), ("1,234", "1234"),
            (r"\sqrt{2}+\sqrt{6}", r"\sqrt{6}+\sqrt{2}"), (r"\text{4:30 p.m.}", r"4:30 \text{ p.m.}")]
    for a, b in same:
        assert m.equivalent(a, b), (a, b)
    for a, b in [("3", "4"), ("x+1", "x+2"), (r"\text{even}", r"\text{odd}")]:
        assert not m.equivalent(a, b), (a, b)

    # a power tower: has to come back within the limit, and the scorer has to keep working afterwards
    t0 = time.time()
    assert not m.equivalent("9^{9^{9^{9}}}", "5")
    print(f"run-away answer handled in {time.time() - t0:.1f}s, timeouts: {timeouts}")
    assert m.equivalent(r"\frac{2}{4}", "0.5")

    ids, reps = group_answers(["1/2", None, "0.5", " ", "3"], m, "q")
    assert ids[0] == ids[2] and ids[1] == ids[3] == -1 and ids[4] != ids[0], (ids, reps)
    assert grade_groups(reps, "0.5", m, "q") == [True, False]

    assert parse_number(r"1.5 \times 10^{-3} \, \text{N}") == 1.5e-3
    assert parse_number(r"14.63 \, \text{m/s}") == 14.63 and parse_number(r"\$5") == 5
    assert parse_number("10-15 years") is None and parse_number("1:1") is None
    v = VistaScorer({"a": {"question_type": "free_form", "answer_type": "integer", "precision": None, "choices": None},
                     "b": {"question_type": "free_form", "answer_type": "float", "precision": 2.0, "choices": None},
                     "c": {"question_type": "multi_choice", "answer_type": "text", "precision": None, "choices": ["6cm", "7cm"]}})
    assert v.prepare("2.7", "a") == "2" and v.prepare("0.2095", "b") == "0.21"
    assert v.prepare("(B)", "c") == "B" and v.prepare("6 cm", "c") == "A" and v.prepare("Z", "c").startswith("unparsed:")
    print("ok")


if __name__ == "__main__":
    main()
