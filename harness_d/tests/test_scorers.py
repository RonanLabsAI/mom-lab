import json
import time

from harness.score import code, general, math_, sql, tools, translate


def G(**kw):
    return {k: {"id": k, "output": v} for k, v in kw.items()}


def item(i, ref=None, meta=None, prompt="q"):
    return {"id": i, "messages": [{"role": "user", "content": prompt}], "reference": ref, "meta": meta or {}}


# ---------------- math ----------------
def test_gsm8k():
    items = [item("a", "72"), item("b", "72"), item("c", "72"), item("d", "1,234"), item("e", "72")]
    gens = G(a="so total is \\boxed{72}.", b="the total is 73", c="", d="Answer: 1234", )
    r = math_.score_gsm8k(items, gens, workers=1)
    assert r["correct"] == [1, 0, 0, 1, 0]  # e: missing generation
    assert r["score"] == 40.0


def test_gsm8k_gold_with_hash_and_last_number():
    r = math_.score_gsm8k([item("a", "blah #### 18")], G(a="First 5 then 18."), workers=1)
    assert r["score"] == 100.0


def test_math500():
    items = [item("a", "\\frac{1}{2}"), item("b", "\\frac{1}{2}"), item("c", "5"), item("d", "x^2+1")]
    gens = G(a="\\boxed{0.5}", b="\\boxed{\\frac{1}{3}}", c="\\boxed{\\boxed{", d="the answer \\boxed{1+x^{2}}")
    r = math_.score_math500(items, gens, workers=1)
    assert r["correct"] == [1, 0, 0, 1]


def test_math_malformed_does_not_crash():
    r = math_.score_math500([item("a", "3")], G(a="\\boxed{\\frac{1}{"), workers=1)
    assert r["score"] == 0.0


# ---------------- code ----------------
HE_TEST = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(0, 0) == 0\n"


def test_code_pass_fail_loop():
    meta = {"test": HE_TEST, "entry_point": "add"}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    loop = "```python\ndef add(a, b):\n    while True:\n        pass\n```"
    items = [item("g", meta=meta), item("b", meta=meta), item("l", meta=meta), item("m", meta=meta),
             item("x", meta=meta)]
    t = time.time()
    r = code.score_code(items, G(g=good, b=bad, l=loop, m="no code at all, sorry"), workers=4, timeout=2.0)
    assert r["correct"] == [1, 0, 0, 0, 0]
    assert time.time() - t < 30


def test_code_timeout_flag():
    ok, _, err = code.run_sandboxed("while True: pass", timeout=1.0)
    assert not ok and err == "TIMEOUT"


def test_code_memory_limit():
    ok, _, _ = code.run_sandboxed("x = bytearray(16 * 1024**3)\nprint(len(x))", timeout=10)
    assert not ok


def test_code_assert_style_and_stdin():
    meta = {"test": "assert f(3) == 9\nassert f(-2) == 4"}
    assert code.score_code([item("a", meta=meta)], G(a="def f(x):\n    return x*x"), workers=1)["score"] == 100.0
    io = {"io_tests": [{"input": "3\n4\n", "output": "7\n"}, {"input": "1\n1\n", "output": "2"}]}
    prog = "```python\na=int(input());b=int(input());print(a+b)\n```"
    assert code.score_code([item("a", meta=io)], G(a=prog), workers=1)["score"] == 100.0
    assert code.score_code([item("a", meta=io)], G(a="print(0)"), workers=1)["score"] == 0.0


def test_code_functional_style():
    meta = {"func_name": "twoSum", "io_tests": [{"input": "[1,2]\n3", "output": "3"}]}
    sol = "class Solution:\n    def twoSum(self, a, b):\n        return a[0] + a[1]"
    # add(1,2)=3 matches expected 3
    assert code.score_code([item("a", meta=meta)], G(a=sol), workers=1)["score"] == 100.0


def test_code_extract_prefers_last_block_and_unterminated():
    assert code.extract_code("x\n```python\na=1\n```\ntext\n```python\nb=2\n```").strip() == "b=2"
    assert code.extract_code("```python\nb=2").strip() == "b=2"


# ---------------- sql ----------------
def test_sql(tiny_db):
    meta = {"db_path": str(tiny_db)}
    gold = "SELECT name FROM emp WHERE dept='eng'"
    items = [item(i, gold, meta) for i in "abcdef"]
    gens = G(
        a="```sql\nSELECT name FROM emp WHERE dept = 'eng' ORDER BY id DESC;\n```",  # order-insensitive
        b="```sql\nSELECT name FROM emp\n```",  # wrong rows
        c="```sql\nSELEKT nonsense\n```",  # syntax error
        d="```sql\nDROP TABLE emp\n```",  # read-only -> error -> 0
        e="",
    )
    r = sql.score_sql(items, gens, workers=1)
    assert r["correct"] == [1, 0, 0, 0, 0, 0]
    # table must still exist after the DROP attempt
    assert sql.execute(tiny_db, "select count(*) from emp") == [(4,)]


def test_sql_set_vs_multiset(tiny_db):
    meta = {"db_path": str(tiny_db)}
    gold = "SELECT salary FROM emp WHERE dept='ops'"  # rows (70),(70)
    gens = G(a="SELECT DISTINCT salary FROM emp WHERE dept='ops'")
    assert sql.score_sql([item("a", gold, meta)], gens, workers=1)["score"] == 100.0  # official: set
    assert sql.score_sql([item("a", gold, meta)], gens, workers=1, strict_multiset=True)["score"] == 0.0


def test_sql_timeout(tiny_db):
    slow = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c")
    t = time.time()
    try:
        sql.execute(tiny_db, slow, timeout=1.0)
        assert False
    except Exception as e:
        assert "interrupt" in str(e).lower()
    assert time.time() - t < 10


# ---------------- tools ----------------
FUNC = [{"name": "get_weather", "description": "w", "parameters": {"type": "dict", "properties": {
    "city": {"type": "string", "description": "c"}, "days": {"type": "integer", "description": "d"}},
    "required": ["city"]}}]
GOLD = [{"get_weather": {"city": ["Paris", "paris"], "days": [3, ""]}}]


def test_bfcl_simple():
    meta = {"function": FUNC, "category": "simple"}
    it = lambda i: item(i, GOLD, meta)  # noqa
    gens = G(
        a='[{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]',
        b='[{"name": "get_weather", "arguments": {"city": "Paris", "days": "3"}}]',  # wrong type
        c='[{"name": "get_forecast", "arguments": {"city": "Paris"}}]',  # wrong name
        d='<tool_call>\n{"name": "get_weather", "arguments": {"city": "paris"}}\n</tool_call>',  # optional omitted
        e="get_weather(city='Paris', days=3)",
        f='[{"name": "get_weather", "arguments": {',  # malformed
        g='{"name": "get_weather", "arguments": {"days": 3}}',  # missing required
    )
    r = tools.score_bfcl([it(i) for i in "abcdefg"], gens)
    assert r["correct"] == [1, 0, 0, 1, 1, 0, 0]


def test_bfcl_parallel():
    meta = {"function": FUNC, "category": "parallel"}
    gold = [{"get_weather": {"city": ["Paris"], "days": [""]}}, {"get_weather": {"city": ["Rome"], "days": [""]}}]
    out = '[{"name":"get_weather","arguments":{"city":"Rome"}},{"name":"get_weather","arguments":{"city":"Paris"}}]'
    assert tools.score_bfcl([item("a", gold, meta)], G(a=out))["score"] == 100.0
    out1 = '[{"name":"get_weather","arguments":{"city":"Rome"}}]'
    assert tools.score_bfcl([item("a", gold, meta)], G(a=out1))["score"] == 0.0


# ---------------- translate ----------------
def test_translate():
    items = [item("a", "Hola, ¿cómo estás hoy?", {"src_lang": "eng_Latn", "tgt_lang": "spa_Latn"}),
             item("b", "Das Wetter ist heute sehr schön.", {"src_lang": "eng_Latn", "tgt_lang": "deu_Latn"})]
    r = translate.score_flores(items, G(a="Hola, ¿cómo estás hoy?", b="Das Wetter ist heute sehr schön."))
    assert r["score"] == 100.0 and all(v == 100.0 for v in r["correct"])
    r = translate.score_flores(items, G(a="", b="qqq zzz"))
    assert r["score"] == 0.0
    r = translate.score_flores(items, {})
    assert r["score"] == 0.0


# ---------------- general ----------------
def test_mmlu_pro():
    items = [item(i, "C") for i in "abcde"]
    gens = G(a="Let me think... Answer: C", b="The answer is (C).", c="Answer: B", d="", e="I like turtles")
    r = general.score_mmlu_pro(items, gens)
    assert r["correct"] == [1, 1, 0, 0, 0]


def test_ifeval():
    meta = {"instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
            "kwargs": [{"keywords": ["apple"]}, {}]}
    it = lambda i: item(i, None, meta, prompt="Write about apple without commas")  # noqa
    gens = G(a="I love an apple pie.", b="I love apple, pie.", c="no fruit here", d="")
    r = general.score_ifeval([it(i) for i in "abcde"], gens)
    assert r["correct"] == [1, 0, 0, 0, 0]


def test_ifeval_bad_meta_no_crash():
    r = general.score_ifeval([item("a", None, {})], G(a="x"))
    assert r["score"] == 0.0
