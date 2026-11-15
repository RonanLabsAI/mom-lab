"""Tests for scripts/release/check_public_safe.py (master plan Review Focus 5)."""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "release"))
import check_public_safe as C  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "release", "check_public_safe.py")


def run(*paths):
    return subprocess.run([sys.executable, "-I", SCRIPT, *map(str, paths)], capture_output=True, text=True)


def test_refuses_flycurrent_path(tmp_path):
    d = tmp_path / "FlyCurrent" / "app"
    d.mkdir(parents=True)
    f = d / "items.jsonl"
    f.write_text('{"q": "ok"}\n')
    r = run(f)
    assert r.returncode == 1
    assert "proprietary path" in r.stderr


def test_refuses_retenzai_reference_in_content(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("source: /example/RetenzAI/product/questions.json\n")
    r = run(f)
    assert r.returncode == 1
    assert "proprietary reference" in r.stderr


def test_refuses_ssn(tmp_path):
    f = tmp_path / "case.txt"
    f.write_text("Patient SSN 123-45-6789 presented with chest pain.\n")
    r = run(f)
    assert r.returncode == 1
    assert "ssn" in r.stderr


def test_refuses_mrn_and_dob(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("MRN: 00123456\n")
    b = tmp_path / "b.txt"
    b.write_text("DOB: 1961-04-02\n")
    assert run(a).returncode == 1
    assert run(b).returncode == 1


def test_refuses_owner_email(tmp_path):
    addr = "private-user" + "@" + "example.org"
    f = tmp_path / "x.md"
    f.write_text(f"contact {addr}\n")
    r = run(f)
    assert r.returncode == 1
    assert "owner email" in r.stderr
    assert addr not in r.stderr  # never echo the matched value


def test_passes_metar_item(tmp_path):
    item = {"id": "metar-test-0000",
            "messages": [{"role": "user", "content": "METAR: K1F0 050355Z AUTO 00000KT 10SM CLR 14/06 A2997"}],
            "reference": {"station": "K1F0", "day": 5, "time_utc": "03:55"}}
    f = tmp_path / "test.jsonl"
    f.write_text(json.dumps(item) + "\n")
    r = run(f)
    assert r.returncode == 0, r.stderr


def test_phone_pattern_skips_metar_numbers():
    # altimeter/time groups and ids must not look like phone numbers
    assert not C.scan_text("A2997 050355Z 10SM 14/06 T01390061 metar-test-0000")
    assert ("phone", ) == tuple(sorted({k for k, _ in C.scan_text("call (925) 555-0142 today")}))


def test_directory_is_walked(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "bad.txt").write_text("SSN 987-65-4321\n")
    (tmp_path / "good.txt").write_text("fine\n")
    r = run(tmp_path)
    assert r.returncode == 1
    assert "bad.txt" in r.stderr and "good.txt" not in r.stderr


def test_allowlist_exempts_reviewed_rule_only(tmp_path):
    f = tmp_path / "bird.jsonl"
    f.write_text("phone 809-555-3360\nSSN 123-45-6789\n")
    allow = tmp_path / "allow.txt"
    allow.write_text(f"{f} phone  # reviewed\n")
    r = subprocess.run([sys.executable, "-I", SCRIPT, "--allow", str(allow), str(f)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "ssn" in r.stderr and "phone" not in r.stderr


def test_allowlist_cannot_exempt_proprietary_path(tmp_path):
    d = tmp_path / "RetenzAI"
    d.mkdir()
    f = d / "x.txt"
    f.write_text("ok\n")
    allow = tmp_path / "allow.txt"
    allow.write_text(f"{f} proprietary_path\n")
    r = subprocess.run([sys.executable, "-I", SCRIPT, "--allow", str(allow), str(f)], capture_output=True, text=True)
    assert r.returncode != 0
