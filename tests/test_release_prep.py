"""CPU-only checks of release transformations and immutable numeric results."""
import importlib.util
import ipaddress
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('release_prep',Path(__file__).resolve().parents[1]/'scripts/release/prepare_public.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_private_ranges_redacted_but_public_and_loopback_preserved():
    private = [str(ipaddress.ip_address(n)) for n in (0x0a010203,0xac100102,0xc0a80102,0x64400102)]
    public = str(ipaddress.ip_address(0x08080808))
    loopback = str(ipaddress.ip_address(0x7f000001))
    result, counts = R.scrub(' '.join(private+[public,loopback]))
    assert counts['private_ipv4']==4
    assert all(ip not in result for ip in private)
    assert public in result and loopback in result
    assert result.count('.invalid')==4


def test_scrub_idempotent():
    addr=str(ipaddress.ip_address(0x0a010203))
    home='/'+'home'+'/researcher/run'
    gpu='GPU-'+'12345678-1234-1234-1234-123456789abc'
    mac=':'.join(['ab','cd','ef','01','23','45'])
    once,counts=R.scrub(' '.join([addr,home,gpu,mac,'contact@example.invalid']))
    twice,again=R.scrub(once)
    assert once==twice
    assert counts and not again


def test_results_numeric_values_unchanged():
    before={'score':99.8,'n':1000,'items':[1,0,1],'metadata':{'endpoint':'http://'+str(ipaddress.ip_address(0x0a010203))+':8000/v1'}}
    import json
    out,_=R.scrub(json.dumps(before))
    after=json.loads(out)
    assert R.numeric_leaves(before)==R.numeric_leaves(after)
    assert before['metadata']!=after['metadata']


def test_named_secret_redacted():
    token='hf_'+'A'*30
    out,counts=R.scrub('credential='+token)
    assert token not in out and counts['credential']==1


def test_private_address_in_filename_is_redacted():
    addr=str(ipaddress.ip_address(0x0a010203))
    result,counts=R.scrub('logs/node-'+addr+'.txt')
    assert addr not in result and result.endswith('.invalid.txt')
    assert counts['private_ipv4']==1
