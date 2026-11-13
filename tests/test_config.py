from filseg.config import load_cfg


def test_cfg_overrides():
    c = load_cfg("B", 2)
    assert c.work_res == 1024 and c.crop == 512 and c.fold == 2
    assert load_cfg("A2", 0).seed == 1
