from src.voice import parse_command, resolve_target


def test_find_my():
    i = parse_command("Where is my bottle?")
    assert i.kind == "find" and i.target == "bottle" and i.my


def test_find_the():
    i = parse_command("find the remote")
    assert i.kind == "find" and i.target == "remote" and not i.my


def test_identify_and_scene():
    assert parse_command("what is this").kind == "identify"
    assert parse_command("What's this?").kind == "identify"
    assert parse_command("what's around me").kind == "scene"
    assert parse_command("describe the room").kind == "scene"


def test_enroll_variants():
    a = parse_command("save this as my house keys")
    assert a.kind == "enroll" and a.target == "house keys"
    b = parse_command("this is my wallet")
    assert b.kind == "enroll" and b.target == "wallet"
    c = parse_command("remember my spectacles")
    assert c.kind == "enroll" and c.target == "spectacles"


def test_forget_list_help():
    assert parse_command("forget my keys").kind == "forget"
    assert parse_command("what do you know").kind == "list"
    assert parse_command("help").kind == "help"
    assert parse_command("blah blah").kind == "unknown"
    assert parse_command("").kind == "unknown"


def test_resolve_personal_beats_coco():
    t = resolve_target("my bottle", ["bottle", "keys"])
    assert t.kind == "personal" and t.value == "bottle"


def test_resolve_coco_synonym():
    assert resolve_target("mobile", []).value == "cell phone"
    assert resolve_target("the tv remote", []).value == "remote"
    assert resolve_target("coffee mug", []).value == "cup"


def test_resolve_fuzzy_personal():
    t = resolve_target("kees", ["keys", "wallet"])
    assert t.kind == "personal" and t.value == "keys"


def test_resolve_unknown():
    assert resolve_target("flux capacitor", ["keys"]).kind == "unknown"
