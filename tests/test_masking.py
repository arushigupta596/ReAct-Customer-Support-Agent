from observability.masking import MAX_CHARS, cap, mask


def test_masks_contact_and_card_details():
    text = "jo@x.com, 555-123-2002, (555) 123-2002, +1 555 123 2002, 4111 1111 1111 1111"
    assert mask(text) == "[email], [phone], [phone], [phone], [card]"


def test_keeps_ids_needed_for_debugging():
    text = "item 1151293680 order #W2974929 user C1001 reservation ZFA04Y"
    assert mask(text) == text


def test_caps_long_values_and_says_so():
    out = cap("a" * 50_000)
    assert len(out) < MAX_CHARS + 100 and "chars truncated in trace" in out
    assert cap("short") == "short"
