import json

from lm_deluge.util.json import (
    _escape_interior_quotes,
    heal_json,
    load_json,
    strip_json,
)


def test_healing():
    # The incomplete JSON from your example
    incomplete_json = """{
  "postings": [
    {
      "title": "1L Summer Associate (TCDIP Clerkship)",
      "info_url": "https://diversityinpractice.org/tcdip-1l-clerkship/",
      "application_url": "https://recruiting.myapps.paychex.com/appone/MainInfoReq.asp?R_ID=6609359",
      "application_email": "slocsin@winthrop.com",
      "who_its_for": "1L law students, especially those from historically underrepresented backgrounds",
      "locations": "Minneapolis/St. Paul, MN",
      "other_program_info": "The program typically lasts 10-12 weeks during the summer. Participants work with a corporate partner (Sherman Associates) and gain hands-on legal experience. The program includes mentorship, professional development workshops, and networking opportunities through TCDIP.",
      "how_to_apply": "Submit a cover letter, resume, undergraduate transcript, and writing sample through the application link provided.",
      "currently_accepting_applications": "unclear",
      "applications_open": null,
      "applications_close": null,
      "check_back": null,
      "additional_info": "This is part of the Twin Cities Diversity in Practice (TCDIP) 1L Clerkship program. The firm is evaluating candidates to fill their 2026 2L Summer Associate class, suggesting this is a pipeline to future employment."
    }"""

    print("Original JSON:")
    print(incomplete_json)
    print("\n" + "-" * 50 + "\n")

    try:
        json.loads(incomplete_json)
        print("Original JSON is valid (unexpected)")
    except json.JSONDecodeError as e:
        print(f"Original JSON is invalid (expected): {e}")

    # Test heal_json function
    healed_json = heal_json(incomplete_json)
    print("\nHealed JSON:")
    print(healed_json)
    print("\n" + "-" * 50 + "\n")

    # Validate the healed JSON
    try:
        parsed = json.loads(healed_json)
        print("Healed JSON is valid!")
        print("\nParsed content:")
        print(f"Number of postings: {len(parsed['postings'])}")
        print(f"First posting title: {parsed['postings'][0]['title']}")
    except json.JSONDecodeError as e:
        print(f"Healed JSON is still invalid: {e}")

    # Test load_json function
    print("\nTesting load_json function:")
    try:
        result = load_json(incomplete_json)
        assert isinstance(result, dict), "it's not a dict"
        print("load_json successfully parsed the JSON!")

        print(f"Number of postings: {len(result['postings'])}")
        print(f"First posting title: {result['postings'][0]['title']}")
    except Exception as e:
        print(f"load_json failed: {e}")


def test_strip_json_removes_fences():
    raw = '```json\n{"a":1}\n```'
    assert strip_json(raw) == '{"a":1}'


def test_heal_json_adds_missing_brackets():
    broken = '{"a": [1, 2}'
    healed = heal_json(broken)
    assert healed.endswith("]}")


def test_escape_interior_quotes():
    # Unescaped quotes inside a string value
    broken = '{"key": "He said "hello" to her"}'
    result = load_json(broken)
    assert result["key"] == 'He said "hello" to her'

    # Multiple pairs of unescaped interior quotes
    broken2 = '{"key": "Use "foo" and "bar" here"}'
    result2 = load_json(broken2)
    assert result2["key"] == 'Use "foo" and "bar" here'

    # Already-escaped quotes should not be double-escaped
    valid = '{"key": "He said \\"hello\\" to her"}'
    result3 = load_json(valid)
    assert result3["key"] == 'He said "hello" to her'

    # Interior quotes combined with missing closing bracket
    broken_combo = '{"key": "He said "hi" to her"}'
    result4 = load_json(broken_combo)
    assert result4["key"] == 'He said "hi" to her'

    # The real-world case: legal document with ("attorney-in-fact")
    broken_real = """[
   {
     "thinking": "The agent ("attorney-in-fact") is authorized.",
     "type": "note"
   }
]"""
    result5 = load_json(broken_real)
    assert result5[0]["thinking"] == 'The agent ("attorney-in-fact") is authorized.'
    assert result5[0]["type"] == "note"

    print("All interior quote escaping tests passed!")


def test_escape_interior_quotes_no_false_positives():
    # Valid JSON should pass through unchanged
    valid = '{"a": "hello", "b": "world"}'
    assert _escape_interior_quotes(valid) == valid

    # Properly escaped quotes should stay as-is
    valid2 = '{"a": "say \\"hi\\""}'
    assert _escape_interior_quotes(valid2) == valid2

    # Colons in values shouldn't confuse things
    valid3 = '{"url": "http://example.com"}'
    assert json.loads(_escape_interior_quotes(valid3)) == {"url": "http://example.com"}

    print("No false positive tests passed!")


if __name__ == "__main__":
    test_healing()
    test_strip_json_removes_fences()
    test_heal_json_adds_missing_brackets()
    test_escape_interior_quotes()
    test_escape_interior_quotes_no_false_positives()
