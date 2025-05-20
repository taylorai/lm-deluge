import json
from src.llm_utils.util.json import heal_json, load_json


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


if __name__ == "__main__":
    test_healing()
