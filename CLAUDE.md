- if there's a .venv (there usually is) always use the python in there (.venv/bin/python) so you get all the installed dependencies needed for it to work, unless you're using python just to do bash-like things that don't require deps.
- whenever you would try to run a test with python -c "[something]" consider instead adding a test to the tests folder so that we can always have that test and catch any regressions. if it's a test we would want to continue running in the future, put it in tests/core. if it's very niche and testing a one-off thing, put it in tests/one_off.
- don't use == True and == False as these always lead to ruff errors
- we currently run tests in this repo by just doing python tests/path_to_test.py, not pytest
- computer use info:

  1. OpenAI computer calls are stored as ToolCall objects with
  names like _computer_click, _computer_type, etc.
  2. The _computer_ prefix is reserved - user-provided
  tools cannot use this prefix, enforced by validation in
   the Tool class
  3. Clean separation - Regular tools and computer
  actions can coexist:
    - User tools: get_weather, search_web, even
  computer_status (no underscore prefix)
    - Computer actions: _computer_click,
  _computer_screenshot, etc.
  4. Provider-specific formatting happens during
  conversion:
    - to_openai_responses() extracts _computer_*
  ToolCalls from assistant messages and formats them as
  separate computer_call items
    - Computer call outputs are stored as ToolResult with
   the _computer_use_output marker

  This approach keeps the core data model clean and
  provider-neutral while ensuring computer actions won't
  conflict with user-defined tools.
