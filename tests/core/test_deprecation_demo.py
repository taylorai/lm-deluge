"""Demonstration of deprecation warnings for add_* methods."""

import warnings
from lm_deluge.prompt import Message

# Show all warnings
warnings.simplefilter("always")

print("=" * 60)
print("Demonstrating deprecation warnings")
print("=" * 60)

print("\n1. Using deprecated add_text method (first call):")
msg1 = Message.user()
msg1.add_text("hello")  # Should show warning
print("   Message created successfully\n")

print("2. Using deprecated add_text method (second call):")
msg1.add_text("world")  # Should NOT show warning (only shown once)
print("   Message created successfully\n")

print("3. Using new with_text method:")
msg2 = Message.user()
msg2.with_text("hello")  # Should NOT show warning
print("   Message created successfully\n")

print("4. Using deprecated add_image method:")
test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
msg3 = Message.user()
msg3.add_image(test_image)  # Should show warning
print("   Message created successfully\n")

print("5. Using new with_image method:")
msg4 = Message.user()
msg4.with_image(test_image)  # Should NOT show warning
print("   Message created successfully\n")

print("6. Using deprecated add_tool_call method:")
msg5 = Message("assistant", [])
msg5.add_tool_call("1", "test", {"arg": "value"})  # Should show warning
print("   Message created successfully\n")

print("7. Chaining with new with_* methods (recommended usage):")
msg6 = Message.user().with_text("First part").with_text("Second part")
print("   Message created successfully with chaining\n")

print("=" * 60)
print("Summary:")
print("- Deprecated add_* methods show a warning on FIRST use only")
print("- New with_* methods never show warnings")
print("- Both methods work identically, but with_* is preferred")
print("=" * 60)
