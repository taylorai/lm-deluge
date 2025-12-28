import base64
import io

from PIL import Image as PILImage

from lm_deluge.prompt import Image

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/Ps4TAAAAAElFTkSuQmCC"
PNG_BYTES = base64.b64decode(PNG_B64)


def test_fingerprint_manual_caching():
    """Test that the fingerprint property uses manual caching instead of @cached_property"""
    # Create a simple test image
    pil_img = PILImage.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    # Create Image instance
    img = Image(data=buffer.getvalue(), media_type="image/png")

    # Check that _fingerprint_cache starts as None
    assert img._fingerprint_cache is None, "Cache should start as None"

    # First access should compute and cache the fingerprint
    fingerprint1 = img.fingerprint
    assert (
        img._fingerprint_cache is not None
    ), "Cache should be populated after first access"
    assert img._fingerprint_cache == fingerprint1, "Cache should match returned value"

    # Second access should use cached value (we can't directly test that computation is skipped,
    # but we can ensure the cache value is returned)
    fingerprint2 = img.fingerprint
    assert fingerprint2 == fingerprint1, "Second access should return same value"
    assert img._fingerprint_cache == fingerprint2, "Cache should still match"

    # Verify the fingerprint is a base64 string
    import base64

    try:
        base64.b64decode(fingerprint1)
        print("✓ Fingerprint is valid base64")
    except Exception:
        raise AssertionError("Fingerprint should be valid base64")

    print("✓ Manual caching for fingerprint property works correctly")


def test_mime_inferred():
    img = Image(PNG_BYTES, media_type="image/png")
    assert img._mime() == "image/png"


def test_base64_header():
    img = Image(PNG_BYTES, media_type="image/png")
    encoded = img._base64()
    assert encoded.startswith("data:image/png;base64,")


if __name__ == "__main__":
    test_mime_inferred()
    test_base64_header()
    test_fingerprint_manual_caching()
