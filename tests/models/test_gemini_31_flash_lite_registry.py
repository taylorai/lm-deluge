from lm_deluge.models import APIModel


def test_gemini_31_flash_lite_native_registry():
    model = APIModel.from_registry("gemini-3.1-flash-lite")

    assert model.id == "gemini-3.1-flash-lite"
    assert model.name == "gemini-3.1-flash-lite"
    assert model.api_base == "https://generativelanguage.googleapis.com/v1beta"
    assert model.api_spec == "gemini"
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.input_cost == 0.25
    assert model.cached_input_cost == 0.025
    assert model.output_cost == 1.5


def test_gemini_31_flash_lite_compat_registry():
    model = APIModel.from_registry("gemini-3.1-flash-lite-compat")

    assert model.id == "gemini-3.1-flash-lite-compat"
    assert model.name == "gemini-3.1-flash-lite"
    assert model.api_base == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert model.api_spec == "openai"
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.input_cost == 0.25
    assert model.cached_input_cost == 0.025
    assert model.output_cost == 1.5


def test_gemini_31_flash_lite_preview_registry_still_available():
    model = APIModel.from_registry("gemini-3.1-flash-lite-preview")

    assert model.id == "gemini-3.1-flash-lite-preview"
    assert model.name == "gemini-3.1-flash-lite-preview"
    assert model.api_base == "https://generativelanguage.googleapis.com/v1beta"
    assert model.api_spec == "gemini"
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.input_cost == 0.25
    assert model.cached_input_cost == 0.025
    assert model.output_cost == 1.5


if __name__ == "__main__":
    test_gemini_31_flash_lite_native_registry()
    test_gemini_31_flash_lite_compat_registry()
    test_gemini_31_flash_lite_preview_registry_still_available()
    print("All tests passed!")
