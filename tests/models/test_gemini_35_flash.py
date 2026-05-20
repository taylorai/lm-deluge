from lm_deluge.models import APIModel


def test_gemini_35_flash_native_registry():
    model = APIModel.from_registry("gemini-3.5-flash")

    assert model.id == "gemini-3.5-flash"
    assert model.name == "gemini-3.5-flash"
    assert model.api_base == "https://generativelanguage.googleapis.com/v1beta"
    assert model.api_spec == "gemini"
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.input_cost == 1.5
    assert model.cached_input_cost == 0.15
    assert model.output_cost == 9.0


def test_gemini_35_flash_compat_registry():
    model = APIModel.from_registry("gemini-3.5-flash-compat")

    assert model.id == "gemini-3.5-flash-compat"
    assert model.name == "gemini-3.5-flash"
    assert model.api_base == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert model.api_spec == "openai"
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.input_cost == 1.5
    assert model.cached_input_cost == 0.15
    assert model.output_cost == 9.0


if __name__ == "__main__":
    test_gemini_35_flash_native_registry()
    test_gemini_35_flash_compat_registry()
    print("All tests passed!")
