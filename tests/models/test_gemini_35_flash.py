from lm_deluge.models import APIModel

NEW_GEMINI_FLASH_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
)


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


def test_new_gemini_flash_native_registry():
    for model_id in NEW_GEMINI_FLASH_MODELS:
        model = APIModel.from_registry(model_id)

        assert model.id == model_id
        assert model.name == model_id
        assert model.api_base == "https://generativelanguage.googleapis.com/v1beta"
        assert model.api_spec == "gemini"
        assert model.reasoning_model
        assert model.supports_json
        assert model.supports_images
        assert model.input_cost == 0.75
        assert model.cached_input_cost == 0.075
        assert model.output_cost == 3.75


if __name__ == "__main__":
    test_gemini_35_flash_native_registry()
    test_new_gemini_flash_native_registry()
    print("All tests passed!")
