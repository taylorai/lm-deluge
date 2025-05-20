from pydantic import BaseModel, ValidationError
from .json import load_json
from .xml import get_tag, xml_to_object


def get_model_from_json(
    json_string: str,
    model_class: BaseModel,
) -> BaseModel:
    try:
        model_dict = load_json(json_string)
        return model_class(**model_dict)  # pyright: ignore
    except ValidationError as ve:
        # Handle validation errors if necessary
        raise ve


def get_model_from_xml(xml_string: str, model_class: BaseModel, shallow: bool = True):
    """
    Convert an XML string to a Pydantic model.
    If shallow is True, we don't try to parse the whole XML tree
    into a Python object, we just try to extract each key's tag
    with regex and fill the model's fields in that way.
    """
    if shallow:
        # iterate over the fields of the model
        model_dict = {}
        for field_name, field_info in model_class.__fields__.items():
            val = get_tag(xml_string, field_name)
            if val is not None:
                # no nested models for 'shallow' mode
                model_dict[field_name] = val

        try:
            return model_class(**model_dict)  # pyright: ignore
        except ValidationError as ve:
            # Handle validation errors if necessary
            raise ve
    else:
        # use helper to parse the whole tree
        model_dict = xml_to_object(xml_string)
        try:
            return model_class(**model_dict)  # pyright: ignore
        except ValidationError as ve:
            # Handle validation errors if necessary
            raise ve
