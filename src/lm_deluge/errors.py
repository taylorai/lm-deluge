def raise_if_modal_exception(e: Exception):
    try:
        import modal  # type: ignore

        if isinstance(e, modal.exception.InputCancellation):
            raise e
    except ImportError:
        pass
