def raise_if_modal_exception(e: Exception):
    try:
        import modal
        if isinstance(e, modal.exception.InputCancellation):
            raise e
    except ImportError:
        pass
