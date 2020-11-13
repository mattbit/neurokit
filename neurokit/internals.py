import importlib


def import_optional_dependency(name: str, raise_on_missing: bool = True):
    msg = (
        f"Missing optional dependency `{name}`"
        f"Run `pip install {name}` if you want to install it."
    )

    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_on_missing:
            raise ModuleNotFoundError(msg) from None
        else:
            return None

    return module
