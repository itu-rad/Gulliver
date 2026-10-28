from importlib import import_module
from types import ModuleType
from typing import List

from utils.schemas import PipelineModel, StageModel


class InstantiationError(Exception):
    """
    Exception raised for errors encountered during the instantiation of a component.

    Attributes:
        errors (List[Exception]): A list of exceptions that were raised during instantiation.

    Methods:
        __str__(): Returns a string representation of the error messages.
    """

    def __init__(self, errors: List[Exception]):
        self.errors = errors

    def __str__(self):
        return (
            "Unable to instantiate component, found the following errors: \n"
            + self.errors
        )


def get_stage_component(stage_config: StageModel, pipeline_config: PipelineModel):
    """
    Retrieves and initializes a stage component based on the provided stage configuration and pipeline configuration.

    Args:
        stage_config (StageModel): The configuration for the specific stage.
        pipeline_config (PipelineModel): The overall pipeline configuration.

    Returns:
        object: An instance of the stage component class.

    Raises:
        ValueError: If the stage class is not found.
    """
    stage_class = get_component(stage_config.component)
    if stage_class is None:
        raise ValueError("Stage class not found")
    return stage_class(stage_config, pipeline_config)


def get_component(path):
    """
    Given a dotstring path to a component, load the component and return it.

    This function takes a string with a dot-separated path to a component and
    returns the component. For example, the string "foo.bar.ClassName" would
    return the class ClassName from the module foo.bar.

    This function is used internally by the Pipeline to load the components
    specified in the pipeline configuration.

    Based on Hydra's `_locate` from Facebook Research:
    https://github.com/facebookresearch/hydra/blob/main/hydra/_internal/utils.py#L614

    Args:
        path (str): The dotstring path to the component to be loaded. Only absolute paths supported.

    Returns:
        The loaded component.

    Raises:
        ValueError: If the path is empty.
        InstantiationError: If the path is invalid or the component does not
            exist.
    """
    if path == "":
        raise ValueError("Empty path")

    parts = [part for part in path.split(".")]
    for part in parts:
        # If a relative path is passed in, the first part will be empty
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    # First module requires trying to import to validate
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except ImportError as exc_import:
        raise InstantiationError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    # Subsequent components can be checked via getattr() on first module
    # It can either be an attribute to return or a submodule to import and
    # continue searching
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        # If getattr fails, check whether it is an importable module and
        # continue down the path
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise InstantiationError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                # Any other error trying to import module can be raised as
                # InstantiationError
                except Exception as exc_import:
                    raise InstantiationError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            # If the component is not an attribute nor a module, it doesn't exist
            raise InstantiationError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj
