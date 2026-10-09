"""Register the modular Kedro pipelines used by this project."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, node


def _namespace_pipeline(module_name: str, module_pipeline: Pipeline) -> Pipeline:
    """Add a visual node namespace without renaming shared datasets."""
    return Pipeline(
        [
            node(
                func=module_node.func,
                inputs=module_node._inputs,
                outputs=module_node._outputs,
                name=module_node.name,
                tags=module_node.tags,
                confirms=module_node._confirms,
                namespace=module_name,
            )
            for module_node in module_pipeline.nodes
        ]
    )


def register_pipelines() -> dict[str, Pipeline]:
    """Discover each stage and compose it into Kedro's default pipeline."""
    discovered_pipelines = find_pipelines()
    default_pipeline = Pipeline([])
    for module_name, module_pipeline in discovered_pipelines.items():
        default_pipeline += _namespace_pipeline(module_name, module_pipeline)
    return {"__default__": default_pipeline}

