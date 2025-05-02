from typing import Callable, Any, Optional, Type

from langgraph.constants import START, END  # noqa: F401
from langgraph.graph import StateGraph


def ManusAgent(
    *,
    state_schema: Optional[Type[Any]] = None,
    config_schema: Optional[Type[Any]] = None,
    input: Optional[Type[Any]] = None,
    output: Optional[Type[Any]] = None,
    impl: list[tuple[str, Callable]],
) -> StateGraph:
    """Create the state graph for ManusAgent."""
    # Declare the state graph
    builder = StateGraph(
        state_schema, config_schema=config_schema, input=input, output=output
    )

    nodes_by_name = {name: imp for name, imp in impl}

    all_names = set(nodes_by_name)

    expected_implementations = {
        "Coordinator",
        "Planner",
        "Operator",
        "to_planner",
        "to_operator",
        "should_end",
    }

    missing_nodes = expected_implementations - all_names
    if missing_nodes:
        raise ValueError(f"Missing implementations for: {missing_nodes}")

    extra_nodes = all_names - expected_implementations

    if extra_nodes:
        raise ValueError(
            f"Extra implementations for: {extra_nodes}. Please regenerate the stub."
        )

    # Add nodes
    builder.add_node("Coordinator", nodes_by_name["Coordinator"])
    builder.add_node("Planner", nodes_by_name["Planner"])
    builder.add_node("Operator", nodes_by_name["Operator"])

    # Add edges
    builder.add_edge(START, "Coordinator")
    builder.add_conditional_edges(
        "Coordinator",
        nodes_by_name["to_planner"],
        [
            END,
            "Planner",
        ],
    )
    builder.add_conditional_edges(
        "Planner",
        nodes_by_name["to_operator"],
        [
            "Operator",
            END,
        ],
    )
    builder.add_conditional_edges(
        "Operator",
        nodes_by_name["should_end"],
        [
            END,
            "Operator",
        ],
    )
    return builder
