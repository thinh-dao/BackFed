"""
Utility functions for defense categorization.
"""

import inspect
from typing import Dict, List, Set, Type, Optional

from fl_bdbench.servers.defense_categories import (
    ClientSideDefenseServer,
    RobustAggregationServer,
    AnomalyDetectionServer,
    PostAggregationServer
)
from fl_bdbench.servers.base_server import BaseServer

def get_defense_category(server_class: Type[BaseServer]) -> List[str]:
    """
    Get the defense category for a server class.

    Args:
        server_class: The server class to check

    Returns:
        List of defense categories the server belongs to
    """
    categories = []
    if issubclass(server_class, ClientSideDefenseServer):
        categories.append("client_side")
    if issubclass(server_class, RobustAggregationServer):
        categories.append("robust_aggregation")
    if issubclass(server_class, AnomalyDetectionServer):
        categories.append("anomaly_detection")
    if issubclass(server_class, PostAggregationServer):
        categories.append("post_aggregation")

    return categories

def get_all_defenses() -> Dict[str, List[str]]:
    """
    Get all defenses categorized by their defense category.

    Returns:
        Dictionary mapping defense categories to lists of defense names
    """
    from fl_bdbench.servers import __all__ as all_servers
    import fl_bdbench.servers as servers_module

    result = {
        "client_side": [],
        "robust_aggregation": [],
        "anomaly_detection": [],
        "post_aggregation": [],
        "hybrid": []
    }

    for server_name in all_servers:
        if not server_name.endswith('Server'):
            continue

        try:
            server_class = getattr(servers_module, server_name)
            if not inspect.isclass(server_class) or not issubclass(server_class, BaseServer):
                continue

            categories = get_defense_category(server_class)

            if len(categories) > 1:
                result["hybrid"].append(server_name)

            for category in categories:
                if category in result:
                    result[category].append(server_name)
        except (AttributeError, TypeError):
            continue

    return result

def get_defenses_by_category(category: str) -> List[str]:
    """
    Get all defenses in a specific category.

    Args:
        category: The defense category to filter by

    Returns:
        List of defense names in the specified category
    """
    all_defenses = get_all_defenses()
    return all_defenses.get(category, [])

def is_defense_in_category(server_class: Type[BaseServer], category: str) -> bool:
    """
    Check if a defense belongs to a specific category.

    Args:
        server_class: The server class to check
        category: The category to check against

    Returns:
        True if the defense belongs to the category, False otherwise
    """
    return category in get_defense_category(server_class)
