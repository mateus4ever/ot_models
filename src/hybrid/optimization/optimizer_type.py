# optimization_types.py
# Python equivalent of Java enum:
#
# public enum OptimizationType {
#     SIMPLE_RANDOM("simple_random"),
#     CACHED_RANDOM("cached_random"),
#     BAYESIAN("bayesian");
#
#     private final String value;
#     OptimizationType(String value) { this.value = value; }
#     public String getValue() { return value; }
# }

from enum import Enum


class OptimizerType(Enum):
    """
    Python equivalent of Java enum with string values
    Usage: OptimizationType.SIMPLE_RANDOM.value  # returns "simple_random"
    """
    SIMPLE_RANDOM = "simple_random"
    CACHED_RANDOM = "cached_random"
    BAYESIAN = "bayesian"