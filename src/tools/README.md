# Experimental Framework Tools

Tools are defined per operational scenario and provide the action space
that agents can use to interact with the satellite environment.

## Guidelines

1. Each tool must subclass a common `BaseTool` interface (TBD per scenario)
2. Tools must be stateless — all state lives in the environment and memory
3. Tool definitions should be serializable to YAML for configuration
4. Tools must provide metadata (name, description, parameters) for LLM-based agents
5. Tools should have deterministic behavior given the same inputs

## Existing Tools (Demo)

The top-level `tools/` directory contains tools from the demo application
(satellite data, Orekit propagation, managed satellites, etc.). These may
be adapted or wrapped for use in the experimental framework.
