# Research-Paper-Resources

Supporting materials for reproducing the research in the paper "Interrogators - Attack Surface Mapping in an Agentic World"

## Repository Structure

This repository contains test targets and harnesses used in the research for automated agent security testing using Agent Interrogator.

### Test Targets (`/test-targets/`)

The repository includes three variants of the Damn Vulnerable LLM Agent (DVLA) used for testing:

1. **damn-vulnerable-llm-agent** - Standard DVLA implementation with full tool functionality (port 8501)
2. **damn-vulnerable-llm-agent-mcp** - DVLA modified with Model Context Protocol (MCP) support (port 8502)  
3. **damn-vulnerable-llm-agent-neutralized** - DVLA with all tools removed for baseline testing (port 8503)

Each test target is containerized and can be run independently for comparative security analysis.

### Interrogator Harnesses (`/interrogator-harnesses/`)

The harnesses execute Agent Interrogator against the test targets:

#### Basic Harnesses (No Metrics)
- `dvla-harness.py` - Standard DVLA harness
- `dvla-mcp-harness.py` - MCP-enabled DVLA harness
- `dvla-n-harness.py` - Neutralized DVLA harness

#### Scientific Harnesses (With Metrics)
Scientific harnesses capture detailed metrics during interrogation:

**Terminal Output Versions:**
- `dvla_harness_scientific_terminal.py`
- `dvla_mcp_harness_scientific_terminal.py`
- `dvla_n_harness_scientific_terminal.py`

**File Output Versions:**
- `dvla_harness_scientific_file.py`
- `dvla_mcp_harness_scientific_file.py`
- `dvla_n_harness_scientific_file.py`

⚠️ **Note**: Terminal output versions may exceed terminal buffer limits. File output versions are recommended for comprehensive metric capture.

### Results

Scientific results files (`scientific_results_*.txt`) will be written to the current working directory.

## Usage

1. Deploy the desired test target(s) using their respective Docker configurations
2. Configure the appropriate harness with your API credentials
3. Run the harness to execute Agent Interrogator against the target
4. Analyze the results for security insights and attack surface mapping
