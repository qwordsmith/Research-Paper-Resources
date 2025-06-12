from langchain.tools import StructuredTool
import json
import os
import asyncio
from typing import Optional, Dict, Any, Union, List
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, Field, ConfigDict
import tiktoken

def truncate_response(text: str, max_tokens: int = 1000) -> str:
    """Truncate text to stay within token limit."""
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens]) + "... [truncated]"
    except Exception:
        # Fallback to character-based truncation if tiktoken fails
        chars_per_token = 4  # Rough estimate
        max_chars = max_tokens * chars_per_token
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "... [truncated]"

def create_mcp_tool(mcp_client: Client, tool_info) -> StructuredTool:
    """Create a Langchain Tool from MCP tool information."""
    
    # Create a Pydantic model for the tool's parameters
    class ToolParameters(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
    param_fields = {}
    if hasattr(tool_info, 'parameters'):
        for param in tool_info.parameters:
            param_fields[param] = (str, Field(description=f"Parameter {param} for {tool_info.name}"))
    else:
        param_fields["query"] = (str, Field(description="Input query for the tool"))
    
    # Create the parameter model with the fields
    tool_model = type(
        f"{tool_info.name}Parameters",
        (ToolParameters,),
        {"__annotations__": param_fields}
    )
    
    async def tool_func_async(params: Dict[str, Any]) -> Any:
        try:
            async with mcp_client:
                result = await mcp_client.call_tool(
                    tool_info.name,
                    {
                        "instructions": f"Execute the {tool_info.name} tool with the following parameters",
                        **params
                    }
                )
                if result and len(result) > 0:
                    response = json.loads(result[0].text)
                    # If response is a string, truncate it directly
                    if isinstance(response, str):
                        return truncate_response(response)
                    # If response is a dict/list, truncate any string values
                    elif isinstance(response, dict):
                        return {k: truncate_response(v) if isinstance(v, str) else v 
                               for k, v in response.items()}
                    elif isinstance(response, list):
                        return [truncate_response(item) if isinstance(item, str) else item 
                               for item in response]
                    return response
                return None
        except Exception as e:
            return f"Error invoking MCP tool: {str(e)}"
    
    def tool_func(query: Optional[str] = None, **kwargs) -> Any:
        # Handle both string query and structured parameters
        if query is not None:
            params = {"query": query}
        else:
            params = kwargs
        return asyncio.run(tool_func_async(params))
    
    return StructuredTool(
        name=tool_info.name,
        description=tool_info.description,
        func=tool_func,
        args_schema=tool_model
    )

async def get_mcp_tools_async() -> List[StructuredTool]:
    """Fetch and create Langchain tools from MCP server asynchronously."""
    mcp_url = os.getenv("MCP_SSE_URL")
    if not mcp_url:
        raise ValueError("MCP_SSE_URL environment variable not set")
    
    # Initialize FastMCP client with StreamableHttpTransport
    transport = StreamableHttpTransport(mcp_url)
    client = Client(transport=transport)
    
    # Use client within context manager
    async with client:
        # Fetch available tools
        tools_info = await client.list_tools()
        
        # Create Langchain tools
        return [create_mcp_tool(client, tool_info) for tool_info in tools_info]

def get_mcp_tools() -> List[StructuredTool]:
    """Synchronous wrapper for get_mcp_tools_async."""
    return asyncio.run(get_mcp_tools_async())
