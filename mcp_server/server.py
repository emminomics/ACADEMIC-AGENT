"""
MCP Server
Wraps two tools into the standard MCP protocol; can be tested with MCP Inspector

How to test:
  npx @modelcontextprotocol/inspector python mcp_server/server.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from tools.arxiv_tool import search_arxiv, format_papers
from tools.summarize_tool import summarize_text

server = Server("academic-agent")


@server.list_tools()
async def list_tools() -> list:
    """List all available tools (MCP Inspector calls this endpoint to discover tools)"""
    return [
        types.Tool(
            name="search_arxiv",
            description="Search for academic papers on arXiv by keyword.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords"},
                    "max_results": {"type": "integer", "description": "Max papers (default 5)", "default": 5}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="summarize_text",
            description="Summarize a piece of text and extract keywords.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to summarize"}
                },
                "required": ["text"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    """Handle tool call requests"""
    if name == "search_arxiv":
        try:
            papers = search_arxiv(arguments["query"], arguments.get("max_results", 5))
            result = format_papers(papers)
        except Exception as e:
            result = f"Error: {str(e)}"
        return [types.TextContent(type="text", text=result)]

    elif name == "summarize_text":
        try:
            r = summarize_text(arguments["text"])
            result = (f"Summary:\n{r['summary']}\n\n"
                      f"Keywords: {', '.join(r['keywords'])}\n"
                      f"Word count: {r['word_count']}")
        except Exception as e:
            result = f"Error: {str(e)}"
        return [types.TextContent(type="text", text=result)]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Start the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
