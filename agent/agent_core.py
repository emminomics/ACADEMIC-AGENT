"""
Agent core logic (Gemini version, using the new google.genai SDK)

Core idea (ReAct pattern):
  User input → Gemini reasoning → decide to call a tool → execute tool → feed result back to Gemini → continue reasoning → final answer

Short-term memory implementation:
  Pass the full conversation history (chat_history) to Gemini so it can "remember" what was said before
"""

import os
#  SDK: google.genai
from google import genai
from google.genai import types as genai_types

from tools.arxiv_tool import search_arxiv, format_papers
from tools.summarize_tool import summarize_text


# ============================================================
# Fill in your API Key directly here (no environment variable needed)
# Note: Remove the key before uploading to GitHub!
# ============================================================

API_KEY = ""  # key


# ============================================================
# Define tools (FunctionDeclaration format for the new SDK)
# ============================================================

search_arxiv_func = genai_types.FunctionDeclaration(
    name="search_arxiv",
    description=(
        "Search for academic papers on arXiv by keyword. "
        "Returns paper titles, authors, abstracts, and PDF links. "
        "Use this when the user wants to find papers on a topic."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "query": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Search keywords in English"
            ),
            "max_results": genai_types.Schema(
                type=genai_types.Type.INTEGER,
                description="Maximum number of papers to return (default 5)"
            ),
        },
        required=["query"]
    )
)

summarize_text_func = genai_types.FunctionDeclaration(
    name="summarize_text",
    description=(
        "Summarize a piece of text and extract its keywords. "
        "Use this when the user wants to understand a long abstract or passage."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "text": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="The text to summarize"
            ),
        },
        required=["text"]
    )
)

# Pack tools into a Tool object to pass to the model
TOOLS = [genai_types.Tool(function_declarations=[search_arxiv_func, summarize_text_func])]


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Actually execute a tool call

    Args:
        tool_name: Name of the tool
        tool_args: Dictionary of tool arguments

    Returns:
        Tool execution result as a string
    """
    if tool_name == "search_arxiv":
        papers = search_arxiv(
            query=tool_args["query"],
            max_results=tool_args.get("max_results", 5)
        )
        return format_papers(papers)

    elif tool_name == "summarize_text":
        result = summarize_text(tool_args["text"])
        return (
            f"Summary:\n{result['summary']}\n\n"
            f"Keywords: {', '.join(result['keywords'])}\n"
            f"Word count: {result['word_count']}"
        )

    return f"Unknown tool: {tool_name}"


class AcademicAgent:
    """
    Academic Research AI Agent (new google.genai SDK)

    Key features:
    - Short-term memory: maintains chat_history, passing the full history to Gemini each turn
    - Tool calling: Gemini uses Function Calling to automatically decide when to call which tool
    - Multi-step reasoning: can search for papers first, then automatically summarize results
    """

    def __init__(self):
        # Create the new Gemini client
        self.client = genai.Client(api_key=API_KEY)

        # System prompt: defines the Agent's "persona"
        self.system_prompt = (
            "You are an Academic Research Assistant Agent. "
            "You help users find and understand academic papers. "
            "You have two tools: search_arxiv and summarize_text. "
            "When users want to find papers, use search_arxiv. "
            "When they want to understand text, use summarize_text. "
            "You can chain tools: search first, then summarize results. "
            "Always explain findings clearly and helpfully."
        )

        # Short-term memory: stores the full conversation history
        # Format: [Content(role="user"/"model", parts=[...])]
        self.chat_history = []

    def clear_memory(self):
        """Clear short-term memory (start a new conversation)"""
        self.chat_history = []

    def chat(self, user_message: str) -> tuple:
        """
        Process user input and return the Agent's reply

        Args:
            user_message: The user's input text

        Returns:
            (agent_reply, tool_calls_used)
        """
        tool_calls_used = []  # Track which tools were used this turn

        # Add the user message to conversation history (short-term memory)
        self.chat_history.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_message)]
            )
        )

        # --------------------------------------------------------
        # Agent main loop (ReAct pattern):
        # Gemini → decide to call a tool → execute tool → feed result back to Gemini → continue
        # --------------------------------------------------------
        while True:
            # Call Gemini with the full conversation history (= short-term memory)
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=self.chat_history,
                config=genai_types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    tools=TOOLS,
                )
            )

            candidate = response.candidates[0]

            # Check whether the response contains any tool calls
            has_tool_call = False
            tool_response_parts = []

            for part in candidate.content.parts:
                # If Gemini decides to call a tool
                if part.function_call:
                    has_tool_call = True
                    tool_name = part.function_call.name
                    tool_args = dict(part.function_call.args)

                    # Record the tool call (for display in the UI)
                    args_preview = str(tool_args)[:80]
                    tool_calls_used.append(f"🔧 {tool_name}({args_preview})")

                    # Actually execute the tool
                    try:
                        result = execute_tool(tool_name, tool_args)
                    except Exception as e:
                        result = f"Tool error: {str(e)}"

                    # Collect tool return results (new SDK format)
                    tool_response_parts.append(
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=tool_name,
                                response={"result": result}
                            )
                        )
                    )

            if has_tool_call:
                # Record Gemini's tool call intent into history
                self.chat_history.append(candidate.content)
                # Add tool execution results to history so Gemini can see them
                self.chat_history.append(
                    genai_types.Content(
                        role="user",
                        parts=tool_response_parts
                    )
                )
                # Continue the loop: Gemini will continue reasoning after seeing the results
                continue

            # No tool call → Gemini gives the final text answer
            final_reply = ""
            for part in candidate.content.parts:
                if part.text:
                    final_reply += part.text

            # Record the final reply into history (short-term memory)
            self.chat_history.append(candidate.content)

            return final_reply, tool_calls_used
