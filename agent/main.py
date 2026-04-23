"""
Main entry point — Gradio chat interface (Gemini version, compatible with Gradio 6)

How to run:
  cd agent
  python main.py

Then open http://localhost:7860 in your browser
"""

import gradio as gr
import os
import sys

# Add the parent directory to the path so we can import tools and agent_core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_core import AcademicAgent

# ============================================================
# Initialize the Agent (global singleton, maintains conversation state across requests)
# ============================================================

try:
    agent = AcademicAgent()
    init_error = None
except Exception as e:
    agent = None
    init_error = str(e)


# ============================================================
# Gradio callback functions
# ============================================================

def respond(message: str, chat_history: list) -> tuple:
    """Process user input and return the updated conversation history"""
    if not message.strip():
        return "", chat_history, ""

    if init_error:
        # Gradio 6 requires dict format, not tuples
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": f"❌ Initialization failed: {init_error}"})
        return "", chat_history, ""

    try:
        reply, tool_calls = agent.chat(message)

        # Correct format for Gradio 6: dicts, not tuples!
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})

        # Right-side status panel: show tool call info
        if tool_calls:
            tool_info = "🔧 Tools used this turn:\n" + "\n".join(tool_calls)
        else:
            tool_info = "💬 No tools used (direct response)"

        # Show how many conversation turns are currently in short-term memory
        memory_turns = len(agent.chat_history) // 2
        tool_info += f"\n\n🧠 Memory: {memory_turns} turns in context"

        return "", chat_history, tool_info

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", chat_history, error_msg


def clear_conversation() -> tuple:
    """Clear conversation history (clear short-term memory)"""
    if agent:
        agent.clear_memory()
    return [], "🗑️ Memory cleared. Starting fresh conversation."


# ============================================================
# Build the Gradio interface (Gradio 6 compatible: theme moved to launch(), removed show_copy_button)
# ============================================================

with gr.Blocks(title="Academic Research Agent") as demo:

    gr.Markdown("""
    # 🎓 Academic Research Assistant Agent
    ### COM6104 Group Project — AI Agent with Tool Use & Short-term Memory
    **Powered by Google Gemini (Free)**

    **How to use:**
    - 🔍 Find papers: `"Find papers about diffusion models"`
    - 📝 Summarize: `"Summarize this abstract: [paste text]"`
    - 🔗 Chain tasks: `"Find papers on LoRA and summarize the first one"`
    - 🧠 The agent **remembers** previous messages in this session
    """)

    with gr.Row():
        # Left chat area
        with gr.Column(scale=3):
            # Gradio 6 Chatbot no longer has the show_copy_button parameter
            chatbot = gr.Chatbot(label="Chat", height=500)

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask me to find papers or summarize text...",
                    label="Your message",
                    scale=4,
                    autofocus=True
                )
                send_btn = gr.Button("Send 🚀", variant="primary", scale=1)

            clear_btn = gr.Button("🗑️ Clear Memory & Start Over", variant="secondary")

        # Right status panel
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Agent Status")
            tool_status = gr.Textbox(
                label="Tool Calls & Memory",
                value="Waiting for your message...",
                lines=10,
                interactive=False  # read-only
            )
            gr.Markdown("""
            ### 📖 Example Questions
            ```
            Find 3 papers about RAG

            Find papers on GPT-4 and
            summarize the first result

            What is a diffusion model?
            Find related papers.
            ```
            """)

    # Bind events
    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, tool_status]
    )
    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, tool_status]
    )
    clear_btn.click(
        fn=clear_conversation,
        outputs=[chatbot, tool_status]
    )


# ============================================================
# Start the server
# ============================================================

if __name__ == "__main__":
    print("🚀 Starting Academic Research Agent (Gemini version)...")
    print("📡 Open http://localhost:7860 in your browser")
    if init_error:
        print(f"⚠️  Warning: {init_error}")
    else:
        print("✅ Gemini API configured successfully!")

    # In Gradio 6, the theme parameter moves here
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()  # Correct syntax for Gradio 6
    )
