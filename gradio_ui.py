import asyncio
import os
import httpx
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TENANT_ID = os.getenv("TENANT_ID", "t1")
SESSION_ID = os.getenv("SESSION_ID", "demo-memory-session")
PERSONAS = {
    "Budget Coach": "budget_coach",
    "Pro Analyst": "pro_analyst",
}

async def ingest_file(file_obj, tenant_id: str):
    # Accept None, string path, or Gradio FileData-like object
    if not file_obj:
        return "Please choose a file first."

    # Normalize to (filename, bytes)
    filename = None
    blob = None

    # Case A: we configured type="filepath" -> str path
    if isinstance(file_obj, str):
        path = file_obj
        filename = os.path.basename(path)
        with open(path, "rb") as f:
            blob = f.read()

    else:
        # Case B: older/newer Gradio returns an object/dict
        # Try attributes: .path (temp path), .orig_name (original)
        path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None)
        orig = getattr(file_obj, "orig_name", None) or getattr(file_obj, "name", None)

        if path and os.path.exists(path):
            filename = os.path.basename(orig or path)
            with open(path, "rb") as f:
                blob = f.read()
        else:
            # Last-ditch: maybe it's a dict
            if isinstance(file_obj, dict):
                path = file_obj.get("path") or file_obj.get("name")
                if path and os.path.exists(path):
                    filename = os.path.basename(file_obj.get("orig_name") or path)
                    with open(path, "rb") as f:
                        blob = f.read()

    if not blob or not filename:
        return "Please choose a valid file (couldn't read temp path)."

    async with httpx.AsyncClient(timeout=120) as client:
        files = {"file": (filename, blob, "application/octet-stream")}
        data = {"tenant_id": tenant_id}
        r = await client.post(f"{API_BASE}/ingest_pg", data=data, files=files)
        try:
            r.raise_for_status()
            return f"✅ Ingested: {filename} → {r.json()}"
        except Exception:
            return f"❌ Ingest failed: {r.text}"

async def chat(messages, persona_label, tenant_id, session_id):
    persona_id = PERSONAS.get(persona_label, "budget_coach")

    # Gradio Chatbot history is a list of [user, assistant] pairs.
    # Flatten to [{"role": "user", ...}, {"role": "assistant", ...}, ...]
    api_messages = []
    for user_msg, assistant_msg in messages:
        if user_msg is not None and user_msg != "":
            api_messages.append({"role": "user", "content": user_msg})
        if assistant_msg is not None and assistant_msg != "":
            api_messages.append({"role": "assistant", "content": assistant_msg})

    # Safety: ensure we have at least one user message
    if not api_messages or api_messages[-1]["role"] != "user":
        # If the last assistant turn is present and user pressed send again,
        # append the pending input in on_send; here we just avoid bad payloads.
        pass

    payload = {
        "session_id": session_id,
        "tenant_id": tenant_id,
        "persona_id": persona_id,
        "messages": api_messages,
    }

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{API_BASE}/agent/run", json=payload)
        try:
            r.raise_for_status()
            data = r.json()
            answer = data.get("assistant_message", "")
            memory_writes = data.get("memory_writes", [])
            if not memory_writes:
                return answer, "No new memory writes."

            write_lines = []
            for i, memory_write in enumerate(memory_writes, 1):
                memory_type = memory_write.get("memory_type", "unknown")
                memory_value = memory_write.get("value", {})
                content = memory_value.get("content", "")
                write_lines.append(f"[{i}] {memory_type}: {content}")
            return answer, "\n".join(write_lines)
        except Exception:
            return f"❌ Agent run failed: {r.text}", ""

def sync_ingest(file_obj, tenant_id):
    return asyncio.run(ingest_file(file_obj, tenant_id))

def sync_chat(history, persona_label, tenant_id, session_id):
    answer, run_info = asyncio.run(chat(history, persona_label, tenant_id, session_id))
    # Append assistant turn for Gradio chat UI
    history = history + [[None, answer]]
    return history, run_info

with gr.Blocks(title="Financial Advisor Bot") as demo:
    gr.Markdown("## Financial Advisor Bot — Demo UI\nUpload docs, then chat with a chosen persona. Backend: FastAPI + Postgres + pgvector + OpenAI.")
    with gr.Row():
        with gr.Column(scale=3):
            tenant = gr.Textbox(value=TENANT_ID, label="Tenant ID")
            session_id = gr.Textbox(value=SESSION_ID, label="Session ID")
            persona = gr.Dropdown(choices=list(PERSONAS.keys()), value="Budget Coach", label="Persona")
            uploader = gr.File(label="Upload CSV/TXT/PDF", file_types=[".csv", ".txt", ".pdf"],  type="filepath")
            ingest_btn = gr.Button("Ingest to /ingest_pg")
            ingest_out = gr.Markdown()
            ingest_btn.click(fn=sync_ingest, inputs=[uploader, tenant], outputs=[ingest_out])

            chatbox = gr.Chatbot(height=340, label="Chat")
            user_in = gr.Textbox(placeholder="Ask about June spend…", label="Your message")
            send = gr.Button("Send")
            cites = gr.Textbox(label="Run details", lines=8)
            def on_send(user_msg, history, persona_label, tenant_id, current_session_id):
                if not user_msg.strip():
                    return "", history, cites.value
                history = history + [[user_msg, None]]
                history, citations = sync_chat(history, persona_label, tenant_id, current_session_id)
                return "", history, citations

            send.click(
                fn=on_send,
                inputs=[user_in, chatbox, persona, tenant, session_id],
                outputs=[user_in, chatbox, cites],
            )
        with gr.Column(scale=2):
            gr.Markdown('### Tips\n- Keep the same Session ID across reloads to preserve memory retrieval context.\n- Ask: *"I prefer low-risk ETFs"*, then ask later: *"What investment style do I like?"*')
            gr.Markdown("### Status")
            status = gr.JSON(value={"api_base": API_BASE})

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
