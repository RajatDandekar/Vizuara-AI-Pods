#!/usr/bin/env python3
"""
Inject AI Teaching Assistant chatbot cell into Colab notebooks.

Usage:
    python scripts/inject_chatbot.py --notebook path/to/notebook.ipynb
    python scripts/inject_chatbot.py --slug world-models
    python scripts/inject_chatbot.py --all-published

The chatbot cell:
  - Reads the full notebook content at runtime via google.colab._message
  - Registers a Python callback that calls the Vizuara chat API
  - Displays a polished HTML/CSS/JS chat widget in the cell output
  - Uses google.colab.kernel.invokeFunction to bridge JS → Python
  - Tagged with metadata {"tags": ["chatbot"]} for idempotent re-injection
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_NB_DIR = PROJECT_ROOT / "public" / "notebooks"
OUTPUT_NB_DIR = PROJECT_ROOT / "output" / "notebooks"
PUBLIC_CS_DIR = PROJECT_ROOT / "public" / "case-studies"

API_URL = "https://course-creator-brown.vercel.app/api/chat"


# ── Chat widget HTML/CSS/JS ──────────────────────────────────────────────────

CHAT_WIDGET_HTML = r'''<style>
  .vc-wrap{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:100%;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,.12);background:#fff;border:1px solid #e5e7eb}
  .vc-hdr{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:16px 20px;display:flex;align-items:center;gap:12px}
  .vc-avatar{width:42px;height:42px;background:rgba(255,255,255,.2);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:22px}
  .vc-hdr h3{font-size:16px;font-weight:600;margin:0}
  .vc-hdr p{font-size:12px;opacity:.85;margin:2px 0 0}
  .vc-msgs{height:420px;overflow-y:auto;padding:16px;background:#f8f9fb;display:flex;flex-direction:column;gap:10px}
  .vc-msg{display:flex;flex-direction:column;animation:vc-fade .25s ease}
  .vc-msg.user{align-items:flex-end}
  .vc-msg.bot{align-items:flex-start}
  .vc-bbl{max-width:85%;padding:10px 14px;border-radius:16px;font-size:14px;line-height:1.55;word-wrap:break-word}
  .vc-msg.user .vc-bbl{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border-bottom-right-radius:4px}
  .vc-msg.bot .vc-bbl{background:#fff;color:#1a1a2e;border:1px solid #e8e8e8;border-bottom-left-radius:4px}
  .vc-bbl code{background:rgba(0,0,0,.07);padding:2px 6px;border-radius:4px;font-size:13px;font-family:'Fira Code',monospace}
  .vc-bbl pre{background:#1e1e2e;color:#cdd6f4;padding:12px;border-radius:8px;overflow-x:auto;margin:8px 0;font-size:13px}
  .vc-bbl pre code{background:none;padding:0;color:inherit}
  .vc-bbl h3,.vc-bbl h4{margin:10px 0 4px;font-size:15px}
  .vc-bbl ul,.vc-bbl ol{margin:4px 0;padding-left:20px}
  .vc-bbl li{margin:2px 0}
  .vc-chips{display:flex;flex-wrap:wrap;gap:8px;padding:0 16px 12px;background:#f8f9fb}
  .vc-chip{background:#fff;border:1px solid #d1d5db;border-radius:20px;padding:6px 14px;font-size:12px;cursor:pointer;transition:all .15s;color:#4b5563}
  .vc-chip:hover{border-color:#667eea;color:#667eea;background:#f0f0ff}
  .vc-input{display:flex;padding:12px 16px;background:#fff;border-top:1px solid #eee;gap:8px}
  .vc-input input{flex:1;padding:10px 16px;border:2px solid #e8e8e8;border-radius:24px;font-size:14px;outline:none;transition:border-color .2s}
  .vc-input input:focus{border-color:#667eea}
  .vc-input button{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;border:none;border-radius:50%;width:42px;height:42px;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:18px;transition:transform .1s}
  .vc-input button:hover{transform:scale(1.05)}
  .vc-input button:disabled{opacity:.5;cursor:not-allowed;transform:none}
  .vc-typing{display:flex;gap:5px;padding:4px 0}
  .vc-typing span{width:8px;height:8px;background:#667eea;border-radius:50%;animation:vc-bounce 1.4s infinite ease-in-out}
  .vc-typing span:nth-child(2){animation-delay:.2s}
  .vc-typing span:nth-child(3){animation-delay:.4s}
  @keyframes vc-bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
  @keyframes vc-fade{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
  .vc-note{text-align:center;font-size:11px;color:#9ca3af;padding:8px 16px 12px;background:#fff}
</style>
<div class="vc-wrap">
  <div class="vc-hdr">
    <div class="vc-avatar">&#129302;</div>
    <div>
      <h3>Vizuara Teaching Assistant</h3>
      <p>Ask me anything about this notebook</p>
    </div>
  </div>
  <div class="vc-msgs" id="vcMsgs">
    <div class="vc-msg bot">
      <div class="vc-bbl">&#128075; Hi! I've read through this entire notebook. Ask me about any concept, code block, or exercise &mdash; I'm here to help you learn!</div>
    </div>
  </div>
  <div class="vc-chips" id="vcChips">
    <span class="vc-chip" onclick="vcAsk(this.textContent)">Explain the main concept</span>
    <span class="vc-chip" onclick="vcAsk(this.textContent)">Help with the TODO exercise</span>
    <span class="vc-chip" onclick="vcAsk(this.textContent)">Summarize what I learned</span>
  </div>
  <div class="vc-input">
    <input type="text" id="vcIn" placeholder="Ask about concepts, code, exercises..." />
    <button id="vcSend" onclick="vcSendMsg()">&#10148;</button>
  </div>
  <div class="vc-note">AI-generated &middot; Verify important information &middot; <a href="#" onclick="vcClear();return false" style="color:#667eea">Clear chat</a></div>
</div>
<script>
(function(){
  var msgs=document.getElementById('vcMsgs'),inp=document.getElementById('vcIn'),
      btn=document.getElementById('vcSend'),chips=document.getElementById('vcChips');

  function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML}

  function md(t){
    return t
      .replace(/```(\w*)\n([\s\S]*?)```/g,function(_,l,c){return '<pre><code>'+esc(c)+'</code></pre>'})
      .replace(/`([^`]+)`/g,'<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g,'<em>$1</em>')
      .replace(/^#### (.+)$/gm,'<h4>$1</h4>')
      .replace(/^### (.+)$/gm,'<h4>$1</h4>')
      .replace(/^## (.+)$/gm,'<h3>$1</h3>')
      .replace(/^\d+\. (.+)$/gm,'<li>$1</li>')
      .replace(/^- (.+)$/gm,'<li>$1</li>')
      .replace(/\n\n/g,'<br><br>')
      .replace(/\n/g,'<br>');
  }

  function addMsg(text,isUser){
    var m=document.createElement('div');m.className='vc-msg '+(isUser?'user':'bot');
    var b=document.createElement('div');b.className='vc-bbl';
    b.innerHTML=isUser?esc(text):md(text);
    m.appendChild(b);msgs.appendChild(m);msgs.scrollTop=msgs.scrollHeight;
  }

  function showTyping(){
    var m=document.createElement('div');m.className='vc-msg bot';m.id='vcTyping';
    m.innerHTML='<div class="vc-bbl"><div class="vc-typing"><span></span><span></span><span></span></div></div>';
    msgs.appendChild(m);msgs.scrollTop=msgs.scrollHeight;
  }

  function hideTyping(){var e=document.getElementById('vcTyping');if(e)e.remove()}

  window.vcSendMsg=function(){
    var q=inp.value.trim();if(!q)return;
    inp.value='';chips.style.display='none';
    addMsg(q,true);showTyping();btn.disabled=true;
    google.colab.kernel.invokeFunction('notebook_chat',[q],{})
      .then(function(r){
        hideTyping();
        var a=r.data['application/json'];
        addMsg(typeof a==='string'?a:JSON.stringify(a),false);
      })
      .catch(function(){
        hideTyping();
        addMsg('Sorry, I encountered an error. Please check your internet connection and try again.',false);
      })
      .finally(function(){btn.disabled=false;inp.focus()});
  };

  window.vcAsk=function(q){inp.value=q;vcSendMsg()};
  window.vcClear=function(){
    msgs.innerHTML='<div class="vc-msg bot"><div class="vc-bbl">&#128075; Chat cleared. Ask me anything!</div></div>';
    chips.style.display='flex';
  };

  inp.addEventListener('keypress',function(e){if(e.key==='Enter')vcSendMsg()});
  inp.focus();
})();
</script>'''


# ── Cell source generation ───────────────────────────────────────────────────


def make_chatbot_cell(api_url: str = API_URL) -> dict:
    """Create the chatbot notebook cell."""

    # Python code that runs when the cell is executed
    python_code = f'''#@title \\U0001f4ac AI Teaching Assistant — Click \\u25b6 to start
#@markdown This AI chatbot has read your entire notebook and can answer questions about any concept, code, or exercise.

import json as _json
import requests as _requests
from google.colab import output as _output
from IPython.display import display, HTML as _HTML, Markdown as _Markdown

# --- Read notebook content for context ---
def _get_notebook_context():
    try:
        from google.colab import _message
        nb = _message.blocking_request('get_ipynb', request='', timeout_sec=10)
        cells = nb.get('ipynb', {{}}).get('cells', [])
        parts = []
        for cell in cells:
            src = ''.join(cell.get('source', []))
            tags = cell.get('metadata', {{}}).get('tags', [])
            if 'chatbot' in tags:
                continue
            if src.strip():
                ct = cell.get('cell_type', 'unknown')
                parts.append(f'[{{ct.upper()}}]\\n{{src}}')
        return '\\n\\n---\\n\\n'.join(parts)
    except Exception:
        return 'Notebook content unavailable.'

_NOTEBOOK_CONTEXT = _get_notebook_context()
_CHAT_HISTORY = []
_API_URL = "{api_url}"

def _notebook_chat(question):
    global _CHAT_HISTORY
    try:
        resp = _requests.post(_API_URL, json={{
            'question': question,
            'context': _NOTEBOOK_CONTEXT[:100000],
            'history': _CHAT_HISTORY[-10:],
        }}, timeout=60)
        data = resp.json()
        answer = data.get('answer', 'Sorry, I could not generate a response.')
        _CHAT_HISTORY.append({{'role': 'user', 'content': question}})
        _CHAT_HISTORY.append({{'role': 'assistant', 'content': answer}})
        return answer
    except Exception as e:
        return f'Error connecting to teaching assistant: {{str(e)}}'

_output.register_callback('notebook_chat', _notebook_chat)

def ask(question):
    """Ask the AI teaching assistant a question about this notebook.
    Usage: ask('What does the reparameterization trick do?')
    """
    answer = _notebook_chat(question)
    display(_Markdown(answer))

print("\\u2705 AI Teaching Assistant is ready!")
print("\\U0001f4a1 Use the chat widget below, or call ask('your question') in any cell.")

# --- Display chat widget ---
display(_HTML(\\'\\'\\'{widget_html}\\'\\'\\')))'''

    # The above f-string approach is too complex with escaping.
    # Let's build the source list directly instead.
    return None  # placeholder — we'll use _build_source() below


def _build_source(api_url: str = API_URL) -> list[str]:
    """Build the cell source as a list of strings (notebook JSON format)."""
    lines = []

    # Title and markdown
    lines.append('#@title \U0001f4ac AI Teaching Assistant \u2014 Click \u25b6 to start\n')
    lines.append('#@markdown This AI chatbot reads your notebook and can answer questions about any concept, code, or exercise.\n')
    lines.append('\n')

    # Imports
    lines.append('import json as _json\n')
    lines.append('import requests as _requests\n')
    lines.append('from google.colab import output as _output\n')
    lines.append('from IPython.display import display, HTML as _HTML, Markdown as _Markdown\n')
    lines.append('\n')

    # Read notebook context
    lines.append('# --- Read notebook content for context ---\n')
    lines.append('def _get_notebook_context():\n')
    lines.append('    try:\n')
    lines.append('        from google.colab import _message\n')
    lines.append('        nb = _message.blocking_request("get_ipynb", request="", timeout_sec=10)\n')
    lines.append('        cells = nb.get("ipynb", {}).get("cells", [])\n')
    lines.append('        parts = []\n')
    lines.append('        for cell in cells:\n')
    lines.append('            src = "".join(cell.get("source", []))\n')
    lines.append('            tags = cell.get("metadata", {}).get("tags", [])\n')
    lines.append('            if "chatbot" in tags:\n')
    lines.append('                continue\n')
    lines.append('            if src.strip():\n')
    lines.append('                ct = cell.get("cell_type", "unknown")\n')
    lines.append('                parts.append(f"[{ct.upper()}]\\n{src}")\n')
    lines.append('        return "\\n\\n---\\n\\n".join(parts)\n')
    lines.append('    except Exception:\n')
    lines.append('        return "Notebook content unavailable."\n')
    lines.append('\n')

    # State and API config
    lines.append('_NOTEBOOK_CONTEXT = _get_notebook_context()\n')
    lines.append('_CHAT_HISTORY = []\n')
    lines.append(f'_API_URL = "{api_url}"\n')
    lines.append('\n')

    # Chat handler
    lines.append('def _notebook_chat(question):\n')
    lines.append('    global _CHAT_HISTORY\n')
    lines.append('    try:\n')
    lines.append('        resp = _requests.post(_API_URL, json={\n')
    lines.append("            'question': question,\n")
    lines.append("            'context': _NOTEBOOK_CONTEXT[:100000],\n")
    lines.append("            'history': _CHAT_HISTORY[-10:],\n")
    lines.append('        }, timeout=60)\n')
    lines.append('        data = resp.json()\n')
    lines.append("        answer = data.get('answer', 'Sorry, I could not generate a response.')\n")
    lines.append("        _CHAT_HISTORY.append({'role': 'user', 'content': question})\n")
    lines.append("        _CHAT_HISTORY.append({'role': 'assistant', 'content': answer})\n")
    lines.append('        return answer\n')
    lines.append('    except Exception as e:\n')
    lines.append("        return f'Error connecting to teaching assistant: {str(e)}'\n")
    lines.append('\n')

    # Register callback
    lines.append("_output.register_callback('notebook_chat', _notebook_chat)\n")
    lines.append('\n')

    # Convenience function
    lines.append('def ask(question):\n')
    lines.append('    """Ask the AI teaching assistant a question about this notebook."""\n')
    lines.append('    answer = _notebook_chat(question)\n')
    lines.append('    display(_Markdown(answer))\n')
    lines.append('\n')

    # Status message
    lines.append('print("\\u2705 AI Teaching Assistant is ready!")\n')
    lines.append('print("\\U0001f4a1 Use the chat below, or call ask(\\\'your question\\\') in any cell.")\n')
    lines.append('\n')

    # Display the HTML widget
    lines.append('# --- Display chat widget ---\n')
    lines.append("display(_HTML('''" + CHAT_WIDGET_HTML + "'''))")

    return lines


def make_chatbot_cell(api_url: str = API_URL) -> dict:
    """Create the chatbot notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["chatbot"], "cellView": "form"},
        "outputs": [],
        "source": _build_source(api_url),
        "id": "vizuara_chatbot",
    }


# ── Injection ─────────────────────────────────────────────────────────────────


def inject(notebook_path: str, api_url: str = API_URL):
    """Inject chatbot cell into a notebook (idempotent)."""
    with open(notebook_path) as f:
        nb = json.load(f)

    original_count = len(nb["cells"])

    # Remove existing chatbot cells
    nb["cells"] = [
        c for c in nb["cells"]
        if "chatbot" not in c.get("metadata", {}).get("tags", [])
    ]
    removed = original_count - len(nb["cells"])

    # Add chatbot cell at the end
    nb["cells"].append(make_chatbot_cell(api_url))

    # Save
    with open(notebook_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    status = f"replaced {removed}" if removed else "added"
    print(f"  [+] Chatbot {status} in {Path(notebook_path).name}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Inject AI Teaching Assistant chatbot into Colab notebooks"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--notebook",
        help="Path to a single .ipynb notebook"
    )
    group.add_argument(
        "--slug",
        help="Inject into all notebooks for a course slug (in output/ and public/)"
    )
    group.add_argument(
        "--all-published",
        action="store_true",
        help="Inject into all published notebooks in public/notebooks/"
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"Chat API endpoint URL (default: {API_URL})"
    )
    args = parser.parse_args()

    if args.notebook:
        path = Path(args.notebook)
        if not path.exists():
            print(f"Error: {path} not found")
            sys.exit(1)
        inject(str(path), args.api_url)

    elif args.slug:
        slug = args.slug
        count = 0
        for nb_dir in [OUTPUT_NB_DIR / slug, PUBLIC_NB_DIR / slug]:
            if nb_dir.exists():
                for nb in sorted(nb_dir.glob("*.ipynb")):
                    if nb.name == "00_index.ipynb":
                        continue
                    inject(str(nb), args.api_url)
                    count += 1
        # Also check case study notebooks
        cs_nb = PUBLIC_CS_DIR / slug / "case_study_notebook.ipynb"
        if cs_nb.exists():
            inject(str(cs_nb), args.api_url)
            count += 1
        print(f"\nInjected chatbot into {count} notebook(s) for '{slug}'")

    elif args.all_published:
        count = 0
        if PUBLIC_NB_DIR.exists():
            for slug_dir in sorted(PUBLIC_NB_DIR.iterdir()):
                if not slug_dir.is_dir():
                    continue
                for nb in sorted(slug_dir.glob("*.ipynb")):
                    if nb.name == "00_index.ipynb":
                        continue
                    inject(str(nb), args.api_url)
                    count += 1
        # Case study notebooks
        if PUBLIC_CS_DIR.exists():
            for slug_dir in sorted(PUBLIC_CS_DIR.iterdir()):
                if not slug_dir.is_dir():
                    continue
                cs_nb = slug_dir / "case_study_notebook.ipynb"
                if cs_nb.exists():
                    inject(str(cs_nb), args.api_url)
                    count += 1
        print(f"\nInjected chatbot into {count} notebook(s)")


if __name__ == "__main__":
    main()
