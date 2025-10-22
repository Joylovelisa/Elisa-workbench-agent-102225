import os
import io
import json
import time
import yaml
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# ============================================================================
# THEME CONFIGURATIONS - 20 Beautiful Themes
# ============================================================================
THEMES = {
    "Streamlit Default": {
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#31333F",
        "font": "sans serif"
    },
    "Ocean Blue": {
        "primaryColor": "#0066CC",
        "backgroundColor": "#F0F8FF",
        "secondaryBackgroundColor": "#E6F3FF",
        "textColor": "#1A1A1A",
        "font": "sans serif"
    },
    "Forest Green": {
        "primaryColor": "#2E7D32",
        "backgroundColor": "#F1F8F4",
        "secondaryBackgroundColor": "#E8F5E9",
        "textColor": "#1B5E20",
        "font": "sans serif"
    },
    "Sunset Orange": {
        "primaryColor": "#FF6B35",
        "backgroundColor": "#FFF8F3",
        "secondaryBackgroundColor": "#FFE5D9",
        "textColor": "#2D2D2D",
        "font": "sans serif"
    },
    "Royal Purple": {
        "primaryColor": "#7B2CBF",
        "backgroundColor": "#FAF5FF",
        "secondaryBackgroundColor": "#F3E5F5",
        "textColor": "#2D2D2D",
        "font": "sans serif"
    },
    "Cherry Blossom": {
        "primaryColor": "#E91E63",
        "backgroundColor": "#FFF0F5",
        "secondaryBackgroundColor": "#FCE4EC",
        "textColor": "#2D2D2D",
        "font": "sans serif"
    },
    "Midnight Dark": {
        "primaryColor": "#00D9FF",
        "backgroundColor": "#0E1117",
        "secondaryBackgroundColor": "#1E2130",
        "textColor": "#FAFAFA",
        "font": "monospace"
    },
    "Nord": {
        "primaryColor": "#88C0D0",
        "backgroundColor": "#2E3440",
        "secondaryBackgroundColor": "#3B4252",
        "textColor": "#ECEFF4",
        "font": "sans serif"
    },
    "Solarized Dark": {
        "primaryColor": "#268BD2",
        "backgroundColor": "#002B36",
        "secondaryBackgroundColor": "#073642",
        "textColor": "#839496",
        "font": "monospace"
    },
    "Dracula": {
        "primaryColor": "#BD93F9",
        "backgroundColor": "#282A36",
        "secondaryBackgroundColor": "#44475A",
        "textColor": "#F8F8F2",
        "font": "monospace"
    },
    "Monokai": {
        "primaryColor": "#F92672",
        "backgroundColor": "#272822",
        "secondaryBackgroundColor": "#3E3D32",
        "textColor": "#F8F8F2",
        "font": "monospace"
    },
    "Cyberpunk": {
        "primaryColor": "#FF00FF",
        "backgroundColor": "#0A0E27",
        "secondaryBackgroundColor": "#1A1F3A",
        "textColor": "#00FFFF",
        "font": "monospace"
    },
    "Mint Fresh": {
        "primaryColor": "#00BFA5",
        "backgroundColor": "#F0FFF4",
        "secondaryBackgroundColor": "#E0F2E9",
        "textColor": "#1B4D3E",
        "font": "sans serif"
    },
    "Coffee": {
        "primaryColor": "#8B4513",
        "backgroundColor": "#FFF8DC",
        "secondaryBackgroundColor": "#F5E6D3",
        "textColor": "#3E2723",
        "font": "serif"
    },
    "Lavender": {
        "primaryColor": "#9C27B0",
        "backgroundColor": "#F3E5F5",
        "secondaryBackgroundColor": "#E1BEE7",
        "textColor": "#4A148C",
        "font": "sans serif"
    },
    "Arctic": {
        "primaryColor": "#00ACC1",
        "backgroundColor": "#E0F7FA",
        "secondaryBackgroundColor": "#B2EBF2",
        "textColor": "#006064",
        "font": "sans serif"
    },
    "Rose Gold": {
        "primaryColor": "#B76E79",
        "backgroundColor": "#FFF5F7",
        "secondaryBackgroundColor": "#FFE4E8",
        "textColor": "#5D4037",
        "font": "serif"
    },
    "Emerald": {
        "primaryColor": "#00897B",
        "backgroundColor": "#E0F2F1",
        "secondaryBackgroundColor": "#B2DFDB",
        "textColor": "#004D40",
        "font": "sans serif"
    },
    "Amber": {
        "primaryColor": "#FF8F00",
        "backgroundColor": "#FFF8E1",
        "secondaryBackgroundColor": "#FFECB3",
        "textColor": "#FF6F00",
        "font": "sans serif"
    },
    "Steel Gray": {
        "primaryColor": "#546E7A",
        "backgroundColor": "#ECEFF1",
        "secondaryBackgroundColor": "#CFD8DC",
        "textColor": "#263238",
        "font": "sans serif"
    }
}

APP_TITLE = "🚀 Agentic Dataset Workbench Pro"
APP_DESC = "Multi-theme, multi-provider agentic dataset processing with OpenAI, Gemini, and Grok"

DEFAULT_SAMPLE_JSON = [
    {"entity":"Documentation Level","title":"Risk-based documentation level","context":"Determines Basic vs Enhanced software documentation based on hazard of death/serious injury prior to risk controls.","keywords":["Basic","Enhanced","risk","hazardous situation"]},
    {"entity":"Basic Documentation Level","title":"Lower-risk software documentation","context":"Applies when failures would not present probable risk of death/serious injury before risk controls.","keywords":["risk","software","510k"]},
    {"entity":"Enhanced Documentation Level","title":"Higher-risk software documentation","context":"Applies when failures could present probable risk of death/serious injury before risk controls; includes SDS and detailed tests.","keywords":["high risk","SDS","unit/integration tests"]},
    {"entity":"Device software function","title":"Software that is a medical device","context":"A software function meeting FD&C Act 201(h).","keywords":["function","device","FDCA"]},
    {"entity":"Off-the-Shelf Software","title":"OTS software components","context":"Software for which manufacturer lacks full lifecycle control (OS, libraries).","keywords":["OTS","COTS","libraries"]},
    {"entity":"Serious injury","title":"Reportable harm definition","context":"Life-threatening, permanent impairment, or necessitating intervention to preclude permanence.","keywords":["21 CFR 803.3(w)","harm"]},
    {"entity":"Software verification","title":"Phase-output conformance","context":"Confirms outputs meet inputs; includes code reviews, inspections, testing, traceability.","keywords":["verification","reviews","testing"]},
    {"entity":"Software validation","title":"Meets user needs/intended use","context":"Objective evidence in actual/simulated use; relies on prior verification.","keywords":["validation","use environment"]},
    {"entity":"QSR Design Controls","title":"21 CFR 820.30","context":"Requires software validation and risk analysis; DHF maintenance.","keywords":["QSR","DHF","design inputs"]},
    {"entity":"Risk Management Plan","title":"Plan for risk activities","context":"Sets acceptability criteria and overall residual risk method.","keywords":["ISO 14971","criteria"]}
]

PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano", "gpt-5-mini"]
    },
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    },
    "grok": {
        "env_key": "GROK_API_KEY",
        "models": ["grok-4-fast-reasoning", "grok-3-mini"]
    }
}

DEFAULT_AGENTS_YAML = """agents:
  - name: Summarizer
    description: Summarize dataset entries into a concise executive overview.
    system_prompt: |
      You are a helpful data analyst. Provide concise, accurate summaries.
    user_prompt: |
      Summarize the following dataset. Focus on key entities, themes, and any risk-related concepts.
      Dataset preview (first 10 rows):
      {dataset_preview}
    params:
      temperature: 0.2
      max_tokens: 800
      top_p: 1.0
      force_json: false

  - name: Keyword Extractor
    description: Extract normalized keywords across records, counting frequency.
    system_prompt: |
      You are an NLP assistant. Return high-quality, deduplicated keywords with counts.
    user_prompt: |
      Extract normalized keywords with frequency from the dataset. Return a JSON with:
      {{
        "keywords": [{{"keyword": str, "count": int}}]
      }}
      Use the first 200 records for performance. If keywords array is empty, return an empty array.
      Dataset sample:
      {dataset_sample_200}
    params:
      temperature: 0.1
      max_tokens: 2048
      top_p: 0.9
      force_json: true

  - name: Risk Classifier
    description: Classify each record into Low, Medium, High risk, with rationale.
    system_prompt: |
      You are Grok, an intelligent classifier of regulatory risk.
      Classify records by the potential for serious injury prior to risk controls.
    user_prompt: |
      Classify each record (up to 50) as Low/Medium/High risk with 1–2 sentence rationale.
      Return valid JSON of the form:
      {{
        "classifications": [
          {{"index": int, "entity": str, "risk": "Low" | "Medium" | "High", "rationale": str}}
        ]
      }}
      Data:
      {dataset_first_50}
    params:
      temperature: 0.3
      max_tokens: 4096
      top_p: 0.95
      force_json: true
"""

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def ensure_session_state():
    """Initialize all session state variables"""
    defaults = {
        "dataset_df": pd.DataFrame(DEFAULT_SAMPLE_JSON),
        "agents_cfg": load_or_default_agents(),
        "results": [],
        "keys": {
            "openai": os.getenv(PROVIDERS["openai"]["env_key"]),
            "gemini": os.getenv(PROVIDERS["gemini"]["env_key"]) or os.getenv("GEMINI_API_KEY"),
            "grok": os.getenv(PROVIDERS["grok"]["env_key"]),
        },
        "connected": {"openai": False, "gemini": False, "grok": False},
        "selected_agents": [],
        "run_history": [],
        "current_theme": "Ocean Blue",
        "dataset_history": [],
        "auto_save": True,
        "show_advanced": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_or_default_agents() -> Dict[str, Any]:
    """Load agents from agents.yaml or return defaults"""
    path = "agents.yaml"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "agents" in data:
                return data
        except Exception as e:
            st.warning(f"Failed to load agents.yaml: {e}")
    return yaml.safe_load(DEFAULT_AGENTS_YAML)

def save_agents_cfg(cfg: Dict[str, Any]):
    """Save agents configuration to file"""
    try:
        with open("agents.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        st.toast("✅ Saved agents.yaml", icon="✅")
    except Exception as e:
        st.toast(f"⚠️ Could not save: {e}", icon="⚠️")

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

def parse_text_as_records(text: str) -> pd.DataFrame:
    """Parse various text formats into DataFrame"""
    text = text.strip()
    
    # Try JSON array
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return pd.json_normalize(obj)
        elif isinstance(obj, dict):
            return pd.json_normalize([obj])
    except:
        pass
    
    # Try JSON Lines
    try:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if rows:
            return pd.json_normalize(rows)
    except:
        pass
    
    # Fallback: newline-separated text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return pd.DataFrame({"text": lines})
    
    return pd.DataFrame()

def load_file_to_df(uploaded_file) -> pd.DataFrame:
    """Load uploaded file into DataFrame with smart detection"""
    name = uploaded_file.name.lower()
    content = uploaded_file.read()
    
    try:
        if name.endswith(".csv"):
            # Try comma first, then semicolon
            try:
                return pd.read_csv(io.BytesIO(content))
            except:
                return pd.read_csv(io.BytesIO(content), sep=";")
        elif name.endswith(".tsv"):
            return pd.read_csv(io.BytesIO(content), sep="\t")
        elif name.endswith((".json", ".jsonl")):
            text = content.decode("utf-8")
            try:
                obj = json.loads(text)
                if isinstance(obj, list):
                    return pd.json_normalize(obj)
                elif isinstance(obj, dict):
                    return pd.json_normalize([obj])
            except:
                # Try JSONL
                lines = [json.loads(l) for l in text.splitlines() if l.strip()]
                return pd.json_normalize(lines)
        else:
            # Generic text
            return parse_text_as_records(content.decode("utf-8"))
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# ============================================================================
# LLM PROVIDER INTEGRATIONS
# ============================================================================

def ensure_openai_client(api_key: str):
    """Initialize OpenAI client"""
    from openai import OpenAI
    return OpenAI(api_key=api_key)

def ensure_gemini_client(api_key: str):
    """Initialize Gemini client"""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai

def ensure_grok_client(api_key: str):
    """Initialize Grok client"""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

def call_openai(model: str, system_prompt: str, user_prompt: str, params: Dict[str, Any], api_key: str) -> str:
    """Call OpenAI API"""
    client = ensure_openai_client(api_key)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_prompt or ""},
        ],
        "temperature": params.get("temperature", 0.2),
        "max_tokens": params.get("max_tokens", 1024),
        "top_p": params.get("top_p", 1.0),
    }
    
    if params.get("force_json"):
        kwargs["response_format"] = {"type": "json_object"}
    
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def call_gemini(model: str, system_prompt: str, user_prompt: str, params: Dict[str, Any], api_key: str) -> str:
    """Call Gemini API"""
    genai = ensure_gemini_client(api_key)
    gen_config = {
        "temperature": params.get("temperature", 0.2),
        "top_p": params.get("top_p", 0.95),
        "max_output_tokens": params.get("max_tokens", 1024),
    }
    
    if params.get("force_json"):
        gen_config["response_mime_type"] = "application/json"
    
    try:
        gmodel = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt or None
        )
        resp = gmodel.generate_content(user_prompt or "", generation_config=gen_config)
        return resp.text or ""
    except Exception as e:
        # Fallback for models without system_instruction
        gmodel = genai.GenerativeModel(model_name=model)
        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        resp = gmodel.generate_content(combined, generation_config=gen_config)
        return resp.text or ""

def call_grok(model: str, system_prompt: str, user_prompt: str, params: Dict[str, Any], api_key: str) -> str:
    """Call Grok API (via OpenAI-compatible endpoint)"""
    client = ensure_grok_client(api_key)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or "You are Grok, a helpful AI assistant."},
            {"role": "user", "content": user_prompt or ""},
        ],
        "temperature": params.get("temperature", 0.2),
        "max_tokens": params.get("max_tokens", 1024),
    }
    
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str, 
             params: Dict[str, Any], api_key: str) -> Tuple[bool, str]:
    """Unified LLM calling interface"""
    try:
        if provider == "openai":
            out = call_openai(model, system_prompt, user_prompt, params, api_key)
        elif provider == "gemini":
            out = call_gemini(model, system_prompt, user_prompt, params, api_key)
        elif provider == "grok":
            out = call_grok(model, system_prompt, user_prompt, params, api_key)
        else:
            return False, f"Unknown provider: {provider}"
        return True, out
    except Exception as e:
        return False, f"{provider} error: {str(e)}"

# ============================================================================
# PROMPT TEMPLATING
# ============================================================================

def build_user_prompt(template: str, df: pd.DataFrame) -> str:
    """Build user prompt with dataset templating"""
    def to_json(df_in: pd.DataFrame):
        try:
            return json.dumps(df_in.to_dict(orient="records"), ensure_ascii=False, indent=2)
        except:
            return df_in.to_json(orient="records")
    
    mapping = {
        "dataset_preview": to_json(df.head(10)),
        "dataset_sample_200": to_json(df.head(200)),
        "dataset_first_50": to_json(df.head(50)),
        "full_dataset_json": to_json(df),
        "columns": json.dumps(list(df.columns), ensure_ascii=False),
        "row_count": str(len(df)),
    }
    
    try:
        class SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        return template.format_map(SafeDict(mapping))
    except:
        return template

# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def apply_custom_theme(theme_name: str):
    """Apply custom theme styling"""
    theme = THEMES.get(theme_name, THEMES["Ocean Blue"])
    
    custom_css = f"""
    <style>
        :root {{
            --primary-color: {theme['primaryColor']};
            --background-color: {theme['backgroundColor']};
            --secondary-bg: {theme['secondaryBackgroundColor']};
            --text-color: {theme['textColor']};
        }}
        
        .stApp {{
            background-color: {theme['backgroundColor']};
            color: {theme['textColor']};
        }}
        
        .stButton>button {{
            background-color: {theme['primaryColor']};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            opacity: 0.8;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .metric-card {{
            background: {theme['secondaryBackgroundColor']};
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid {theme['primaryColor']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        
        .success-badge {{
            background: #4CAF50;
            color: white;
        }}
        
        .error-badge {{
            background: #F44336;
            color: white;
        }}
        
        .info-badge {{
            background: {theme['primaryColor']};
            color: white;
        }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def render_header():
    """Render app header with theme selector"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(APP_TITLE)
        st.caption(APP_DESC)
    
    with col2:
        theme_name = st.selectbox(
            "🎨 Theme",
            options=list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.current_theme),
            key="theme_selector"
        )
        if theme_name != st.session_state.current_theme:
            st.session_state.current_theme = theme_name
            st.rerun()
    
    apply_custom_theme(st.session_state.current_theme)

def provider_badge(provider: str, connected: bool) -> str:
    """Generate provider status badge"""
    icon = "🟢" if connected else "🔴"
    return f"{icon} **{provider.upper()}**"

def render_sidebar():
    """Render sidebar with settings and quick actions"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Quick Stats
        st.subheader("📊 Quick Stats")
        df = st.session_state.dataset_df
        st.metric("Dataset Rows", len(df))
        st.metric("Columns", len(df.columns))
        st.metric("Total Runs", len(st.session_state.run_history))
        
        st.divider()
        
        # Auto-save toggle
        st.session_state.auto_save = st.checkbox(
            "💾 Auto-save results",
            value=st.session_state.auto_save
        )
        
        # Advanced features
        st.session_state.show_advanced = st.checkbox(
            "🔧 Show advanced options",
            value=st.session_state.show_advanced
        )
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Reload Agents", use_container_width=True):
            st.session_state.agents_cfg = load_or_default_agents()
            st.toast("Agents reloaded!", icon="✅")
        
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.results = []
            st.toast("Results cleared!", icon="🗑️")
        
        if st.button("📋 Reset Dataset", use_container_width=True):
            st.session_state.dataset_df = pd.DataFrame(DEFAULT_SAMPLE_JSON)
            st.toast("Dataset reset to default!", icon="📋")
        
        st.divider()
        
        # Export options
        st.subheader("📦 Export")
        if st.session_state.results:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "theme": st.session_state.current_theme,
                "results": st.session_state.results,
                "dataset_rows": len(st.session_state.dataset_df)
            }
            st.download_button(
                "⬇️ Download Results",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"results_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )

def render_provider_keys():
    """Render API key configuration"""
    st.subheader("🔐 API Connections")
    
    cols = st.columns(3)
    
    for idx, (provider, config) in enumerate(PROVIDERS.items()):
        with cols[idx]:
            # FIX: Use dictionary-style access for session_state to avoid attribute errors.
            env_present = bool(st.session_state['keys'].get(provider))
            st.markdown(provider_badge(provider, env_present))
            
            if not env_present:
                key = st.text_input(
                    f"{provider.upper()} API Key",
                    type="password",
                    key=f"{provider}_key_input",
                    help=f"Enter your {provider.upper()} API key"
                )
                if key:
                    # FIX: Use dictionary-style access to update the nested dictionary.
                    st.session_state['keys'][provider] = key
                    st.session_state.connected[provider] = True
                    st.toast(f"✅ {provider.upper()} connected!", icon="🔐")
                    st.rerun()
            else:
                st.session_state.connected[provider] = True
                st.success(f"✓ Connected via environment", icon="🔒")

def render_dataset_tab():
    """Render dataset management tab"""
    st.subheader("📊 Dataset Management")
    
    # Upload section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Upload File")
        uploaded = st.file_uploader(
            "Choose a file",
            type=["txt", "csv", "tsv", "json", "jsonl"],
            help="Upload your dataset in various formats"
        )
        
        if uploaded:
            with st.spinner("Loading file..."):
                df = load_file_to_df(uploaded)
                if df is not None and not df.empty:
                    st.session_state.dataset_df = df.reset_index(drop=True)
                    st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                else:
                    st.warning("Empty or invalid file")
    
    with col2:
        st.markdown("#### Paste Data")
        raw_data = st.text_area(
            "Paste JSON, JSONL, or text",
            height=150,
            placeholder='[{"name": "item1"}, {"name": "item2"}]'
        )
        
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("📋 Parse Data", use_container_width=True):
                if raw_data.strip():
                    df = parse_text_as_records(raw_data)
                    if not df.empty:
                        st.session_state.dataset_df = df.reset_index(drop=True)
                        st.success(f"✅ Parsed {len(df)} rows")
                    else:
                        st.warning("Could not parse data")
        
        with col2b:
            if st.button("📝 Load Sample", use_container_width=True):
                st.session_state.dataset_df = pd.DataFrame(DEFAULT_SAMPLE_JSON)
                st.success("✅ Sample data loaded")
    
    st.divider()
    
    # Dataset preview
    df = st.session_state.dataset_df
    st.markdown("#### Dataset Preview (First 10 rows)")
    st.dataframe(df.head(10), use_container_width=True, height=300)
    
    # Data editor
    with st.expander("✏️ Edit Dataset", expanded=False):
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="dataset_editor"
        )
        
        if st.button("💾 Apply Changes"):
            st.session_state.dataset_df = edited_df.reset_index(drop=True)
            st.success("✅ Changes applied!")
            st.rerun()
    
    # Transformations
    with st.expander("🔧 Data Transformations"):
        transform_col1, transform_col2 = st.columns(2)
        
        with transform_col1:
            transform = st.selectbox(
                "Select transformation",
                [
                    "None",
                    "Drop duplicate rows",
                    "Fill NaN with empty string",
                    "Remove empty rows",
                    "Lowercase all text columns",
                    "Trim whitespace",
                    "Remove special characters"
                ]
            )
        
        with transform_col2:
            if st.button("▶️ Apply Transform", use_container_width=True):
                df_copy = st.session_state.dataset_df.copy()
                
                if transform == "Drop duplicate rows":
                    df_copy = df_copy.drop_duplicates().reset_index(drop=True)
                    st.success(f"Removed {len(df) - len(df_copy)} duplicates")
                
                elif transform == "Fill NaN with empty string":
                    df_copy = df_copy.fillna("")
                    st.success("Filled NaN values")
                
                elif transform == "Remove empty rows":
                    df_copy = df_copy.dropna(how='all').reset_index(drop=True)
                    st.success(f"Removed {len(df) - len(df_copy)} empty rows")
                
                elif transform == "Lowercase all text columns":
                    for col in df_copy.select_dtypes(include=['object']).columns:
                        df_copy[col] = df_copy[col].astype(str).str.lower()
                    st.success("Converted text to lowercase")
                
                elif transform == "Trim whitespace":
                    for col in df_copy.select_dtypes(include=['object']).columns:
                        df_copy[col] = df_copy[col].astype(str).str.strip()
                    st.success("Trimmed whitespace")
                
                elif transform == "Remove special characters":
                    import re
                    for col in df_copy.select_dtypes(include=['object']).columns:
                        df_copy[col] = df_copy[col].astype(str).apply(
                            lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x)
                        )
                    st.success("Removed special characters")
                
                st.session_state.dataset_df = df_copy

def render_agents_tab():
    """Render agents configuration tab"""
    st.subheader("🤖 Agent Configuration")
    
    # YAML editor
    with st.expander("📝 Edit agents.yaml", expanded=False):
        yaml_text = st.text_area(
            "YAML Configuration",
            value=yaml.safe_dump(st.session_state.agents_cfg, sort_keys=False, allow_unicode=True),
            height=400,
            key="yaml_editor"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Reload from Text", use_container_width=True):
                try:
                    new_cfg = yaml.safe_load(yaml_text)
                    if isinstance(new_cfg, dict) and "agents" in new_cfg:
                        st.session_state.agents_cfg = new_cfg
                        st.success("✅ Configuration reloaded")
                    else:
                        st.error("❌ Invalid YAML structure")
                except Exception as e:
                    st.error(f"❌ Parse error: {e}")
        
        with col2:
            if st.button("💾 Save to File", use_container_width=True):
                save_agents_cfg(st.session_state.agents_cfg)
        
        with col3:
            if st.button("↩️ Reset to Default", use_container_width=True):
                st.session_state.agents_cfg = yaml.safe_load(DEFAULT_AGENTS_YAML)
                st.success("✅ Reset to defaults")
                st.rerun()
    
    st.divider()
    
    # Agent selection
    agents = st.session_state.agents_cfg.get("agents", [])
    
    if not agents:
        st.warning("⚠️ No agents configured. Please add agents in agents.yaml")
        return
    
    st.markdown("#### Select and Configure Agents")
    
    agent_names = [a.get("name", f"Agent {i}") for i, a in enumerate(agents)]
    selected_names = st.multiselect(
        "Choose agents to run",
        agent_names,
        default=[agent_names[0]] if agent_names else [],
        help="Select one or more agents to execute"
    )
    
    if not selected_names:
        st.info("👆 Select at least one agent to configure")
        return
    
    # Per-agent configuration
    overridden_agents = []
    
    for agent in agents:
        if agent.get("name") not in selected_names:
            continue
        
        with st.expander(f"⚙️ {agent['name']}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {agent.get('description', 'No description')}")
            
            with col2:
                st.markdown(f"**Default Provider:** `{agent.get('provider', 'N/A')}`")
            
            # Provider and model selection
            cfg_col1, cfg_col2 = st.columns(2)
            
            with cfg_col1:
                provider = st.selectbox(
                    "Provider",
                    list(PROVIDERS.keys()),
                    index=list(PROVIDERS.keys()).index(agent.get("provider", "openai")),
                    key=f"provider_{agent['name']}"
                )
            
            with cfg_col2:
                model_list = PROVIDERS[provider]["models"]
                default_model = agent.get("model", model_list[0])
                model_index = model_list.index(default_model) if default_model in model_list else 0
                
                model = st.selectbox(
                    "Model",
                    model_list,
                    index=model_index,
                    key=f"model_{agent['name']}"
                )
            
            # Prompts
            system_prompt = st.text_area(
                "System Prompt",
                value=agent.get("system_prompt", ""),
                height=100,
                key=f"sys_{agent['name']}"
            )
            
            user_prompt = st.text_area(
                "User Prompt Template",
                value=agent.get("user_prompt", ""),
                height=150,
                key=f"user_{agent['name']}",
                help="Use {dataset_preview}, {dataset_sample_200}, {dataset_first_50}, {full_dataset_json}"
            )
            
            # Parameters
            params = agent.get("params", {})
            
            if st.session_state.show_advanced:
                st.markdown("**Advanced Parameters**")
                param_cols = st.columns(4)
                
                with param_cols[0]:
                    temperature = st.slider(
                        "Temperature",
                        0.0, 2.0, float(params.get("temperature", 0.2)),
                        0.05,
                        key=f"temp_{agent['name']}"
                    )
                
                with param_cols[1]:
                    top_p = st.slider(
                        "Top P",
                        0.0, 1.0, float(params.get("top_p", 1.0)),
                        0.05,
                        key=f"topp_{agent['name']}"
                    )
                
                with param_cols[2]:
                    max_tokens = st.number_input(
                        "Max Tokens",
                        16, 32768, int(params.get("max_tokens", 1024)),
                        step=128,
                        key=f"maxtok_{agent['name']}"
                    )
                
                with param_cols[3]:
                    force_json = st.checkbox(
                        "Force JSON",
                        value=bool(params.get("force_json", False)),
                        key=f"json_{agent['name']}"
                    )
            else:
                temperature = float(params.get("temperature", 0.2))
                top_p = float(params.get("top_p", 1.0))
                max_tokens = int(params.get("max_tokens", 1024))
                force_json = bool(params.get("force_json", False))
            
            overridden_agents.append({
                "name": agent.get("name"),
                "description": agent.get("description", ""),
                "provider": provider,
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "params": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "force_json": force_json
                }
            })
    
    st.session_state.selected_agents = overridden_agents

def render_run_tab():
    """Render agent execution tab"""
    st.subheader("🚀 Execute Agents")
    
    selected_agents = st.session_state.get("selected_agents", [])
    df = st.session_state.dataset_df
    
    if not selected_agents:
        st.warning("⚠️ No agents selected. Go to the Agents tab to configure.")
        return
    
    # Execution controls
    control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
    
    with control_col1:
        st.markdown(f"**Ready to execute {len(selected_agents)} agent(s)**")
        agent_list = ", ".join([a["name"] for a in selected_agents])
        st.caption(f"Agents: {agent_list}")
    
    with control_col2:
        max_agents = st.number_input(
            "Max agents to run",
            1, len(selected_agents),
            len(selected_agents),
            help="Limit execution"
        )
    
    with control_col3:
        run_button = st.button(
            "▶️ Execute Now",
            type="primary",
            use_container_width=True
        )
    
    # Connection status
    st.markdown("#### Connection Status")
    status_cols = st.columns(3)
    
    for idx, (provider, connected) in enumerate(st.session_state.connected.items()):
        with status_cols[idx]:
            if connected:
                st.success(f"✅ {provider.upper()}", icon="🟢")
            else:
                st.error(f"❌ {provider.upper()}", icon="🔴")
    
    st.divider()
    
    # Execution logic
    if run_button:
        st.session_state.results = []
        to_run = selected_agents[:int(max_agents)]
        total = len(to_run)
        
        progress_bar = st.progress(0, text="Initializing...")
        
        with st.status("🔄 Running agents...", expanded=True) as status:
            for i, agent in enumerate(to_run, start=1):
                provider = agent["provider"]
                model = agent["model"]
                
                # Check API key
                # FIX: Use dictionary-style access for robustness.
                api_key = st.session_state['keys'].get(provider)
                if not api_key:
                    st.error(f"❌ {agent['name']}: Missing {provider.upper()} API key")
                    st.session_state.results.append({
                        "agent": agent["name"],
                        "provider": provider,
                        "model": model,
                        "ok": False,
                        "error": f"Missing API key for {provider}",
                        "output": "",
                        "timestamp": datetime.now().isoformat()
                    })
                    progress_bar.progress(int(100 * i / total))
                    continue
                
                # Build prompt
                user_prompt = build_user_prompt(agent["user_prompt"], df)
                
                st.write(f"⏳ Executing **{agent['name']}** ({provider}/{model})...")
                start_time = time.time()
                
                # Call LLM
                ok, output = call_llm(
                    provider,
                    model,
                    agent["system_prompt"],
                    user_prompt,
                    agent["params"],
                    api_key
                )
                
                elapsed = time.time() - start_time
                
                if ok:
                    st.success(f"✅ {agent['name']} completed in {elapsed:.2f}s")
                    
                    # Try to parse JSON
                    parsed = None
                    if agent["params"].get("force_json"):
                        try:
                            parsed = json.loads(output)
                        except:
                            st.warning(f"⚠️ Could not parse JSON output from {agent['name']}")
                    
                    st.session_state.results.append({
                        "agent": agent["name"],
                        "provider": provider,
                        "model": model,
                        "ok": True,
                        "output": output,
                        "parsed": parsed,
                        "elapsed": elapsed,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    st.error(f"❌ {agent['name']} failed: {output}")
                    st.session_state.results.append({
                        "agent": agent["name"],
                        "provider": provider,
                        "model": model,
                        "ok": False,
                        "error": output,
                        "output": "",
                        "elapsed": elapsed,
                        "timestamp": datetime.now().isoformat()
                    })
                
                progress_bar.progress(int(100 * i / total))
            
            status.update(label="✅ Execution complete!", state="complete")
        
        # Add to history
        st.session_state.run_history.append({
            "timestamp": datetime.now().isoformat(),
            "agents_count": len(to_run),
            "success_count": sum(1 for r in st.session_state.results if r["ok"]),
            "results": st.session_state.results
        })
        
        # Auto-save
        if st.session_state.auto_save:
            st.toast("💾 Results auto-saved", icon="💾")
    
    # Results display
    if st.session_state.results:
        st.divider()
        render_results()

def render_results():
    """Render execution results"""
    st.markdown("### 📊 Results Dashboard")
    
    results = st.session_state.results
    success_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - success_count
    total_time = sum(r.get("elapsed", 0) for r in results)
    
    # Summary metrics
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Agents", len(results))
    with metric_cols[1]:
        st.metric("Successful", success_count, delta=None)
    with metric_cols[2]:
        st.metric("Failed", fail_count, delta=None if fail_count == 0 else -fail_count)
    with metric_cols[3]:
        st.metric("Total Time", f"{total_time:.2f}s")
    
    st.divider()
    
    # Individual results
    for idx, result in enumerate(results):
        status_icon = "✅" if result["ok"] else "❌"
        
        with st.expander(
            f"{status_icon} {result['agent']} [{result['provider']}/{result['model']}] "
            f"({result.get('elapsed', 0):.2f}s)",
            expanded=False
        ):
            if result["ok"]:
                # Display parsed JSON if available
                if result.get("parsed"):
                    st.json(result["parsed"])
                else:
                    st.text_area(
                        "Output",
                        value=result["output"],
                        height=300,
                        key=f"output_{idx}"
                    )
                
                # Copy button
                st.code(result["output"], language="text")
            else:
                st.error(f"**Error:** {result.get('error', 'Unknown error')}")
    
    # Export options
    st.divider()
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "theme": st.session_state.current_theme,
            "dataset_info": {
                "rows": len(st.session_state.dataset_df),
                "columns": len(st.session_state.dataset_df.columns)
            },
            "results": results
        }
        
        st.download_button(
            "📥 Download Results (JSON)",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name=f"agent_results_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with export_col2:
        # Create CSV from results
        results_df = pd.DataFrame([
            {
                "Agent": r["agent"],
                "Provider": r["provider"],
                "Model": r["model"],
                "Status": "Success" if r["ok"] else "Failed",
                "Time (s)": r.get("elapsed", 0),
                "Timestamp": r.get("timestamp", "")
            }
            for r in results
        ])
        
        st.download_button(
            "📊 Download Summary (CSV)",
            data=results_df.to_csv(index=False),
            file_name=f"results_summary_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )

def render_insights_tab():
    """Render dataset insights and analytics"""
    st.subheader("📈 Dataset Insights")
    
    df = st.session_state.dataset_df
    
    if df.empty:
        st.info("📊 Load a dataset to see insights")
        return
    
    # Overview metrics
    st.markdown("#### Overview")
    overview_cols = st.columns(5)
    
    with overview_cols[0]:
        st.metric("Total Rows", len(df))
    with overview_cols[1]:
        st.metric("Columns", len(df.columns))
    with overview_cols[2]:
        st.metric("Empty Cells", int(df.isna().sum().sum()))
    with overview_cols[3]:
        st.metric("Duplicates", int(len(df) - len(df.drop_duplicates())))
    with overview_cols[4]:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_mb:.2f} MB")
    
    st.divider()
    
    # Column analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Column Types")
        dtype_counts = df.dtypes.value_counts()
        st.bar_chart(dtype_counts)
    
    with col2:
        st.markdown("#### Missing Data")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.bar_chart(missing)
        else:
            st.success("✅ No missing data!")
    
    st.divider()
    
    # Distribution analysis
    st.markdown("#### Value Distributions")
    
    text_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
    if text_cols:
        selected_col = st.selectbox("Select column", text_cols)
        
        value_counts = df[selected_col].astype(str).value_counts().head(20)
        
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            st.bar_chart(value_counts)
        
        with chart_col2:
            st.markdown("**Top Values**")
            for val, count in value_counts.items():
                st.write(f"• `{val}`: {count}")
    else:
        st.info("No text columns to analyze")
    
    # Sample records
    with st.expander("🔍 Sample Records", expanded=False):
        sample_size = st.slider("Sample size", 1, min(100, len(df)), 10)
        st.dataframe(df.sample(min(sample_size, len(df))), use_container_width=True)

def render_history_tab():
    """Render execution history"""
    st.subheader("📜 Execution History")
    
    history = st.session_state.run_history
    
    if not history:
        st.info("No execution history yet. Run some agents to see history here.")
        return
    
    st.markdown(f"**Total Runs:** {len(history)}")
    
    for idx, run in enumerate(reversed(history), start=1):
        with st.expander(
            f"Run #{len(history) - idx + 1} - {run['timestamp']} "
            f"({run['success_count']}/{run['agents_count']} successful)",
            expanded=(idx == 1)
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Agents Run", run['agents_count'])
            with col2:
                st.metric("Successful", run['success_count'])
            with col3:
                st.metric("Failed", run['agents_count'] - run['success_count'])
            
            # Show results
            for result in run['results']:
                status = "✅" if result['ok'] else "❌"
                st.write(f"{status} **{result['agent']}** - {result['provider']}/{result['model']}")

def render_about_tab():
    """Render about and help tab"""
    st.subheader("ℹ️ About")
    
    st.markdown("""
    ### 🚀 Agentic Dataset Workbench Pro
    
    A powerful multi-agent, multi-provider dataset processing system with beautiful themes and advanced features.
    
    #### ✨ Features
    
    - **20 Beautiful Themes**: Choose from 20 carefully crafted color themes
    - **Multi-Provider Support**: OpenAI, Gemini, and Grok integration
    - **Flexible Dataset Loading**: Upload CSV, JSON, JSONL, or paste data
    - **Interactive Editing**: Edit your dataset with dynamic rows
    - **Agent Configuration**: Customize agents with YAML or UI
    - **Real-time Execution**: Live progress tracking and status updates
    - **Advanced Analytics**: Dataset insights and visualizations
    - **Export Options**: Download results in JSON or CSV format
    - **Execution History**: Track all your agent runs
    - **Auto-save**: Automatic result saving
    
    #### 🔧 Prompt Templating
    
    Use these placeholders in your agent prompts:
    
    - `{dataset_preview}` - First 10 rows
    - `{dataset_sample_200}` - First 200 rows
    - `{dataset_first_50}` - First 50 rows
    - `{full_dataset_json}` - Complete dataset
    - `{columns}` - Column names
    - `{row_count}` - Number of rows
    
    #### 🔑 API Keys
    
    API keys can be provided in two ways:
    1. **Environment Variables** (recommended for Hugging Face Spaces)
    2. **UI Input** (session-only, secure)
    
    #### 📚 Tips
    
    - Use `force_json` parameter for structured outputs
    - Enable advanced options for fine-grained control
    - Save your agents.yaml for reusability
    - Export results regularly
    - Use transformations to clean data before processing
    
    #### 🐛 Bug Fixes & Improvements
    
    - Fixed file upload parsing for multiple formats
    - Improved error handling for API calls
    - Enhanced theme application and styling
    - Better memory management for large datasets
    - Robust JSON parsing with fallbacks
    - Secure API key handling
    """)
    
    st.divider()
    
    st.markdown("""
    ### 🤝 Support
    
    For issues or questions:
    - Check the execution logs in the Run tab
    - Verify API keys are correctly set
    - Ensure dataset format is compatible
    - Review agent configuration in agents.yaml
    """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""
    ensure_session_state()
    render_header()
    render_sidebar()
    
    # Create tabs
    tabs = st.tabs([
        "📊 Dataset",
        "🤖 Agents",
        "🚀 Run",
        "📈 Insights",
        "📜 History",
        "ℹ️ About"
    ])
    
    with tabs[0]:
        render_provider_keys()
        st.divider()
        render_dataset_tab()
    
    with tabs[1]:
        render_agents_tab()
    
    with tabs[2]:
        render_run_tab()
    
    with tabs[3]:
        render_insights_tab()
    
    with tabs[4]:
        render_history_tab()
    
    with tabs[5]:
        render_about_tab()

if __name__ == "__main__":
    main()
