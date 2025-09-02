

import os
import json
import streamlit as st
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    from transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

APP_TITLE = "OTON-AI: One Tata One Network - Strategic Intelligence Platform"
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = "group_level_outputs"

# Strategic task templates (limited to requested set)
STRATEGIC_TASKS = {
    "Cross-Business Synergy": {
        "instruction": "Identify cross-business leverage opportunities between Tata companies.",
        "business_context": "Business Unit 1: Tata Motors launching new EV model 'Nexon EV Pro' in Q2 2024 | Business Unit 2: Tata Power expanding EV charging infrastructure to 5000+ stations | Business Unit 3: Tata Chemicals investing in green hydrogen production | Market opportunity: India's EV adoption growing 150% YoY | Synergy potential: Integrated EV ecosystem, joint marketing, shared R&D"
    },
    "Market Expansion Analysis": {
        "instruction": "Analyze market expansion opportunities for Tata Group.",
        "business_context": "Market Trends: Green energy sector in Europe growing at 20% annually | Regulatory Environment: EU Green Deal mandates 55% CO2 reduction by 2030 | Tata Companies: Tata Chemicals, Tata Power, Tata Motors | Current Capabilities: Green hydrogen, EV technology, sustainable materials | Market size: €500 billion green energy market by 2030 | Competition: Reliance, Adani, international energy majors"
    },
    "Innovation Roadmap": {
        "instruction": "Develop cross-business innovation roadmap.",
        "business_context": "Innovation Focus: AI-powered smart city solutions integrating multiple Tata businesses | Companies Involved: TCS (AI/ML), Tata Power (smart grid), Tata Motors (smart mobility), Tata Communications (IoT) | Market Opportunity: Global smart city market $2.5 trillion by 2025 | Current Capabilities: Individual AI solutions, smart grid technology, EV ecosystem | Innovation gaps: Integrated platform, data sharing protocols, cross-business APIs | Timeline: 18-24 months to market"
    }
}

def build_strategic_prompt(instruction: str, business_context: str) -> str:
    """Build strategic prompt for enterprise intelligence tasks"""
    return f"### Strategic Task:\n{instruction}\n\n### Business Context:\n{business_context}\n\n### Strategic Analysis:\n"

def find_best_adapter_directory(base_dir: str) -> str:
    """Find the best adapter directory with valid files"""
    # Check if base directory has valid adapter files
    if os.path.exists(os.path.join(base_dir, "adapter_config.json")) and \
       os.path.exists(os.path.join(base_dir, "adapter_model.safetensors")):
        return base_dir
    
    # Check for checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    if checkpoint_dirs:
        # Sort by checkpoint number and return the latest
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0, reverse=True)
        latest_checkpoint = os.path.join(base_dir, checkpoint_dirs[0])
        if os.path.exists(os.path.join(latest_checkpoint, "adapter_config.json")) and \
           os.path.exists(os.path.join(latest_checkpoint, "adapter_model.safetensors")):
            return latest_checkpoint
    
    return base_dir

def find_best_tokenizer_source(base_dir: str, base_model_name: str) -> str:
    """Find the best tokenizer source, preferring local files over base model"""
    # Check for valid tokenizer files in the adapter directory
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    
    # First try the base directory
    if all(os.path.exists(os.path.join(base_dir, f)) for f in tokenizer_files):
        # Check if tokenizer.json is not empty
        tokenizer_json_path = os.path.join(base_dir, "tokenizer.json")
        if os.path.getsize(tokenizer_json_path) > 1000:  # More than 1KB
            return base_dir
    
    # Check checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0, reverse=True)
        for checkpoint_dir in checkpoint_dirs:
            checkpoint_path = os.path.join(base_dir, checkpoint_dir)
            if all(os.path.exists(os.path.join(checkpoint_path, f)) for f in tokenizer_files):
                tokenizer_json_path = os.path.join(checkpoint_path, "tokenizer.json")
                if os.path.getsize(tokenizer_json_path) > 1000:  # More than 1KB
                    return checkpoint_path
    
    # Fall back to base model
    return base_model_name

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(base_model_name: str, adapter_dir: str, use_gpu: bool, show_debug: bool = False):
    """Load the strategic intelligence model and tokenizer with improved error handling"""
    try:
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Find the best adapter directory
        best_adapter_dir = find_best_adapter_directory(adapter_dir)
        if show_debug:
            st.info(f"🔍 Using adapter directory: {best_adapter_dir}")
        
        # Find the best tokenizer source
        tokenizer_src = find_best_tokenizer_source(best_adapter_dir, base_model_name)
        if show_debug:
            st.info(f"🔍 Using tokenizer from: {tokenizer_src}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if show_debug:
                st.success("✅ Tokenizer loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ Tokenizer loading failed: {str(e)}")
            if show_debug:
                st.info("🔄 Falling back to base model tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        if show_debug:
            st.info("🔍 Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        if show_debug:
            st.success("✅ Base model loaded successfully")
        
        # Load adapter if available
        adapter_config_path = os.path.join(best_adapter_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                if show_debug:
                    st.info("🔍 Loading LoRA adapter...")
                model = PeftModel.from_pretrained(model, best_adapter_dir)
                if show_debug:
                    st.success("✅ LoRA adapter loaded successfully")
            except Exception as e:
                st.warning(f"⚠️ Adapter loading failed: {str(e)}")
                if show_debug:
                    st.info("🔄 Continuing with base model only")
        else:
            if show_debug:
                st.warning("⚠️ No adapter found, using base model only")
        
        # Move to device
        if device == "cuda":
            model = model.to("cuda")
        model.eval()
        
        if show_debug:
            st.success("🎯 Model ready for strategic analysis!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        raise e
        

def extract_strategic_analysis(full_decoded: str) -> str:
    """Extract the strategic analysis from the full response"""
    marker = "### Strategic Analysis:\n"
    if marker in full_decoded:
        return full_decoded.split(marker, 1)[1].strip()
    return full_decoded.strip()

def parse_structured_response(response: str) -> dict:
    """Parse the structured response into components"""
    components = {
        "strategic_recommendation": "",
        "key_insight": "",
        "recommendation": "",
        "actionable_items": []
    }
    
    # First, try to split by common section separators
    # Handle cases where sections are concatenated with 'n'
    if "nKey Insight:" in response:
        response = response.replace("nKey Insight:", "\nKey Insight:")
    if "nRecommendation:" in response:
        response = response.replace("nRecommendation:", "\nRecommendation:")
    if "nActionable Items:" in response:
        response = response.replace("nActionable Items:", "\nActionable Items:")
    
    # Handle numbered items that start with 'n'
    for i in range(1, 10):
        response = response.replace(f"n{i}.", f"\n{i}.")
    
    lines = response.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.startswith("Strategic Recommendation:"):
            components["strategic_recommendation"] = line.replace("Strategic Recommendation:", "").strip()
            current_section = "strategic_recommendation"
        elif line.startswith("Key Insight:"):
            components["key_insight"] = line.replace("Key Insight:", "").strip()
            current_section = "key_insight"
        elif line.startswith("Recommendation:"):
            components["recommendation"] = line.replace("Recommendation:", "").strip()
            current_section = "recommendation"
        elif line.startswith("Actionable Items:"):
            current_section = "actionable_items"
        elif current_section == "actionable_items" and line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            # Clean up actionable items - remove the number prefix
            item = line
            # Remove the number prefix (e.g., "1. " -> "")
            for i in range(1, 10):
                if item.startswith(f"{i}. "):
                    item = item[3:]  # Remove "X. "
                    break
            if item.strip():
                components["actionable_items"].append(item.strip())
        elif current_section and current_section != "actionable_items":
            # Continue building the current section
            if components[current_section]:
                components[current_section] += " " + line
            else:
                components[current_section] = line
    
    return components

def display_structured_response(components: dict):
    """Display the structured response with proper formatting"""
    if components["strategic_recommendation"]:
        st.subheader("🎯 Strategic Recommendation")
        st.success(components["strategic_recommendation"])
    
    if components["key_insight"]:
        st.subheader("💡 Key Insight")
        st.info(components["key_insight"])
    
    if components["recommendation"]:
        st.subheader("📋 Recommendation")
        st.write(components["recommendation"])
    
    if components["actionable_items"]:
        st.subheader("✅ Actionable Items")
        for i, item in enumerate(components["actionable_items"], 1):
            st.write(f"**{i}.** {item}")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("OTON-AI: One Tata One Network - AI-powered strategic intelligence platform for Tata Group CXOs - providing unified enterprise insights, cross-business synergies, and strategic decision support")

    # Initialize session state for form inputs
    if 'instruction' not in st.session_state:
        st.session_state.instruction = ""
    if 'business_context' not in st.session_state:
        st.session_state.business_context = ""

    # Sidebar configuration
    with st.sidebar:
        st.header("🎯 Model Configuration")
        base_model_name = st.text_input("Base Model", value=DEFAULT_BASE_MODEL)
        adapter_dir = st.text_input("Adapter Directory", value=DEFAULT_ADAPTER_DIR)
        use_gpu = st.checkbox("Use GPU if available", value=True)
        show_debug = st.toggle("Show debug logs", value=False, help="Display detailed loading banners and logs")
        show_extras = st.toggle("Show metrics & quick actions", value=False)

        st.divider()
        st.header("⚙️ Generation Parameters")
        max_new_tokens = st.slider("Max New Tokens", 100, 800, 300, step=50)
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1)
        top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.9, step=0.05)

        st.divider()
        st.header("📚 Quick Templates")
        st.caption("Click any template to auto-populate the form")
        
        # Create functional buttons for each template
        for task_name, task_info in STRATEGIC_TASKS.items():
            if st.button(f"📋 {task_name}", key=f"btn_{task_name}", use_container_width=True):
                st.session_state.instruction = task_info["instruction"]
                st.session_state.business_context = task_info["business_context"]
                st.success(f"✅ Loaded: {task_name}")
                st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🚀 Strategic Intelligence Query")
        
        # Input fields with session state
        instruction = st.text_area(
            "Strategic Task",
            value=st.session_state.instruction,
            height=100,
            placeholder="Describe the strategic task or question you need analyzed...",
            key="instruction_input"
        )
        
        business_context = st.text_area(
            "Business Context",
            value=st.session_state.business_context,
            height=150,
            placeholder="Provide relevant business context, data, and parameters...",
            key="business_context_input"
        )
        
        # Generate button
        if st.button("🎯 Generate Strategic Analysis", type="primary", use_container_width=True):
            if instruction and business_context:
                with st.spinner("🔍 Analyzing strategic scenario with OTON-AI..."):
                    try:
                        # Load model
                        model, tokenizer, device = load_model_and_tokenizer(base_model_name, adapter_dir, use_gpu, show_debug)
                        
                        # Generate response
                        prompt = build_strategic_prompt(instruction, business_context)
                        inputs = tokenizer(prompt, return_tensors="pt")
                        
                        if device == "cuda":
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                do_sample=True,
                                top_p=top_p,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        
                        # Decode response
                        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        strategic_analysis = extract_strategic_analysis(full_response)
                        
                        # Parse and display structured response
                        st.success("✅ Strategic Analysis Generated by OTON-AI!")
                        
                        # Try to parse as structured response first
                        components = parse_structured_response(strategic_analysis)
                        
                        # Check if we have structured components
                        if components["strategic_recommendation"] or components["key_insight"] or components["recommendation"] or components["actionable_items"]:
                            display_structured_response(components)
                        else:
                            # Fallback to original display if not structured
                            st.subheader("📋 Strategic Analysis")
                            st.write(strategic_analysis)
                        
                        # Show full prompt for transparency
                        with st.expander("🔍 View Full Prompt"):
                            st.code(prompt, language="text")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating analysis: {str(e)}")
                        st.info("💡 Troubleshooting tips:")
                        st.info("1. Check if the adapter directory contains valid files")
                        st.info("2. Ensure the base model is accessible")
                        st.info("3. Try clearing the model cache and reloading")
                        st.info("4. Check system memory and GPU availability")
            else:
                st.warning("⚠️ Please provide both strategic task and business context")

    with col2:
        st.header("📊 Strategic Insights")
        
        # Show currently loaded template info
        if st.session_state.instruction and st.session_state.business_context:
            st.subheader("📋 Current Template")
            st.info("✅ Template loaded and ready for analysis")
            
            # Quick preview of current template
            with st.expander("👀 Preview Current Template"):
                st.write("**Task:**", st.session_state.instruction[:100] + "..." if len(st.session_state.instruction) > 100 else st.session_state.instruction)
                st.write("**Context:**", st.session_state.business_context[:150] + "..." if len(st.session_state.business_context) > 150 else st.session_state.business_context)
        
        if show_extras:
            st.divider()
            
            # Key metrics display
            st.subheader("📈 Key Metrics")
            st.metric("Training Samples", "12", "Strategic Tasks")
            st.metric("Model Parameters", "1.1B", "Base Model")
            st.metric("LoRA Adapters", "32", "Rank")
            
            st.divider()
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            if st.button("🔄 Refresh Model", use_container_width=True):
                st.cache_resource.clear()
                st.success("Model cache cleared!")
            
            if st.button("📊 View Training Data", use_container_width=True):
                st.info("Training data: group_level_train.jsonl\nEvaluation data: group_level_eval.jsonl")
        
        # Clear form button
        if st.button("🗑️ Clear Form", use_container_width=True):
            st.session_state.instruction = ""
            st.session_state.business_context = ""
            st.rerun()

    # Footer
    st.divider()
    st.caption("""
    **OTON-AI: One Tata One Network** | Strategic Intelligence Platform | 
    Powered by LoRA fine-tuning on enterprise intelligence tasks | 
    Designed for CXO-level strategic decision making across the Tata ecosystem
    """)

if __name__ == "__main__":
    main()
