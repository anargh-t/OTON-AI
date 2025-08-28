

import os
import json
import streamlit as st
import torch
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    from transformers.models.auto import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

APP_TITLE = "Tata Group-Level LLM – Strategic Intelligence Platform"
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = "group_level_outputs"

# Strategic task templates for CXOs with demo-ready business contexts
STRATEGIC_TASKS = {
    "Business Intelligence Synthesis": {
        "instruction": "Synthesize a business intelligence report combining internal and external data.",
        "business_context": "Internal data: TCS Q3 revenue growth is 12% | External data: Global IT services market growing at 8% annually | Market context: AI adoption in enterprises increased by 35% this year | Competition: Infosys focusing on AI, Wipro on cloud | Regulatory: New data privacy laws effective Q2 2024"
    },
    "Strategic Scenario Analysis": {
        "instruction": "Provide strategic scenario analysis for business planning.",
        "business_context": "Scenario: Global inflation rises by 5% over the next 12 months | Companies affected: Titan, Tata Consumer Products, Tata Motors | Impact areas: Raw material costs, consumer spending, interest rates | Current market position: Strong domestic demand, export challenges | Mitigation strategies needed: Cost optimization, pricing models, supply chain diversification"
    },
    "Cross-Business Synergy": {
        "instruction": "Identify cross-business leverage opportunities between Tata companies.",
        "business_context": "Business Unit 1: Tata Motors launching new EV model 'Nexon EV Pro' in Q2 2024 | Business Unit 2: Tata Power expanding EV charging infrastructure to 5000+ stations | Business Unit 3: Tata Chemicals investing in green hydrogen production | Market opportunity: India's EV adoption growing 150% YoY | Synergy potential: Integrated EV ecosystem, joint marketing, shared R&D"
    },
    "Market Expansion Analysis": {
        "instruction": "Analyze market expansion opportunities for Tata Group.",
        "business_context": "Market Trends: Green energy sector in Europe growing at 20% annually | Regulatory Environment: EU Green Deal mandates 55% CO2 reduction by 2030 | Tata Companies: Tata Chemicals, Tata Power, Tata Motors | Current Capabilities: Green hydrogen, EV technology, sustainable materials | Market size: €500 billion green energy market by 2030 | Competition: Reliance, Adani, international energy majors"
    },
    "Risk Assessment & Mitigation": {
        "instruction": "Generate strategic risk assessment and mitigation plan.",
        "business_context": "Risk Scenario: Supply chain disruption due to geopolitical tensions in Asia-Pacific | Impact Areas: Raw materials, manufacturing, logistics, customer delivery | Tata Companies Affected: Tata Steel, Tata Motors, TCS | Current Risk Exposure: High dependency on Asian suppliers (70% of materials) | Mitigation needed: Supplier diversification, inventory management, alternative sourcing | Timeline: 6-12 months implementation"
    },
    "Boardroom Recommendation": {
        "instruction": "Provide boardroom-ready strategic recommendation.",
        "business_context": "Business Context: Tata Group needs to increase market share in premium consumer goods segment | Current Position: 15% market share, behind HUL and ITC | Available Resources: ₹25,000 Cr investment capacity, strong brand portfolio | Strategic Objective: Achieve 25% market share within 3 years | Market opportunity: Premium segment growing 18% annually | Competition: HUL launching premium brands, ITC expanding FMCG"
    },
    "Competitive Analysis": {
        "instruction": "Analyze competitive landscape and strategic positioning.",
        "business_context": "Competitor Analysis: Reliance investing ₹75,000 Cr in green energy, Adani expanding into ports and logistics | Market Context: India's infrastructure sector growing 12% annually | Tata Group Position: Strong in steel, automotive, IT services | Strategic Focus: Sustainable mobility, digital transformation, global expansion | Competitive advantage: Integrated ecosystem, strong R&D, global presence | Threat level: High from Reliance, Medium from Adani"
    },
    "Innovation Roadmap": {
        "instruction": "Develop cross-business innovation roadmap.",
        "business_context": "Innovation Focus: AI-powered smart city solutions integrating multiple Tata businesses | Companies Involved: TCS (AI/ML), Tata Power (smart grid), Tata Motors (smart mobility), Tata Communications (IoT) | Market Opportunity: Global smart city market $2.5 trillion by 2025 | Current Capabilities: Individual AI solutions, smart grid technology, EV ecosystem | Innovation gaps: Integrated platform, data sharing protocols, cross-business APIs | Timeline: 18-24 months to market"
    },
    "Financial Analysis": {
        "instruction": "Generate strategic financial analysis and investment recommendation.",
        "business_context": "Financial Context: Tata Group has ₹50,000 Cr available for strategic investments | Investment Criteria: Minimum 15% ROI, risk tolerance: Medium, timeline: 3-5 years | Priority Sectors: Green energy, digital transformation, healthcare technology | Market Conditions: Interest rates stable, inflation moderating, strong domestic growth | Investment options: Internal R&D (₹15,000 Cr), acquisitions (₹25,000 Cr), partnerships (₹10,000 Cr) | Expected returns: R&D (20-25%), Acquisitions (18-22%), Partnerships (15-18%)"
    },
    "Talent Strategy": {
        "instruction": "Create strategic talent acquisition and development plan.",
        "business_context": "Business Context: Tata Group needs 10,000+ AI/ML engineers over next 3 years | Current Challenge: AI talent shortage, 40% salary premium, high attrition rates | Strategic Priority: Build world-class AI talent pool, reduce dependency on external hiring | Available Resources: ₹2,000 Cr for talent initiatives, 50+ global locations | Market reality: AI engineers in high demand globally, competition from tech giants | Solution needed: Internal training programs, university partnerships, competitive compensation"
    }
}

def build_strategic_prompt(instruction: str, business_context: str) -> str:
    """Build strategic prompt for enterprise intelligence tasks"""
    return f"### Strategic Task:\n{instruction}\n\n### Business Context:\n{business_context}\n\n### Strategic Analysis:\n"

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(base_model_name: str, adapter_dir: str, use_gpu: bool):
    """Load the strategic intelligence model and tokenizer"""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Prefer tokenizer from adapter directory if present
    tokenizer_src = adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model from hub
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    # If adapter exists, attach it
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        model = PeftModel.from_pretrained(model, adapter_dir)

    if device == "cuda":
        model = model.to("cuda")
    model.eval()
    return model, tokenizer, device

def extract_strategic_analysis(full_decoded: str) -> str:
    """Extract the strategic analysis from the full response"""
    marker = "### Strategic Analysis:\n"
    if marker in full_decoded:
        return full_decoded.split(marker, 1)[1].strip()
    return full_decoded.strip()

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("AI-powered strategic intelligence platform for Tata Group CXOs - providing unified enterprise insights, cross-business synergies, and strategic decision support")

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
                with st.spinner("🔍 Analyzing strategic scenario..."):
                    try:
                        # Load model
                        model, tokenizer, device = load_model_and_tokenizer(base_model_name, adapter_dir, use_gpu)
                        
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
                        
                        # Display results
                        st.success("✅ Strategic Analysis Generated!")
                        st.subheader("📋 Strategic Analysis")
                        st.write(strategic_analysis)
                        
                        # Show full prompt for transparency
                        with st.expander("🔍 View Full Prompt"):
                            st.code(prompt, language="text")
                            
                    except Exception as e:
                        st.error(f"❌ Error generating analysis: {str(e)}")
                        st.info("💡 Make sure the model and adapter are properly loaded")
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
    **Tata Group-Level LLM** | Strategic Intelligence Platform | 
    Powered by LoRA fine-tuning on enterprise intelligence tasks | 
    Designed for CXO-level strategic decision making
    """)

if __name__ == "__main__":
    main()
