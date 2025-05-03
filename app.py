"""Main application"""

import pandas as pd
import streamlit as st

from utils import file_contents, load_models_info

# Model metadata with both realtime and batch pricing
MODELS = load_models_info("resources/models.csv")

# Sample text from Moby Dick
MOBY_DICK_PROMPT = file_contents("resources/prompt.txt")
MOBY_DICK_OUTPUT = file_contents("resources/completion.txt")


# Streamlit app layout
st.set_page_config(page_title="LLM Token & Cost Calculator", layout="centered")
st.title("🧮 LLM Token & Cost Calculator")
st.markdown(
    "Now updates in real-time! Compare **Realtime vs Batch** pricing for multiple models."
)

# Input/output areas
col1, col2 = st.columns(2)
with col1:
    input_prompt = st.text_area("📥 Input Prompt", value=MOBY_DICK_PROMPT, height=200)
    manual_input_tokens = st.number_input(
        "🔢 Manually specify Input Token Count (optional)", min_value=0, step=1, value=0
    )


with col2:
    output_text = st.text_area("📤 Model Output", value=MOBY_DICK_OUTPUT, height=200)
    manual_output_tokens = st.number_input(
        "🔢 Manually specify Output Token Count (optional)",
        min_value=0,
        step=1,
        value=0,
    )

# Number of inputs for batch calculation
num_inputs = st.number_input(
    "📦 Number of Inputs to Process (for batch estimation)",
    min_value=1,
    value=1,
)

# Sidebar model selection
st.sidebar.title("Select Models")
selected_models = [
    model
    for model in MODELS
    if st.sidebar.checkbox(model, value=(model in ["gpt-4o", "o1", "gemini-2.0-flash"]))
]

# Reactive output logic
if selected_models and (
    input_prompt.strip()
    or output_text.strip()
    or manual_input_tokens > 0
    or manual_output_tokens > 0
):
    token_results = []
    cost_results = []

    for model_name in selected_models:
        model = MODELS[model_name]

        input_tokens = manual_input_tokens or model.count_tokens(text=input_prompt)
        output_tokens = manual_output_tokens or model.count_tokens(text=output_text)
        total_tokens = input_tokens + output_tokens

        info = model.pricing.get_cost(input_tokens, output_tokens)
        info["max_tokens"] = 1024

        limit_flag = (
            "⚠️ Exceeds Limit"
            if total_tokens > info["max_tokens"]
            else "✅ Within Limit"
        )

        # Table 1: token info
        token_results.append(
            {
                "Model": model_name,
                f"Input Tokens × {num_inputs}": input_tokens * num_inputs,
                f"Output Tokens  × {num_inputs}": output_tokens * num_inputs,
                f"Total Tokens × {num_inputs}": total_tokens * num_inputs,
            }
        )

        rt_cost_per = info["input"]["rt_cost"]
        bt_cost_per = info["input"]["batch_cost"]
        rt_total = rt_cost_per * num_inputs
        bt_total = bt_cost_per * num_inputs

        # Table 2: cost comparison
        cost_results.append(
            {
                "Model": model_name,
                "Realtime / Input": f"${rt_cost_per:.4f}",
                f"Realtime × {num_inputs}": f"${rt_total:.4f}",
                "Batch / Input": f"${bt_cost_per:.4f}",
                f"Batch × {num_inputs}": f"${bt_total:.4f}",
            }
        )

    # Display tables
    st.subheader("🔢 Token Usage Summary")
    st.table(pd.DataFrame(token_results))

    st.subheader("💰 Cost Comparison (Realtime vs Batch)")
    st.table(pd.DataFrame(cost_results))
