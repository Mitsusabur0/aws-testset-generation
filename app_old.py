import streamlit as st
import pandas as pd
import numpy as np


# FILE
INPUT_PARQUET = "evaluations/testset_results.parquet"


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    layout="wide"
)

# ==========================================
# 2. LOAD & PREPROCESS DATA
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet(INPUT_PARQUET)

        # --- PREPROCESSING: Map Synthesizer to Complexity ---
        def get_complexity(name):
            name = str(name).lower()
            if 'single' in name:
                return 'Single Hop'
            elif 'multi' in name:
                return 'Multi Hop'
            return 'Other'

        df['complexity'] = df['synthesizer_name'].apply(get_complexity)
        return df
    except FileNotFoundError:
        st.error("File 'evaluation_results.parquet' not found. Please run the evaluator script first.")
        return pd.DataFrame()

df_original = load_data()

if df_original.empty:
    st.stop()

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
st.sidebar.header("Filters")

# Filter by Complexity (Single vs Multi Hop)
all_complexities = df_original['complexity'].unique().tolist()
selected_complexity = st.sidebar.multiselect(
    "Query Complexity", 
    all_complexities, 
    default=all_complexities
)

# Filter by Query Style
all_styles = df_original['query_style'].unique().tolist()
selected_styles = st.sidebar.multiselect(
    "Query Style", 
    all_styles, 
    default=all_styles
)

# Apply Filters
df = df_original[
    (df_original['complexity'].isin(selected_complexity)) &
    (df_original['query_style'].isin(selected_styles))
]

st.sidebar.markdown("---")
st.sidebar.info(f"Showing **{len(df)}** out of **{len(df_original)}** test cases.")

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
st.title("RAG Retrieval Evaluation")

tab1, tab2 = st.tabs(["Dashboard Overview", "Test Case Explorer"])

# ---------------------------------------------------------------------
# TAB 1: DASHBOARD OVERVIEW
# ---------------------------------------------------------------------
with tab1:
    # A. KPI Cards
    st.subheader("Aggregate Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Hit Rate", f"{df['hit_rate'].mean():.1%}", help="Queries with at least 1 correct context.")
    col2.metric("MRR", f"{df['mrr'].mean():.3f}", help="Mean Reciprocal Rank (Higher is better).")
    col3.metric("Recall", f"{df['recall'].mean():.1%}", help="% of expected contexts found.")
    col4.metric("Precision", f"{df['precision'].mean():.1%}", help="% of retrieved contexts that were relevant.")

    st.markdown("---")

    # B. Charts
    col_chart_1, col_chart_2 = st.columns(2)

    with col_chart_1:
        st.subheader("Performance by Complexity")
        # Compare Single Hop vs Multi Hop
        if not df.empty:
            chart_data = df.groupby("complexity")[["hit_rate", "mrr"]].mean()
            st.bar_chart(chart_data)
        else:
            st.info("No data available for this filter.")

    with col_chart_2:
        st.subheader("Performance by Query Style")
        if not df.empty:
            chart_data_style = df.groupby("query_style")[["hit_rate", "mrr"]].mean()
            st.bar_chart(chart_data_style)
        else:
            st.info("No data available for this filter.")

# ---------------------------------------------------------------------
# TAB 2: TEST CASE EXPLORER
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Deep Dive: Individual Test Cases")
    st.caption("Click on any row to inspect the retrieval details.")

    # 1. THE SELECTOR TABLE
    # Showing 'complexity' instead of 'persona_name'
    # display_cols = ['user_input', 'hit_rate', 'recall', 'complexity']
    display_cols = ['user_input', 'hit_rate', 'mrr', 'complexity']
    
    event = st.dataframe(
        df[display_cols].style.format({'hit_rate': '{:.2f}', 'mrr': '{:.2f}'})
        .background_gradient(subset=['mrr'], cmap="RdYlGn", vmin=0, vmax=1),
        # use_container_width=True,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=300
    )

    # 2. THE DETAIL VIEW
    if len(event.selection['rows']) > 0:
        # Get selected row index relative to the filtered dataframe
        selected_index = event.selection['rows'][0]
        selected_row = df.iloc[selected_index]
        
        st.divider()
        st.markdown(f"### Selected Query: _{selected_row['user_input']}_")
        
        # Badges for context
        st.caption(f"**Complexity:** {selected_row['complexity']} | **Style:** {selected_row['query_style']}")

        # Row Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hit Rate", f"{selected_row['hit_rate']:.0%}")
        m2.metric("MRR", f"{selected_row['mrr']:.2f}")
        m3.metric("Recall", f"{selected_row['recall']:.2f}")
        m4.metric("Precision", f"{selected_row['precision']:.2f}")

        # Comparison View
        c1, c2 = st.columns(2)
        
        # Normalization helper for display logic
        def norm(t): return str(t).lower().replace('\n', ' ').strip()
        
        # Handle Ground Truth
        gt_list = selected_row['reference_contexts']
        # If it's a numpy array, convert to list
        if isinstance(gt_list, np.ndarray):
            gt_list = gt_list.tolist()
        
        gt_normalized = [norm(x) for x in gt_list]

        with c1:
            st.info("**Expected Contexts (Ground Truth)**")
            for i, ctx in enumerate(gt_list):
                st.markdown(f"**{i+1}.** {ctx}")
                st.markdown("---")

        with c2:
            st.success("**Retrieved Contexts (System Output)**")
            
            retrieved_data = selected_row['retrieved_contexts']
            
            # --- FIX: Safe check for array length ---
            if len(retrieved_data) == 0:
                st.warning("No contexts retrieved.")
            else:
                for i, ctx in enumerate(retrieved_data):
                    ctx_str = str(ctx) # Ensure string format
                    is_match = norm(ctx_str) in gt_normalized
                    
                    if is_match:
                        st.markdown(f"✅ **{i+1}.** {ctx_str}")
                    else:
                        st.markdown(f"❌ **{i+1}.** {ctx_str}")
                    st.markdown("---")
    else:
        st.info("Select a row in the table above to view details.")