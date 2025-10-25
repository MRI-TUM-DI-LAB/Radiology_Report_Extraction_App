import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
from logic.report_structuring.report_structuring import ReportStructuringProcessor
from logic.icd_coding.icd_coding import ICDInference
from logic.radlex.radlex_coding import RadLexInference, RadLexProcessor #new
from config import CONFIG


def status_block(title):
    with st.container():
        st.markdown("#### " + title)
        progress_placeholder = st.empty()
        result_placeholder = st.empty()
    return progress_placeholder, result_placeholder

html(
    """
    <script>
    window.addEventListener('load', () => {
        const scrollToBottom = () => {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        };
        const observer = new MutationObserver(scrollToBottom);
        observer.observe(document.body, { childList: true, subtree: true });
    });
    </script>
    """,
    height=0,
)

# --- Main Input Area ---
st.markdown("### Radiology Report Processing")

if "freetext" not in st.session_state:
    st.session_state.freetext = ""

st.session_state.freetext = st.text_area(
    "Paste or write the radiology report here...",
    value=st.session_state.freetext,
    height=300,
    placeholder="Observations, findings, impressions etc in freetext e.g. 'The chest X-ray shows no acute cardiopulmonary abnormalities.'",
)

if st.button("Submit"):
    if st.session_state.freetext.strip():

        # --- ICD Coding ---
        i_progress_placeholder, i_result = status_block("ICD Coding")

        if CONFIG['icd_coding']['enabled']:
            with st.spinner("Running ICD coding..."):
                try:
                    icd_infer = ICDInference()  
                    original_phrase, icd_codes, icd_descriptions, similarity = icd_infer.predict_icd(st.session_state.freetext)

                    if icd_codes:
                        
                        table_data = {
                            "Phrase": original_phrase,
                            "ICD Code": icd_codes,
                            "Description": icd_descriptions,
                            "Similarity": [f"{sim:.3f}" for sim in similarity]
                        }

                        df = pd.DataFrame(table_data)

                        styled_df = df.style.set_table_styles([
                            {
                                'selector': 'thead th',
                                'props': [
                                    ('background-color', '#4682B4'),  # Steel Blue
                                    ('color', 'white'),
                                    ('font-weight', 'bold'),
                                    ('text-align', 'center')
                                ]
                            }
                        ])

                        st.dataframe(styled_df, use_container_width=True)
                                        
                    else:
                        i_result.warning("No ICD codes found.")
                except Exception as e:
                    i_result.error(f"Error during ICD coding: {e}")
        else:
            i_progress_placeholder.warning("ICD Coding is disabled.")

        # --- RadLex Mapping ---
        r_progress, r_result = status_block("RadLex Mapping")
        if CONFIG["radlex"]["enabled"]:
            RadLexProcessor(r_progress, r_result).run(st.session_state.freetext)
        else:
            r_progress.warning("RadLex Mapping is disabled.")
    
        # --- Report Structuring ---
        t_progress_placeholder, t_result = status_block("Report Structuring")
        if CONFIG['report_structuring']['enabled']:
            # Modified: pass the pipeline_format into the processor constructor
            processor = ReportStructuringProcessor(
                t_progress_placeholder,
                t_result
            )
            processor.process_freetext(st.session_state.freetext)
        else:
            t_progress_placeholder.warning("Report Structuring is disabled.")

    else:
        st.warning("Please enter a valid radiology report.")