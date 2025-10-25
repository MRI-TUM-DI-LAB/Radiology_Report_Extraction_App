import streamlit as st
from datetime import datetime
import operator
from shared.chroma_interface import upload_files, get_all_reports, delete_report, set_language, DETECT_LANGUAGES
import streamlit as st
import uuid

# Initialize key in session_state if not set
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = str(uuid.uuid4())

# Function to reset uploader key
def reset_uploader():
    st.session_state["uploader_key"] = str(uuid.uuid4())

# Get all reports
reports = get_all_reports()

# Convert raw data into structured list
report_list = []
for idx in range(len(reports["ids"])):
    meta = reports["metadatas"][idx]
    report_list.append({
        "id": reports["ids"][idx],
        "text": reports["documents"][idx],
        "name": meta.get("filename", "Unknown"),
        "modality": meta.get("modality", "Unknown"),
        "upload time": meta.get("upload_time", "Unknown"),
        "language": meta.get("language", "Unknown"),
        "num_placeholders": meta.get("num_placeholders", 0),
    })

with st.expander("Delete All Reports", expanded=False):
    st.warning("This will permanently delete all reports.")
    if st.button("Delete All", type="primary"):
        progress_bar = st.progress(0)
        total = len(report_list)
        for i, rep in enumerate(report_list):
            delete_report(rep['id'])
            progress_bar.progress((i + 1) / total)
        st.success("All reports deleted.")
        st.rerun()

st.markdown("&nbsp;" * 20, unsafe_allow_html=True)

# Set page title and icon
uploaded_files = st.file_uploader("Select files", type=["json"], accept_multiple_files=True, key=st.session_state["uploader_key"])

if st.button("Upload", type="primary"):
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing file {i + 1} of {len(uploaded_files)}: {file.name}")
            result = upload_files([file])  # Adjust as needed
            results.append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))

        if all(results):
            st.success("All files uploaded to ChromaDB!")
        else:
            st.error("Some files could not be uploaded.")

        reset_uploader()  # Clear file input
        st.rerun()        # Refresh app to apply reset
    else:
        st.warning("Please select at least one file.")


st.markdown("&nbsp;" * 20, unsafe_allow_html=True)

st.sidebar.header("Filter & Sort Reports")

search_term = st.sidebar.text_input("Search filename", placeholder="e.g. abdomen")

modalities = list(set([r["modality"] for r in report_list]))
selected_modalities = st.sidebar.multiselect("DICOM Modality", modalities, default=modalities)

languages = list(set([r["language"] for r in report_list]))
selected_languages = st.sidebar.multiselect("Language", languages, default=languages)


# Sort options
sort_fields = ["name", "category", "created", "upload time", "size", "language", "modality"]
sort_by = st.sidebar.selectbox("Sort by", sort_fields)
sort_order = st.sidebar.radio("Sort order", ["Ascending", "Descending"])
reverse_sort = sort_order == "Descending"

filtered_reports = [
    r for r in report_list
    if r["modality"] in selected_modalities
    and r["language"] in selected_languages
    and search_term.lower() in r["name"].lower()
]

try:
    filtered_reports.sort(key=operator.itemgetter(sort_by), reverse=reverse_sort)
except TypeError:
    # Fallback if field is not directly comparable
    pass

st.write(f"### Showing {len(filtered_reports)} Templates")

for rep in filtered_reports:
    with st.expander(f"`{rep['language']}` {rep['name']}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.json(rep["text"])
        
        with col2:
            dt = datetime.fromisoformat(rep['upload time'])
            formatted_time = dt.strftime("%B %d, %Y — %H:%M")
            
            # Display metadata
            st.markdown("### Metadata")
            st.write(f"**ID:** {rep['id']}")
            st.write(f"**Modality:** {rep['modality']}")
            st.write(f"**Upload Time:** {formatted_time}")
            st.write(f"**Number of Placeholders:** {rep['num_placeholders']}")
            
            st.markdown("---")

            # Language selector
            new_lang = st.selectbox(
                "Change language",
                DETECT_LANGUAGES,
                index=DETECT_LANGUAGES.index(rep["language"]),
                key=f"lang_select_{rep['id']}"
            )
            if new_lang != rep["language"]:
                if st.button("Update Language", key=f"update_lang_{rep['id']}"):
                    success = set_language(rep["id"], new_lang)
                    if success:
                        st.success("Language updated.")
                        st.rerun()
                    else:
                        st.error("Failed to update language.")
            
            st.markdown("---")

            # Delete button
            if st.button("🗑️ Delete Template", key=f"del_{rep['id']}"):
                delete_report(rep['id'])
                st.rerun()
