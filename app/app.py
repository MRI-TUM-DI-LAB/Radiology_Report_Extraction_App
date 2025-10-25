import streamlit as st
from dotenv import load_dotenv
st.set_page_config(layout="wide")
import log

logger = log.setup_custom_logger('root')
logger.debug('App started')

load_dotenv()

pages = [
        st.Page("pages/process_report.py", title="Process Report"),
        st.Page("pages/manage_templates.py", title="Manage Templates"),
    ]

pg = st.navigation(pages)
pg.run()
