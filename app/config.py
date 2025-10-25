import streamlit as st
import yaml

@st.cache_data
def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

