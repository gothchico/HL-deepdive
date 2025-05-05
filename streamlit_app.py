import streamlit as st

def main():
    # builds the sidebar menu
    with st.sidebar:
        st.page_link('pages/1_tokens_analysis.py', label='HL Tokens Analysis', icon='📈')
        st.page_link('pages/2_airdrop_analysis.py', label='Airdrop ROI', icon='🎁')
        st.page_link('pages/3_fees_analysis.py', label='Fee Analysis', icon='💲')
        st.page_link('pages/4_liquidatorROI.py', label='HLP Liquidator ROI', icon='💧')
        st.page_link('pages/5_vault_analysis.py', label='HLP Vault Analysis', icon='🏦')
        st.page_link('pages/6_hybrid_analysis.py', label='Hybrid Analysis', icon='🧬')
        st.page_link('pages/7_regression_analysis.py', label='Regression Analysis', icon='📊')
if __name__ == '__main__':
    main()