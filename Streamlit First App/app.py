import home
import one 
import two
import three


import streamlit as st

st.audio(open('sweet.mp3', 'rb').read(), format='audio/ogg')

PAGES = {
    "Home": home,
    "Uni-Grams": one,
    "Bi-Grams": two,
    "Tri-Grams": three,
}

st.sidebar.title('Navigation Bar')

selection = st.sidebar.selectbox("Go to: \n", list(PAGES.keys()))
page = PAGES[selection]
page.app()