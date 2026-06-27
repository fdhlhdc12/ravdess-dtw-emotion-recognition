# =====================================================
# RENDER PAGE
# =====================================================
if st.session_state["page"] == "📊 Dashboard":
    show_dashboard()
elif st.session_state["page"] == "🎯 Prediksi":
    show_prediksi()
elif st.session_state["page"] == "📈 Analytics":
    show_analytics()
elif st.session_state["page"] == "🧠 Model & Algoritma":
    show_model()
elif st.session_state["page"] == "📁 Dataset":
    show_dataset()
else:
    # Fallback jika halaman tidak dikenal
    st.warning("⚠️ Halaman tidak ditemukan. Mengarahkan ke Dashboard.")
    show_dashboard()
