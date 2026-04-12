import streamlit as st
import base64
import pandas as pd

from src.config import SEMANTIC_SCHOLAR_KEY, SERPAPI_KEY, IEEE_API_KEY, GEMINI_API_KEY1, GEMINI_API_KEY2
from src.database import authenticate_user, create_chat, get_user_chats, create_user, chats_col, get_chat, update_chat_ranked_papers, save_outputs_db, append_chat_qa
from src.models import Paper
from src.utils import markdown_to_pdf_bytes
from src.pipeline import fetch_papers, dedup_papers, exclude_papers, score_and_rank
from src.ai import enrich_research_query, generate_report, analyze_intent, generate_question_from_input, answer_question, generate_query_from_input

st.set_page_config(page_title="Deep Research Agent", layout="wide")
st.markdown(
    """
        <style>
            .sidebar-chat {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 6px 8px;
                    margin: 4px 0;
                    border-radius: 6px;
                    background: #f9f9f9;
                }
        </style>
    """, 
    unsafe_allow_html=True)

st.title("🔬 Deep Research Agent")

# session login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Authentication
if not st.session_state.logged_in:
    st.header("Login / Signup")
    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
    with tab1:
        uname = st.text_input("Username", key="signin_username")
        pwd = st.text_input("Password", type="password", key="signin_password")
        if st.button("Sign In"):
            if authenticate_user(uname, pwd):
                st.session_state.logged_in = True
                st.session_state.username = uname
                user_chats = get_user_chats(uname)
                st.session_state.active_chat_id = user_chats[0]["_id"] if user_chats else create_chat(uname)["_id"]
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        uname = st.text_input("New Username", key="signup_username")
        pwd1 = st.text_input("Password", type="password", key="signup_password1")
        pwd2 = st.text_input("Re-enter Password", type="password", key="signup_password2")
        if st.button("Sign Up"):
            if pwd1 != pwd2:
                st.error("Passwords do not match")
            else:
                if create_user(uname, pwd1):
                    st.success("Account created! Please sign in.")
                else:
                    st.error("Username exists")
    st.stop()

username = st.session_state.username

# Sidebar: chats + create
with st.sidebar:
    cols = st.columns([8,1])
    with cols[0]:
        st.markdown(f"#### **Hello,** {username}")
    with cols[1]:
        if st.button("➜]"):
            st.session_state.logged_in = False
            st.rerun()
    st.markdown("### 💬 Research Chats")
    if st.button("✚ Create New Chat", use_container_width=True):
        new_chat = create_chat(username)
        st.session_state.active_chat_id = new_chat["_id"]
        st.rerun()

    chats = get_user_chats(username)
    for chat in chats:
        cols = st.columns([8,1])
        with cols[0]:
            if st.button(chat.get("title") or "Untitled", key=f"chat_{chat['_id']}"):
                st.session_state.active_chat_id = chat["_id"]
                st.rerun()
        with cols[1]:
            if st.button("✖", key=f"del_{chat['_id']}"):
                chats_col.delete_one({"_id": chat["_id"]})
                if st.session_state.active_chat_id == chat["_id"]:
                    st.session_state.active_chat_id = None
                st.rerun()

# Main area (research-only)
if st.session_state.active_chat_id:
    chat = get_chat(st.session_state.active_chat_id)
    st.subheader(f"📝 {chat.get('title').capitalize() or 'Untitled'}")

    st.markdown("Use the panel below to run a research report, refine it, ask questions, or accept the report.")

    with st.expander("**Research Settings**", expanded=True):
        topic = st.text_input("Research Topic", value=chat.get("meta", {}).get("last_topic", ""))
        max_papers = st.number_input("Max papers to fetch", min_value=10, max_value=500, value=chat.get("meta", {}).get("max_papers", 50))
        top_k = st.number_input("Top papers for context", min_value=3, max_value=100, value=chat.get("meta", {}).get("top_k", 10))
        w_rel = st.number_input("Weight - Relevance", min_value=0.0, max_value=1.0, value=chat.get("meta", {}).get("w_rel", 0.4))
        w_cit = st.number_input("Weight - Citations", min_value=0.0, max_value=1.0, value=chat.get("meta", {}).get("w_cit", 0.25))
        w_rec = st.number_input("Weight - Recency", min_value=0.0, max_value=1.0, value=chat.get("meta", {}).get("w_rec", 0.35))
        st.caption("Note: Weights should ideally sum to 1.0, but the system will normalize them if they don't.")

    if "ranked_papers" not in st.session_state:
        st.session_state.ranked_papers = None
    if "last_fetch_topic" not in st.session_state:
        st.session_state.last_fetch_topic = None
    if "feedback_value" not in st.session_state:
        st.session_state.feedback_value = ""
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    if st.session_state.get("active_chat_id"):
        chat_doc = get_chat(st.session_state.active_chat_id)
        if chat_doc:
            if chat_doc.get("ranked_papers"):
                st.session_state.ranked_papers = chat_doc.get("ranked_papers", None)
                st.session_state.last_fetch_topic = chat_doc.get("meta", {}).get("last_topic", st.session_state.last_fetch_topic)
            if chat_doc.get("qa_history") is not None:
                st.session_state.qa_history = chat_doc.get("qa_history", [])

    if st.button("▶️ Run Research (Fetch & Rank)"):
        if not topic:
            st.error("Please provide a topic.")
        else:
            api_keys = {
                "SEMANTIC_SCHOLAR_KEY": SEMANTIC_SCHOLAR_KEY,
                "SERPAPI_KEY": SERPAPI_KEY,
                "IEEE_KEY": IEEE_API_KEY
            }
            with st.spinner("Enriching research query..."):
                enriched_topic = enrich_research_query(topic, GEMINI_API_KEY1)
                st.info(f"Expanded query: {enriched_topic}")

            with st.spinner("Fetching papers..."):
                collected = fetch_papers(enriched_topic, max_papers, api_keys)
            if not collected:
                st.error("No papers found. Try broadening the query.")
            else:
                collected = dedup_papers(collected)
                st.info(f"{len(collected)} papers after deduplication.")
                st.info("Scoring and ranking papers.")
                ranked = score_and_rank(collected, enriched_topic, (w_rel, w_cit, w_rec), GEMINI_API_KEY1)
                ranked_serialized = [p.to_row() for p in ranked]
                st.session_state.ranked_papers = ranked_serialized
                st.session_state.last_fetch_topic = topic
                st.session_state.enriched_topic = enriched_topic

                try:
                    update_chat_ranked_papers(chat["_id"], ranked_serialized, meta={"last_topic": topic, "enriched_topic": enriched_topic, "max_papers": max_papers, "top_k": top_k, "w_rel": w_rel, "w_cit": w_cit, "w_rec": w_rec})
                    st.success("Ranking complete and persisted to chat.")
                except Exception as e:
                    st.error(f"Failed to persist ranked papers: {e}")

                st.rerun()

    if st.session_state.ranked_papers and st.session_state.last_fetch_topic == topic:
        st.markdown("### 🔢 Ranked Papers (Top results)")
        ranked_preview = st.session_state.ranked_papers
        for p in ranked_preview:
            val = p.get("citation_count")
            try:
                p["citation_count"] = int(val) if val is not None else 0
            except Exception:
                p["citation_count"] = 0
        df = pd.DataFrame(ranked_preview)
        display_cols = ["title", "year", "citation_count", "similarity", "score", "source"]
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].copy()
        display_df = display_df.rename(columns={"title": "Title", "year": "Year", "citation_count": "Cites", "similarity": "Relevance", "score": "Overall Score", "source": "Source"})
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(display_df.head(100), column_config={"Year": st.column_config.NumberColumn(format="%d")})

        if st.button("📝 Generate Report"):
            ranked_source = st.session_state.get("ranked_papers")
            if not ranked_source:
                chat_refreshed = get_chat(chat["_id"])
                ranked_source = chat_refreshed.get("ranked_papers", [])

            ranked_objs = []
            for d in (ranked_source or []):
                c = d.get("citation_count")
                try:
                    c = int(c) if c is not None else None
                except Exception:
                    c = None
                d["citation_count"] = c
                ranked_objs.append(Paper(**d))
            with st.spinner("Generating Report..."):
                report_topic = chat.get("meta", {}).get("enriched_topic", st.session_state.get("enriched_topic", topic))
                report_md = generate_report(report_topic, ranked_objs, top_k, GEMINI_API_KEY1)
            
            save_outputs_db(chat["_id"], ranked_objs, report_md, out_base=f"report_{chat['_id']}")
            st.success("Report generated and saved.")
            
            if chat.get("title", "").startswith(""):
                new_title = f"{topic[:50]}"
                chats_col.update_one({"_id": chat["_id"]}, {"$set": {"title": new_title}})
            chats_col.update_one({"_id": chat["_id"]}, {"$set": {"meta.last_topic": topic, "meta.max_papers": max_papers, "meta.top_k": top_k, "meta.w_rel": w_rel, "meta.w_cit": w_cit, "meta.w_rec": w_rec}})
            st.rerun()

    chat = get_chat(st.session_state.active_chat_id)
    if chat.get("report_md"):
        st.markdown("### 📄 Latest Report")
        try:
            pdf_bytes = markdown_to_pdf_bytes(chat["report_md"], title=chat.get("title", "Report"))
            st.download_button(
                label="Download Report (as PDF)",
                data=pdf_bytes,
                file_name=f"{chat['title'].replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Could not generate PDF: {e}")
        with st.expander("Show report (markdown)"):
            st.code(chat["report_md"][:200000])

        st.markdown("### 🔁 Refinement & QnA")

        if "feedback_value" not in st.session_state:
            st.session_state.feedback_value = ""
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        if st.session_state.qa_history:
            for i, pair in enumerate(st.session_state.qa_history, 1):
                if pair.get("q") != "accept":
                    st.markdown(f"**Q{i}:** **{pair.get('q')}**")
                    st.markdown(f"**A{i}:** {pair.get('a')}")
                else:
                    st.markdown(f"***You have accepted the report and you are satisfied with the results.***")
                st.markdown("---")

        last_qa = st.session_state.qa_history[-1] if st.session_state.qa_history else None
        is_accepted = last_qa and "accept" in (last_qa.get('q', '').lower())

        if not is_accepted:
            if "clear_feedback" not in st.session_state:
                st.session_state["clear_feedback"] = False

            if st.session_state.get("clear_feedback", False):
                st.session_state["feedback_value"] = ""
                st.session_state["clear_feedback"] = False

            user_feedback = st.text_area(
                "Enter feedback to refine the report, ask a question, or say 'accept' to finish.",
                key="feedback_value",
                height=140
            )

            col1, col2 = st.columns(2)
            if col1.button("Classify Intent"):
                if not st.session_state.feedback_value.strip():
                    st.warning("Type something first.")
                else:
                    intent = analyze_intent(st.session_state.feedback_value, GEMINI_API_KEY2)
                    st.info(f"Intent detected: **{intent}**")

            if col2.button("Process Input"):
                if not st.session_state.feedback_value.strip():
                    st.warning("Type something first.")
                else:
                    user_text = st.session_state.feedback_value
                    intent = analyze_intent(user_text, GEMINI_API_KEY2)
                    
                    if intent == "accept":
                        acceptance_q = "accept"
                        acceptance_a = "User accepted the report and is satisfied with the results."
                        st.session_state.qa_history.append({"q": acceptance_q, "a": acceptance_a})
                        try:
                            append_chat_qa(chat["_id"], acceptance_q, acceptance_a)
                        except Exception:
                            pass
                        st.success("Marked as accepted. Report finalized.")
                        st.session_state["clear_feedback"] = True
                        st.rerun()

                    elif intent == "ask":
                        question = generate_question_from_input(user_text, GEMINI_API_KEY2)
                        st.info(f"Generated question: {question}")
                        st.markdown("**Question:**")
                        st.markdown(f"> {question}")

                        answer_placeholder = st.empty()
                        with st.spinner("Generating answer..."):
                            papers_objs = [Paper(**p) for p in (chat.get("papers") or [])]
                            answer = answer_question(question, chat.get("report_md", ""), papers_objs, chat.get("meta", {}).get("top_k", 10), GEMINI_API_KEY2)
                            answer_placeholder.markdown("**Answer:**\n\n" + answer)

                        st.session_state.qa_history.append({"q": question, "a": answer})
                        try:
                            append_chat_qa(chat["_id"], question, answer)
                        except Exception as e:
                            st.error(f"Failed to persist QA: {e}")

                        st.session_state["clear_feedback"] = True
                        st.rerun()

                    elif intent == "refine":
                        extracted_prompt = generate_query_from_input(user_text, GEMINI_API_KEY2)
                        st.info(f"Extracted query: '{extracted_prompt}'")
                        
                        with st.spinner("Enriching research query..."):
                            refinement_prompt = enrich_research_query(extracted_prompt, GEMINI_API_KEY2)
                            st.info(f"Enriched query: '{refinement_prompt}'")

                        api_keys = {"SEMANTIC_SCHOLAR_KEY": SEMANTIC_SCHOLAR_KEY, "SERPAPI_KEY": SERPAPI_KEY, "IEEE_KEY": IEEE_API_KEY}

                        with st.spinner("Fetching additional papers for refinement..."):
                            new_papers = fetch_papers(refinement_prompt, max(10, int(chat.get("meta", {}).get("max_papers", 80) // 2)), api_keys)
                            if not new_papers:
                                st.error("No new papers found for refinement query.")
                                st.stop()
                            
                        existing_papers = [Paper(**p) for p in (chat.get("papers") or [])]
                        new_papers = exclude_papers(new_papers, existing_papers[:int(chat.get("meta", {}).get("top_k", 25))])
                        merged = dedup_papers(existing_papers + new_papers)
                            
                        weights = (chat.get("meta", {}).get("w_rel", 0.4), chat.get("meta", {}).get("w_cit", 0.25), chat.get("meta", {}).get("w_rec", 0.35))
                        st.info("Re-ranking the papers based on refinement prompt...")
                        reranked = score_and_rank(merged, refinement_prompt, weights, GEMINI_API_KEY2)
                        reranked_serialized = [p.to_row() for p in reranked]

                        update_chat_ranked_papers(chat["_id"], reranked_serialized, meta={"last_topic": refinement_prompt, "enriched_topic": refinement_prompt})
                        st.session_state.ranked_papers = reranked_serialized
                        st.session_state.last_fetch_topic = refinement_prompt
                        st.session_state.enriched_topic = refinement_prompt

                        refine_note = f"🧠 Paper Refinement applied: **{refinement_prompt}**"
                        st.session_state.qa_history.append({"q": refine_note, "a": "Ranking updated. Please inspect and generate."})
                        append_chat_qa(chat["_id"], refine_note, "Ranking updated. Please inspect and generate.")

                        st.success("Refinement complete. Inspect and generate a new report.")
                        st.session_state["clear_feedback"] = True
                        st.rerun()

    else:
        st.info("No report generated yet. Please run the research pipeline above.")

else:
    st.info("Select or create a research chat from the sidebar.")
