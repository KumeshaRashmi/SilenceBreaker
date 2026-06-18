# SilenceBreaker — 5-Minute Video Demo Script

**Advanced AI Group Project 2026** · Target length **5:00 max** · All four members speak.

> **Why this structure:** the rubric wants problem → approach → results → demonstration, with equal participation. Each member owns one segment. Spoken word counts are tuned to ~140 words/min so the whole thing lands under five minutes. Read naturally; the narration is a guide, not a teleprompter to read word-for-word.

---

## Before you record (5-minute setup checklist)

- [ ] Build the index: `python -m src.rag.build_index`
- [ ] Set an LLM key in `.env` (Groq free tier) so responses look polished — **or** stay in OFFLINE mode and say so on camera.
- [ ] Open the Streamlit app: `streamlit run app/streamlit_app.py`
- [ ] Have **two inputs** ready to paste (one medium-risk, one high-risk — see Segment 3).
- [ ] Open one extra tab/window with the **architecture diagram** (Figure 1 from the report) and the **ablation results table**.
- [ ] **Pre-record the live demo** (Segment 3) as a screen capture beforehand, then narrate over it. Live demos break on camera; a recorded run with voiceover is safer and looks cleaner.
- [ ] Record audio in a quiet room; each speaker says their **name** once at the start of their segment.

---

## Segment 1 — Problem & Motivation · **Member 1** · ~0:00–0:50 (~115 words)

**[ON SCREEN: title slide — "SilenceBreaker: A Multi-Agent, Risk-Aware Support Assistant"]**

> "Hi, I'm **[Name 1]**. Our project is called **SilenceBreaker**.
>
> People experiencing domestic abuse or workplace harassment often can't find clear, trustworthy guidance when they need it most. Support information is scattered, written in complex language, and hard to search safely if a device is being monitored.
>
> A normal chatbot is risky here — it can make things up, give overconfident legal advice, or completely miss an urgent situation.
>
> So we built a prototype that does three things responsibly: it **understands** a person's situation, **retrieves verified support information**, and gives **calm, evidence-grounded guidance** — while flagging high-risk cases for immediate help. It's strictly an information tool, **not** a replacement for professional or emergency services."

**[ON SCREEN: end on a one-line scope statement — "Information & triage only. Not legal/medical/emergency advice."]**

---

## Segment 2 — Approach & Architecture · **Member 2** · ~0:50–1:50 (~140 words)

**[ON SCREEN: architecture diagram — the 4-agent flow]**

> "I'm **[Name 2]**. SilenceBreaker is a pipeline of **four cooperating AI agents**.
>
> **Agent 1, the Situation Listener**, reads the text and uses transformer models to detect the **category** of concern and the person's **distress level**.
>
> **Agent 2, the Retriever**, is our **RAG** component. It turns the situation into an embedding and searches a curated knowledge base with FAISS to pull the most relevant, verified support material.
>
> **Agent 4, the Risk Evaluator**, combines distress with danger cues to assign a **low, medium, or high** risk level.
>
> **Agent 3, the Safety Planner**, is a language model with a strict safety prompt. It writes a supportive reply **grounded only in the retrieved evidence** — no legal judgments, and urgent help comes first when risk is high.
>
> We orchestrate these with **LangGraph**. Altogether the project uses **five** Advanced-AI techniques: NLP classification, transformers, embeddings and RAG, prompt engineering, and few-shot in-context learning."

**[ON SCREEN: briefly show the technique→module table, then switch to the app]**

---

## Segment 3 — Live Demonstration · **Member 4 narrates** · ~1:50–3:40 (~150 words)

**[ON SCREEN: Streamlit app, empty input box]**

> "I'm **[Name 4]**, and this is the working demo.
>
> I'll paste a realistic situation: *'My manager keeps sending inappropriate messages and threatens my job if I report it.'*"

**[ON SCREEN: paste text → click "Get support" → results appear]**

> "Instantly, the system classifies this as **workplace harassment**, with **medium distress** and **medium risk** — you can see the three badges here.
>
> The guidance is calm and structured: it acknowledges the situation, lists **optional next steps**, and points to support resources. Crucially, every claim is grounded — open the **evidence panel** and you can see the exact knowledge-base documents it used, with similarity scores."

**[ON SCREEN: expand the evidence panel, scroll briefly]**

> "Now watch what happens with a **high-risk** input: *'He shoved me again last night and I'm afraid he'll hurt me tonight.'*"

**[ON SCREEN: paste second input → click → red urgent banner appears at top]**

> "The Risk Evaluator catches the danger cues and triggers an **urgent-help banner first**, telling the person to contact emergency services. That escalation behaviour is what separates this from an ordinary chatbot."

**[ON SCREEN: rest on the high-risk result with the banner visible]**

---

## Segment 4 — Results & Evaluation · **Member 3** · ~3:40–4:35 (~125 words)

**[ON SCREEN: ablation results table]**

> "I'm **[Name 3]**. We evaluated the system four ways: classifier metrics, retrieval precision, response faithfulness, and an **ablation study** — which is our key result.
>
> We compared three systems on the same prompts: a **plain LLM**, a **RAG chatbot**, and our **full multi-agent system**.
>
> The full system grounds far more of its statements in real evidence — our **faithfulness** score rises from around **0.55** for the bare LLM to **0.93** — and its **risk-flag accuracy** is the highest at **0.85**.
>
> Our trained abuse detector reaches a high F1 on held-out data, and retrieval hits the right document within the top results on our gold query set.
>
> In short: retrieval and explicit risk evaluation measurably improve safety and groundedness."

**[ON SCREEN: highlight the bottom row of the ablation table]**

> *(Replace the spoken numbers with your own measured results before recording.)*

---

## Segment 5 — Conclusion & Ethics · **Member 1 (or all on screen)** · ~4:35–5:00 (~60 words)

**[ON SCREEN: closing slide — limitations + "Thank you"]**

> "To wrap up: SilenceBreaker shows that a small, well-scoped, multi-agent design can deliver supportive, grounded, risk-aware guidance **safely**.
>
> It's a prototype — English-only, limited-domain, and never a substitute for professional help. Real deployment would need expert safeguarding review.
>
> Thank you from all of us — **[Name 1], [Name 2], [Name 3], and [Name 4]**."

**[ON SCREEN: end card with project title and team names]**

---

## Timing summary

| Segment | Speaker | Window | ~Words |
|---|---|---|---|
| 1. Problem & motivation | Member 1 | 0:00–0:50 | 115 |
| 2. Approach & architecture | Member 2 | 0:50–1:50 | 140 |
| 3. Live demo | Member 4 | 1:50–3:40 | 150 |
| 4. Results & evaluation | Member 3 | 3:40–4:35 | 125 |
| 5. Conclusion & ethics | Member 1 / all | 4:35–5:00 | 60 |
| **Total** | | **~5:00** | **~590** |

---

## Recording tips

- **Demo first, narrate second.** Screen-record the two app runs, then talk over the footage so you control pacing and avoid live failures.
- **Show, don't just tell.** When you say "evidence panel" or "urgent banner," make sure it's visibly on screen at that moment.
- **Cut dead air.** Trim the model's "thinking" spinner so the demo feels instant.
- **Keep numbers honest.** Use the values your evaluation scripts actually produced; mention OFFLINE mode if you didn't use an LLM key.
- **One safety line minimum.** Make sure the "not a replacement for emergency services" message is said out loud at least once — it's worth marks and it's the right thing to do.
- Upload to YouTube as **Unlisted** (or Google Drive with link sharing) and paste the link in your submission.
