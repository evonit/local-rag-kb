🇺🇸 English | 🇰🇷 [한국어](README.ko.md)

# Self-Hosted Local RAG LLM (Prototype)

This repository contains a **self-hosted local RAG (Retrieval-Augmented Generation) LLM prototype**.

The project was developed during **technical research for building commercial LLM-based services**,  
and is shared here in a **deployable, reusable form**.

It is **not a personal local-PC document search tool**,  
but is intended for **knowledge exploration within an organization or team**  
(e.g. internal documents, shared files, reference materials).

---

## 📌 Purpose

- Share a **practical RAG prototype** created during LLM service development research
- Provide a reference for developers exploring:
  - local / self-hosted LLMs
  - document retrieval pipelines
  - non-English (Korean) document QA
- Enable discussion and knowledge sharing around RAG system design

---

## 🎯 Design Goals

- No dependency on commercial LLM APIs (e.g. OpenAI)
- Operable with **local or limited computing resources**
- Focus on **document retrieval and QA**, not chat or agent systems
- Optimized for **non-English documents**, especially Korean

---

## ⚠️ Scope & Limitations

- This is a **research-driven prototype**, not a polished product
- Only code that reached a **deployable state** is published
- The provided Front-End UI is **minimal and intended only for testing and demonstration**
  - It is not production-ready
  - Commercial UI code developed for actual client delivery is **not included**
- Not designed for:
  - personal desktop file browsing tools
  - large-scale, high-concurrency SaaS environments
- New experiments and prototypes may be added gradually

---

## 🧱 Version Overview

### 🔒 Ver 0.1 (Private)
- Initial prototype
- PDF upload and basic retrieval

---

### ✅ Ver 0.2
- PDF upload and retrieval
- Hybrid search (Dense + Sparse)
- Multi-step retrieval result refinement for Korean documents

---

### 🔒 Ver 0.3 (Private, Experimental)
Exploratory attempts that were later discarded:

- Chunk-level summary generation  
  → Increased processing time with limited benefit
- Advanced interpretation of tables, charts, and images using local LLMs  
  → Limited effectiveness and high cost
- **Knowledge Graph (KG) integration**
  - Tested automatic KG construction using LLM-based entity and relation extraction
  - Evaluated KG-assisted retrieval and answer refinement on top of hybrid search
  - Observed improvements in limited, well-scoped domains

  → Ultimately discarded due to the following limitations:
  - Local LLMs showed inconsistent quality when generating graphs in open-domain document sets
  - Graph structure quality degraded rapidly as document scope expanded
  - Maintaining a reliable KG without strict domain constraints proved impractical

---

### ✅ Ver 0.4
- Replaced custom parsing with **Docling**
- Added Vision LLM-based document parsing via Docling
- Support for additional document formats (e.g. Word)

---

## 📚 Documentation

Technical details are documented separately:

Technical Overview: [docs/INDEX.md](docs/INDEX.md)

---

## 📜 License

Apache License 2.0

---

## 💬 Feedback & Discussion

Questions, suggestions, and discussions are welcome.

If you have ideas, feedback, or run into issues while experimenting with this project,
please feel free to open a GitHub Issue.

Using GitHub Issues helps keep discussions public and useful
for others exploring similar RAG systems.
