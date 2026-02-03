# The 60-Minute RAG Blueprint

**A framework-free RAG reference architecture.** Built with Python, Supabase, and Koyeb. Designed for clarity, not bloat.

Get from zero to deployed in **under 60 minutes**. Built by [Claudia Ng](https://substack.com/@claudiang1) for [AI Weekender](https://aiweekender.substack.com).

## ⚠️ Setup Guide Required

**This code is fully functional, but requires the setup guide to run.**

The database schema (essential for vector search) is not included in this repo. It's available in the [Substack post](https://open.substack.com/pub/aiweekender/p/rag-starter-kit-from-zero-to-live?r=2nz2lx&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true), along with:

- Complete database schema with explanations
- Step-by-step setup (Supabase, OpenAI, deployment)
- Architecture decisions explained
- Troubleshooting guide
- Code walkthroughs

---

## 🚀 What You'll Build

**With the premium guide, you'll:**

- Deploy a working RAG system in under 60 minutes
- Understand how every component works (not just copy-paste)
- Learn practical AI engineering skills by building
- Have a portfolio-ready foundation you can customize

**The guide turns this from "code I can't get working" into "a system I understand and can modify."**

---

## 💰 Premium Guide Includes

1. **Complete Setup Instructions**
   - Step-by-step Supabase setup
   - OpenAI configuration
   - Local environment setup
   - Koyeb deployment

2. **Architecture Decisions Explained**
   - Why each technology choice
   - Tradeoffs and alternatives
   - Cost vs quality considerations

3. **Code Walkthroughs**
   - Ingestion pipeline (chunking, embeddings, database)
   - Retrieval and generation
   - Vector similarity search

4. **Troubleshooting**
   - Common errors with fixes
   - Debugging strategies

---

## 📁 What's in the Repo

A professional reference implementation optimized for clarity:

- **Framework-free architecture** - See how RAG works without abstraction layers
- **Clean structure** - Understand how every component connects
- **Educational code** - Comments explain engineering tradeoffs and decisions
- **Production patterns** - Error handling, validation, configuration management
- **Deployment-ready** - Docker and Koyeb configs included

**How it works:** Add Markdown files to `./data/`, run the ingestion script, and you'll have a searchable knowledge base. Clone it and make it your own.

---

## 🎯 Who This Is For

- Engineers who want a clean reference implementation without framework bloat
- Data Scientists moving from theory to practice
- Content creators making their writing searchable
- Builders wanting a production-ready baseline to adapt
- Portfolio builders demonstrating AI engineering skills

**Not for:** Complex enterprise architectures, framework comparisons, or video courses. This is about **code**, not lectures.

**Why framework-free?** When you understand the pieces, you can adapt them. This blueprint shows you the architecture, not just how to use someone else's abstraction.

---

## 🚀 Quick Start

1. **[Read the Setup Guide](https://open.substack.com/pub/aiweekender/p/rag-starter-kit-from-zero-to-live?r=2nz2lx&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)**

2. Follow the setup instructions

3. Deploy your RAG system and customize it

The guide includes everything: Supabase setup, OpenAI configuration, local testing, and Koyeb deployment. **Under 60 minutes to a working RAG system.**

---

## 📁 Project Structure

```
rag-starter-kit/
├── src/
│   ├── app.py              # Streamlit chat UI + RAG logic
│   ├── ingest.py           # Data processing script
│   ├── parse_substack.py   # Substack RSS/HTML parsing
│   ├── config.py           # Configuration
│   └── validate_setup.py   # Setup validation
├── data/                    # Your Markdown files go here
├── requirements.txt
├── Dockerfile
├── Procfile
└── .env.example

Note: schema.sql is in the premium guide.
```

---

## 📰 Substack RSS Ingestion (Upgrade)

Use this when your knowledge base lives in a Substack publication.

**Parse feed to Markdown:**
```
python src/parse_substack.py --feed-url https://aiweekender.substack.com/feed --fetch-full-html
```

**Filter recent posts:**
```
python src/parse_substack.py --feed-url https://aiweekender.substack.com/feed --since-days 30
```

**Dry-run (print cleaned samples only):**
```
python src/parse_substack.py --feed-url https://aiweekender.substack.com/feed --dry-run
```

**Then ingest the generated Markdown:**
```
python src/ingest.py
```

**Intended data flow:**
1. `parse_substack.py` cleans RSS/HTML and writes `.md` files to `./data/substack/` (creates the folder if missing).
2. `ingest.py` reads those Markdown files and upserts embeddings into Supabase.

**Optional environment variables:**
- `SUBSTACK_PUBLICATION_NAME` (used to build `https://<name>.substack.com/feed`)
- `SUBSTACK_FEED_URL` (explicit feed URL override)

**Proof to capture during testing:**
- Raw RSS content vs cleaned text (before/after)
- A citation mismatch example, then the fixed citation after re-ingest

---

## 💡 The Full Loop

Most people get stuck because they never build the *full loop:* ingest → retrieve → generate → deploy.

Once you see it end-to-end, you **stop feeling like you're "playing with LLMs" and start thinking like an engineer who ships AI systems.**

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Use it however you want! Modify it, build on it, use it commercially. Just include the original license notice if you redistribute it.

---

## 🚀 Ready to Build?

**[See the setup guide](https://open.substack.com/pub/aiweekender/p/rag-starter-kit-from-zero-to-live?r=2nz2lx&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) and go from zero to deployed in under an hour.**

[Subscribe to AI Weekender →](https://aiweekender.substack.com)
