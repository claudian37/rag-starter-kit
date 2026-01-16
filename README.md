# RAG Starter Kit

An opinionated, working RAG system designed to get you from a blank screen to a deployed prototype in **under an hour**. This kit reflects the tradeoffs made when building real systems under time and reliability constraints in production environments.

## ⚠️ This Repository Requires the Premium Setup Guide

**This code is fully functional, but you cannot run it without the premium guide.**

The repository contains production-ready code with extensive comments, but **the database schema is not included**. The schema (which sets up the vector database, indexes, and search functions) is essential—without it, the system won't work.

The [premium Substack post](https://aiweekender.substack.com/p/rag-starter-kit) (paid subscribers only) includes:
- **Complete database schema** with line-by-line explanations
- Step-by-step setup instructions
- Architecture decisions explained
- Troubleshooting guide
- Code walkthroughs

---

## 🚀 The Transformation

**With the premium guide, you'll:**

- Deploy a working RAG system in under 60 minutes
- Ingest your own content (add Markdown files to create your knowledge base)
- Understand how every component works (not just copy-paste)
- Learn practical AI engineering skills by building
- Have a foundation you can customize and add to your portfolio

**The guide turns this from "code I can't get working" into "a system I understand and can modify."**

You'll learn by doing: setting up Supabase, configuring embeddings, understanding vector search, and deploying to production. These are real skills that transfer to any RAG system.

---

## 💰 What You Get With the Premium Guide

The [premium setup guide](https://aiweekender.substack.com/p/rag-starter-kit) includes:

1. **Complete Setup Instructions**
   - Step-by-step Supabase setup (which buttons to click, what to copy where)
   - OpenAI configuration and billing setup
   - Local environment setup with troubleshooting
   - How to add your own Markdown files and ingest them
   - Koyeb deployment with exact steps

2. **Architecture Decisions Explained**
   - Why each technology choice (Supabase, Koyeb, specific models)
   - When to use alternatives and tradeoffs
   - Cost vs quality considerations

3. **Code Walkthroughs**
   - How the ingestion pipeline works (chunking, embeddings, database)
   - How retrieval and generation work together
   - Understanding vector similarity search

4. **Troubleshooting**
   - Common errors with specific fixes
   - Debugging strategies for each component

**The guide gets you from zero to a working, minimal RAG system in 60 minutes, with understanding—not just copy-paste.**

---

## 📁 What's in the Repo

This repository contains working code optimized for learning:

- **Flat structure** - No deep abstractions, you can see how everything connects
- **Educational code** - Every function includes comments explaining engineering tradeoffs
- **Production patterns** - Error handling, validation, and configuration management
- **Deployment-ready** - Docker and Koyeb configs included
- **Your content** - Add your own Markdown files to `./data/` to build your knowledge base

The code is designed to be read and understood. You'll need the guide to get it working and understand the "why" behind each decision.

**How it works:** Add your Markdown files to the `./data/` directory, run the ingestion script, and you'll have a searchable knowledge base of your content.

---

## 🎯 Who This Is For

- **Engineers & Data Scientists** who want to move from theory to a tangible AI asset
- **Content creators** who want to make their writing/blog posts searchable
- **Builders** who want a clean baseline they can adapt to their own data
- **Portfolio builders** who want to demonstrate real AI engineering skills
- **People who learn by building** - get practical skills by deploying a working system

**Add your own content:** Drop Markdown files into `./data/` and you'll have a RAG system powered by your knowledge base. Perfect for documentation, blog posts, research notes, or any text content you want to make searchable.

You can tweak this for your own use case, add it to your portfolio, and use it as a foundation for more complex systems.

## 🚫 Who This Is Not For

- If you want a complex, multi-agent "Enterprise" architecture
- If you enjoy endlessly comparing 5 different vector databases
- If you want a 10-hour video course. This is about **code**, not lectures

---

## 🚀 Quick Start

1. **Get the Premium Guide:** [Subscribe and access the complete walkthrough](https://aiweekender.substack.com/p/rag-starter-kit)

2. **Follow the setup instructions** in the guide

3. **Deploy your RAG system** and start customizing it

The guide includes everything: Supabase setup, OpenAI configuration, local testing, and Koyeb deployment. By the end of 60 minutes, you'll have a working RAG system and a knowledge base of your chosen content.

---

## 📁 Project Structure

```
rag-starter-kit/
├── src/
│   ├── app.py              # Streamlit chat UI + RAG logic
│   ├── ingest.py           # Data processing script
│   ├── config.py            # Configuration (models, chunk size, thresholds)
│   └── validate_setup.py   # Setup validation script
├── data/                    # Your Markdown files go here
├── requirements.txt         # Python dependencies
├── Dockerfile               # Universal container (works on Koyeb, Railway, Render, etc.)
├── Procfile                 # Fallback run command (works on Heroku, some platforms)
├── runtime.txt              # Python version for buildpack deployments
└── .env.example             # Environment variable template

**Note:** `schema.sql` is not included in this repository. It's available in the premium guide with complete explanations.
```

---

## 💡 The Full Loop

Most people get stuck because they never build the *full loop:* ingest → retrieve → generate → deploy.

Once you see it end-to-end, you **stop feeling like you're "playing with LLMs" and start thinking like an engineer who ships AI systems**.

The premium guide gets you through that full loop in under 60 minutes, with understanding—not just copy-paste.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

This is a starter kit - use it however you want! Modify it, build on it, use it commercially. The only requirement is that you include the original license notice if you redistribute it.

## 🙏 Credits

Built as a simplified, educational version of production RAG systems. Every function includes comments explaining the engineering tradeoffs.

---

## 🚀 Ready to Build?

**Get the [premium setup guide](https://aiweekender.substack.com/p/rag-starter-kit) and go from zero to deployed in under an hour.**

[Get the Premium Guide →](https://aiweekender.substack.com/p/rag-starter-kit)
