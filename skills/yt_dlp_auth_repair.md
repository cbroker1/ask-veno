# Skill: Repair yt-dlp Auth and Cookie Failures

## Purpose

Use this skill when a YouTube audio ingestion job fails because `yt-dlp` appears to need authentication, browser cookies, refreshed cookies, or a manual browser login.

This skill is designed for the Hermes Self-Healing YouTube RAG Agent pipeline.

## When to use this skill

Use this skill when a pipeline step fails with symptoms such as:

- `failed_auth`
- `auth_or_cookie`
- `Sign in to confirm`
- `confirm you are not a bot`
- `cookies`
- `authentication`
- `HTTP Error 403`
- `HTTP Error 401`
- `This video may be inappropriate`
- `This video is unavailable`
- `yt-dlp` extraction failure that succeeds after adding browser cookies

## Safety rules

Never print, summarize, copy, or commit:

- browser cookies
- `cookies.txt`
- `.env`
- browser profile files
- auth tokens
- raw cookie export contents

Treat cookies as secrets.

Do not attempt to bypass CAPTCHA or access controls. If manual login is required, ask the user to complete login in their browser.

## Repair procedure

### 1. Inspect pipeline status

Run:

```bash
python scripts/pipeline_status.py
