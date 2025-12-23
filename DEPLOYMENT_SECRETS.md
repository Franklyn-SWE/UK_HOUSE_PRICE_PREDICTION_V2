Deployment secrets — OpenAI API key

This file shows safe ways to store and use your OpenAI API key for deployments. Do NOT commit real keys to the repository.

Quick checklist
- Revoke any exposed key and create a new one in the OpenAI dashboard.
- Store the new key in your deployment's secret manager or as an environment variable.
- Do not commit `.streamlit/secrets.toml` or other secret files.

Streamlit Cloud
- In Streamlit Cloud, go to your app → Settings → Secrets and add:
  OPENAI_API_KEY = "sk-..."
- Streamlit will expose this as `st.secrets["OPENAI_API_KEY"]` server-side.

Docker / Linux server
- Set the env var for the process (example systemd or docker run):
  ```bash
  export OPENAI_API_KEY="sk-..."
  streamlit run app.py
  ```
- Or with Docker:
  ```bash
  docker run -e OPENAI_API_KEY="sk-..." your-image
  ```

GitHub Actions (CI/CD)
- Add the secret in the repository Settings → Secrets → Actions as `OPENAI_API_KEY`.
- Reference it in workflow:
  ```yaml
  - name: Deploy
    env:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    run: |
      streamlit run app.py
  ```

Heroku
- In the Heroku dashboard, go to Settings → Config Vars and add `OPENAI_API_KEY`.

Azure App Service
- In Deployment Center or Configuration → Application settings add `OPENAI_API_KEY`.

AWS
- Use Secrets Manager or Parameter Store and inject the value into your container/task as an env var named `OPENAI_API_KEY`.

Local dev (safe)
- Create a local `.streamlit/secrets.toml` (and ensure `.streamlit/` is in `.gitignore`):
  ```toml
  OPENAI_API_KEY = "sk-..."
  ```
- Or use a local `.env` and a loader (do NOT commit).

Verify & test
1. Start your app in the deployment environment.
2. Confirm `get_explanation_service()` can read the key (no client-side exposure).
3. Make one explanation request and monitor OpenAI usage in the dashboard.

Extras / hardening
- Set usage/billing alerts in OpenAI dashboard.
- Limit in-app usage per user and add caching (already implemented).
- For production, use a managed secret store (AWS Secrets Manager, Azure Key Vault, etc.) and assign least privilege.

If you want, I can add a short section to `README.md` linking to this document and showing how to run locally with a placeholder key. Let me know.