# Security Policy

## Supported Versions

We actively provide security fixes, bug patches, and dependency updates for the following versions:

| Version | Supported          |
| :--- | :---: |
| `1.0.x` | :white_check_mark: |
| `< 1.0` | :x:                |

---

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please report it responsibly so we can address it before public disclosure.

### How to Report

Please **do not** report security vulnerabilities via public GitHub issues, discussions, or pull requests.

Instead, please use one of the following private reporting methods:

1. **GitHub Private Vulnerability Reporting (Recommended)**:
   - Navigate to the **Security** tab of the repository on GitHub.
   - Click **Report a vulnerability** to open an advisory draft directly with the maintainers.

2. **Email**:
   - Send an email to the project maintainer at `thisiskamaljoshi@gmail.com` with the subject line `[SECURITY] Simple Bigram Model Vulnerability Report`.

### What to Include in Your Report

To help us investigate and triage the issue quickly, please provide:
- A clear description of the vulnerability and its potential impact.
- Step-by-step instructions or a minimal Proof of Concept (PoC) script to reproduce the issue.
- Details regarding your environment (Python version, operating system, package dependencies).
- Any proposed remediation or patch, if available.

---

## Expected Response Timeline (Best-Effort)

This is an open-source educational project maintained on personal time. While there are no commercial SLAs, maintainers aim for the following best-effort targets:

- **Initial Acknowledgment**: Typically within 48 to 72 hours.
- **Triage & Assessment**: Best-effort review within 1–2 weeks as personal schedule permits.
- **Fix & Disclosure**: Coordinated release schedule and credit in release notes.


---

## Security Scope & Threat Model

This repository is an educational and research-grade statistical language model. Key security considerations include:

- **Model Serialization & Deserialization**:
  The project utilizes standard JSON format for model persistence (`serializer.py`) rather than arbitrary code execution vectors like Python's `pickle`. The serializer strictly validates JSON structure, keys, and numeric probability types to guard against malformed data.
- **Resource Exhaustion**:
  Input text processing, token generation loops, and numerical calculations enforce boundary safeguards (`max_tokens`, bounded log operations, and recursion protection) to prevent denial-of-service conditions.
- **Dependency Hygiene**:
  Dependencies are pinned and audited to minimize supply chain attack surfaces.
