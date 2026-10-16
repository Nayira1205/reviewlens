cat > README.md << 'EOF'
# ReviewLens
Aspect-based sentiment analysis of e-commerce product reviews
(Electronics and Food/Agriculture) - CSE4039 Advanced Text & Social Media Analytics lab project.

Status: in progress
EOF

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet
```