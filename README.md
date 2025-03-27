<div align = "center">
<h1>
QUDS: Quran Deep Search
<br>
</h1>

QUDS is a tool for searching the Quran using deep learning techniques. It allows users to search for specific ayas or topics in the Quran using natural language. It also supports fuzzy search for keyword matching.
</div>

# The Idea
Rather than searching in the text of the Quran itself, we search in multiple explanations of the Quran "tafsirs", combined with fuzzy search in the Quran Ayas.
This allows us to get closer to the meaning and get closer to the topic we are looking for along with decent keyword matching.

# Embedding Models
Theoretically, we can use any embedding or reranking models for this task. I really encourage you to try different models especially models with open weights.

But the best one I found right now is Embed 3 and Rerank 3 by [Cohere AI](https://cohere.com) since their performance in Arabic is unmatched!

# Usage
You can use the free hosted app from [here](https://quds.onrender.com) but note that it's quite limited. You may also wait for `30-40` seconds until the server responds.

## Local Setup
1. Install the requirements using
```bash
pip install -r requirements.txt
```
2. Download the vector database from [here](https://drive.google.com/drive/folders/1KyitRoB2UxeSkwKfxIRvA1IzF7PJWSOX?usp=share_link)
3. In your environment set `cohere_api` to your API key.
4. Run the flask app using
```bash
python3 app.py
```
5. Open the UI or connect to the `/search` endpoint using `curl` or any other tool.

# Multilingual possibility
Since QUDS is based on explanation of the Quran, we can use any tafsir in any language to create a vector database if we have a good embedding model for that language. More on that later isa!

# Future Work (All Contributions are so welcome 🤝)
- I will release the full code to create the vector database using any embedding model / any tafsir.
- I will try to implement a better way combining the semantic search and the fuzzy search. The current implementations use exact key word matching. But I find the "fuzzy" component is very important.
- Optimizations to the search process itself.
- UI Improvements.

# Resources
- [data-quran](https://github.com/hablullah/data-quran) Repo